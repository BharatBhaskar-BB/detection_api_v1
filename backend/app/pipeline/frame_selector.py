"""Step 3: Smart frame selection — dense sampling + quality filter + CLIP diversity."""

import cv2
import numpy as np
from dataclasses import dataclass
from loguru import logger

from app.config import get_settings


@dataclass
class SelectedFrame:
    frame_index: int        # index in all_frames[]
    timestamp_s: float      # seconds into the video
    frame: np.ndarray       # the image (BGR, 480px wide)
    quality_score: float    # blur score (higher = sharper)


class FrameSelector:
    """Select diverse, high-quality frames from a video for LLM analysis."""

    def __init__(self):
        settings = get_settings()
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_device = None
        self._sample_interval_s = getattr(settings, "FRAME_SAMPLE_INTERVAL_S", 5)
        self._blur_threshold = getattr(settings, "FRAME_BLUR_THRESHOLD", 50.0)
        self._face_area_threshold = getattr(settings, "FRAME_FACE_AREA_THRESHOLD", 0.25)
        self._edge_threshold = getattr(settings, "FRAME_EDGE_THRESHOLD", 10.0)
        self._clip_model_name = settings.CLIP_MODEL

    def _get_clip(self):
        if self._clip_model is None:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load(
                self._clip_model_name, device=device
            )
            self._clip_device = device
        return self._clip_model, self._clip_preprocess, self._clip_device

    # ── Quality Filters ──

    def _blur_score(self, frame: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def _edge_density(self, frame: np.ndarray) -> float:
        """Fraction of pixels that are edges — low = blank wall/ceiling."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges.mean()

    def _face_fraction(self, frame: np.ndarray) -> float:
        """Fraction of frame area occupied by faces."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use Haar cascade (fast, good enough for filtering)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3, minSize=(60, 60))
        if len(faces) == 0:
            return 0.0
        total_face_area = sum(w * h for (_, _, w, h) in faces)
        frame_area = frame.shape[0] * frame.shape[1]
        return total_face_area / frame_area

    # ── CLIP Embeddings ──

    def _compute_embeddings(self, frames: list[np.ndarray]) -> np.ndarray:
        """Compute CLIP embeddings for a batch of frames. Returns (N, D) array."""
        import torch
        from PIL import Image

        model, preprocess, device = self._get_clip()
        embeddings = []

        # Process in batches of 32 for memory efficiency
        batch_size = 32
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            images = []
            for frame in batch:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                images.append(preprocess(img))

            image_tensor = torch.stack(images).to(device)
            with torch.no_grad():
                embs = model.encode_image(image_tensor)
                embs = embs / embs.norm(dim=-1, keepdim=True)
            embeddings.append(embs.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def _mmr_select(self, embeddings: np.ndarray, budget: int,
                    lambda_: float = 0.5,
                    temporal_anchor_frac: float = 0.25) -> list[int]:
        """
        Maximal Marginal Relevance selection for diversity.
        Returns indices into the embeddings array.

        MMR balances:
        - Relevance: prefer frames with high quality scores (not used here,
          all candidates passed quality filter, so equal relevance)
        - Diversity: prefer frames different from already-selected ones

        With lambda_=0.5. Lower lambda_ = more diversity.

        temporal_anchor_frac: fraction of budget reserved as temporal anchors
        (evenly-spaced across the timeline) so that spatially-similar rooms
        that appear at different times are never fully dropped by MMR.
        """
        n = len(embeddings)
        if n <= budget:
            return list(range(n))

        # ── Temporal anchors: guarantee even coverage across the timeline ──
        n_anchors = max(2, int(budget * temporal_anchor_frac))
        anchor_indices = [
            int(round(i * (n - 1) / (n_anchors - 1))) for i in range(n_anchors)
        ]
        # de-duplicate while preserving order
        seen = set()
        anchors = []
        for a in anchor_indices:
            if a not in seen:
                anchors.append(a)
                seen.add(a)

        # Similarity matrix
        sim_matrix = embeddings @ embeddings.T  # (N, N) cosine similarity

        selected = list(anchors)
        remaining = set(range(n)) - set(anchors)

        # Fill remaining budget with MMR
        mmr_budget = budget - len(selected)

        # Start with the frame most different from the average (most "unique")
        if not selected:
            avg_sim = sim_matrix.mean(axis=1)
            first = int(np.argmin(avg_sim))
            selected.append(first)
            remaining.discard(first)
            mmr_budget -= 1

        for _ in range(mmr_budget):
            if not remaining:
                break
            best_idx = -1
            best_score = -float("inf")

            for idx in remaining:
                # Max similarity to any already-selected frame
                max_sim_to_selected = max(sim_matrix[idx, s] for s in selected)
                # MMR score: lower similarity to selected = higher score
                score = -lambda_ * max_sim_to_selected
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                break
            selected.append(best_idx)
            remaining.remove(best_idx)

        return selected

    # ── Main Selection Pipeline ──

    def select(
        self,
        all_frames: list[np.ndarray],
        fps: float = 30.0,
        stride: int = 30,
        duration_s: float | None = None,
    ) -> list[SelectedFrame]:
        """
        Select diverse, high-quality frames from the video.

        Args:
            all_frames: All sampled frames (stride=30)
            fps: Original video FPS
            stride: Frame sampling stride used
            duration_s: Video duration in seconds (auto-calculated if None)

        Returns:
            List of SelectedFrame with timestamps, sorted by time.
        """
        n = len(all_frames)
        if n == 0:
            return []

        if duration_s is None:
            duration_s = n * stride / fps

        # ── 3a. Dense temporal sampling (every N seconds) ──
        # For short videos, reduce interval to keep more frames
        interval_s = self._sample_interval_s
        if duration_s < 30:
            interval_s = max(1, interval_s // 2)  # halve for short clips
        sample_every_n = max(1, int(interval_s * fps / stride))
        candidate_indices = list(range(0, n, sample_every_n))
        logger.info(f"Dense sample: {len(candidate_indices)} candidates "
                    f"from {n} frames (every {interval_s}s)")

        # ── 3b. Quality filter ──
        good_indices = []
        reject_blur = 0
        reject_face = 0
        reject_edge = 0

        for idx in candidate_indices:
            frame = all_frames[idx]

            # Blur check
            blur = self._blur_score(frame)
            if blur < self._blur_threshold:
                reject_blur += 1
                continue

            # Edge density check (blank wall/ceiling/floor)
            edges = self._edge_density(frame)
            if edges < self._edge_threshold:
                reject_edge += 1
                continue

            # Face check (person talking to camera)
            face_frac = self._face_fraction(frame)
            if face_frac > self._face_area_threshold:
                reject_face += 1
                continue

            good_indices.append((idx, blur))

        logger.info(f"Quality filter: {len(good_indices)} passed, "
                    f"rejected {reject_blur} blur, {reject_face} face, "
                    f"{reject_edge} low-edge")

        min_frames = min(3, len(candidate_indices))
        if len(good_indices) < min_frames:
            # Fallback: too few high-quality frames → send all to avoid missing items
            logger.warning(f"Only {len(good_indices)} frames passed quality "
                          f"(min {min_frames}) — falling back to unfiltered")
            good_indices = [(idx, self._blur_score(all_frames[idx]))
                           for idx in candidate_indices]

        # ── 3c. CLIP diversity selection ──
        duration_min = duration_s / 60
        # Scale budget generously: 8 frames/min, min 60, max 150
        # Higher cap ensures multi-room walkthroughs don't lose entire rooms
        budget = max(60, min(150, int(duration_min * 8)))
        logger.info(f"Frame budget: {budget} (video={duration_min:.1f}min)")

        if len(good_indices) <= budget:
            # All good frames fit within budget
            selected_pairs = good_indices
        else:
            # Compute CLIP embeddings and select diverse subset
            good_frames = [all_frames[idx] for idx, _ in good_indices]
            embeddings = self._compute_embeddings(good_frames)
            logger.info(f"CLIP embeddings computed: {embeddings.shape}")

            mmr_indices = self._mmr_select(embeddings, budget)
            selected_pairs = [good_indices[i] for i in mmr_indices]

        # ── Build output sorted by timestamp ──
        results = []
        for idx, blur in selected_pairs:
            timestamp_s = idx * stride / fps
            results.append(SelectedFrame(
                frame_index=idx,
                timestamp_s=timestamp_s,
                frame=all_frames[idx],
                quality_score=blur,
            ))

        results.sort(key=lambda f: f.timestamp_s)
        logger.info(f"Selected {len(results)} frames, "
                    f"spanning {results[0].timestamp_s:.0f}s - {results[-1].timestamp_s:.0f}s")
        return results
