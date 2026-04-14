"""Two-stage scene detection: Histogram+SSIM (cheap) → CLIP (semantic confirm)."""

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings


def compute_ssim_gray(img1: np.ndarray, img2: np.ndarray) -> float:
    """Simplified SSIM on grayscale images."""
    C1, C2 = 6.5025, 58.5225
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(ssim_map.mean())


class SceneDetector:
    """Detect distinct scenes in a sequence of frames."""

    def __init__(self):
        settings = get_settings()
        self.hist_threshold = settings.SCENE_HIST_THRESHOLD
        self.ssim_threshold = settings.SCENE_SSIM_THRESHOLD
        self.clip_threshold = settings.SCENE_CLIP_THRESHOLD
        self.min_scene_frames = settings.SCENE_MIN_FRAMES
        self.max_scenes = settings.SCENE_MAX_SCENES
        self._clip_model = None
        self._clip_preprocess = None

    def _get_clip(self):
        """Lazy-load CLIP model."""
        if self._clip_model is None:
            import clip
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model, self._clip_preprocess = clip.load(
                get_settings().CLIP_MODEL, device=device
            )
            self._device = device
        return self._clip_model, self._clip_preprocess, self._device

    def _clip_embed(self, frame: np.ndarray) -> np.ndarray:
        """Compute CLIP embedding for a frame."""
        import torch
        from PIL import Image

        model, preprocess, device = self._get_clip()
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            emb = model.encode_image(preprocess(img).unsqueeze(0).to(device))
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def detect(self, frames: list[np.ndarray]) -> list[dict]:
        """
        Run two-stage scene detection.
        Returns list of {"frame_index": int, "frame": np.ndarray}.
        """
        if len(frames) == 0:
            return []
        if len(frames) == 1:
            return [{"frame_index": 0, "frame": frames[0]}]

        n = len(frames)

        # ── Short video fast path ──
        # For short videos (≤ max_scenes * 3 frames), skip both stages —
        # pick up to max_scenes evenly-spaced frames as representatives.
        # This avoids CLIP killing valid within-room angle changes.
        if n <= self.max_scenes * 3:
            num_scenes = min(n, self.max_scenes)
            step = n / num_scenes
            indices = [int(i * step + step / 2) for i in range(num_scenes)]
            # Clamp to valid range
            indices = [min(idx, n - 1) for idx in indices]
            scenes = [{"frame_index": idx, "frame": frames[idx]} for idx in indices]
            logger.info(f"Short video ({n} frames): selected {len(scenes)} evenly-spaced scenes")
            return scenes

        # ── Stage A: Cheap pixel filter ──
        candidates = [0]
        for i in range(1, len(frames)):
            prev = cv2.resize(frames[i - 1], (128, 128))
            curr = cv2.resize(frames[i], (128, 128))

            # Histogram difference
            hist_prev = cv2.calcHist([prev], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist_curr = cv2.calcHist([curr], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            cv2.normalize(hist_prev, hist_prev)
            cv2.normalize(hist_curr, hist_curr)
            hist_diff = 1.0 - cv2.compareHist(hist_prev, hist_curr, cv2.HISTCMP_CORREL)

            # SSIM difference
            gray_prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
            gray_curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            ssim_diff = 1.0 - compute_ssim_gray(gray_prev, gray_curr)

            if hist_diff > self.hist_threshold or ssim_diff > self.ssim_threshold:
                candidates.append(i)

        logger.info(f"Stage A: {len(candidates)} candidates from {len(frames)} frames")

        # ── Stage B: CLIP semantic confirmation ──
        confirmed_boundaries = [0]
        last_emb = self._clip_embed(frames[0])

        for idx in candidates[1:]:
            emb = self._clip_embed(frames[idx])
            distance = 1.0 - float(np.dot(emb, last_emb))
            if distance > self.clip_threshold:
                confirmed_boundaries.append(idx)
                last_emb = emb

        logger.info(f"Stage B: {len(confirmed_boundaries)} confirmed from {len(candidates)} candidates")

        # ── Build scenes, pick middle frame ──
        scenes = []
        for i in range(len(confirmed_boundaries)):
            start = confirmed_boundaries[i]
            end = confirmed_boundaries[i + 1] if i + 1 < len(confirmed_boundaries) else len(frames)
            if (end - start) >= self.min_scene_frames:
                mid = (start + end) // 2
                scenes.append({"frame_index": mid, "frame": frames[mid]})

        # Guardrails
        if not scenes:
            scenes = [{"frame_index": 0, "frame": frames[0]}]
        if len(scenes) > self.max_scenes:
            # Keep most diverse via even spacing
            step = len(scenes) / self.max_scenes
            scenes = [scenes[int(i * step)] for i in range(self.max_scenes)]

        logger.info(f"Final: {len(scenes)} scene representative frames")
        return scenes
