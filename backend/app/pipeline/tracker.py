"""CLIP-appearance tracker — two-stage association with Hungarian assignment.

At stride-30 objects move hundreds of pixels between frames, making IoU
unreliable.  This tracker uses CLIP crop embeddings as the primary
association signal (appearance doesn't change with position) combined with
spatial cues after CMC.

Cost = 0.60 * (1 - clip_similarity)   # appearance is king
     + 0.25 * centroid_distance/diag  # spatial proximity (after CMC warp)
     + 0.10 * (1 - area_ratio)        # size consistency
     + 0.05 * (1 - bbox_iou)          # small IoU bonus when available

Label-gated: only same-label track↔detection pairs can match.
Two-stage: high-confidence detections matched first, then low-confidence.
"""

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from scipy.optimize import linear_sum_assignment


# ── Track class (unchanged interface) ────────────────────────────────────────


class Track:
    """A tracked object spanning multiple frames."""

    _next_id = 1

    def __init__(self, label: str, bbox: list[float], frame_idx: int, confidence: float = 1.0):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = label
        self.bboxes: list[tuple[int, list[float]]] = [(frame_idx, bbox)]
        self.confidence = confidence
        self.lost_frames = 0
        self.is_active = True
        self.embedding: np.ndarray | None = None  # CLIP embedding (512-d, L2-normed)

    @property
    def last_bbox(self) -> list[float]:
        return self.bboxes[-1][1]

    @property
    def last_frame(self) -> int:
        return self.bboxes[-1][0]

    @property
    def first_frame(self) -> int:
        return self.bboxes[0][0]

    @property
    def frame_count(self) -> int:
        return len(self.bboxes)

    def update(self, bbox: list[float], frame_idx: int, embedding: np.ndarray | None = None) -> None:
        self.bboxes.append((frame_idx, bbox))
        self.lost_frames = 0
        self.is_active = True
        if embedding is not None:
            # Exponential moving average of embeddings for stability
            if self.embedding is not None:
                self.embedding = 0.7 * self.embedding + 0.3 * embedding
                self.embedding /= np.linalg.norm(self.embedding) + 1e-8
            else:
                self.embedding = embedding


def compute_iou(box1: list[float], box2: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


# ── CLIP singleton ───────────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device: str = "cpu"


def _ensure_clip():
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return True
    try:
        import clip as clip_module
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        _clip_model, _clip_preprocess = clip_module.load("ViT-B/32", device=device)
        _clip_model.eval()
        _clip_device = device
        logger.info(f"Tracker CLIP ViT-B/32 loaded on {device}")
        return True
    except Exception as e:
        logger.warning(f"CLIP unavailable for tracker — falling back to spatial-only: {e}")
        return False


def _crop_embedding(frame_bgr: np.ndarray, bbox: list[float], pad: float = 0.1) -> np.ndarray | None:
    """Extract a 512-d L2-normalised CLIP embedding from a bbox crop."""
    if _clip_model is None:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    # Skip tiny crops
    if bw < 10 or bh < 10:
        return None
    px, py = bw * pad, bh * pad
    x1i = max(0, int(x1 - px))
    y1i = max(0, int(y1 - py))
    x2i = min(w, int(x2 + px))
    y2i = min(h, int(y2 + py))
    crop = frame_bgr[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return None
    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = _clip_preprocess(pil_img).unsqueeze(0).to(_clip_device)
    with torch.no_grad():
        feat = _clip_model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().flatten().astype(np.float32)


def _batch_crop_embeddings(frame_bgr: np.ndarray, bboxes: list[list[float]], pad: float = 0.1) -> list[np.ndarray | None]:
    """Batch-compute CLIP embeddings for a list of bbox crops — much faster than one-by-one."""
    if _clip_model is None or not bboxes:
        return [None] * len(bboxes)

    h, w = frame_bgr.shape[:2]
    tensors = []
    valid_indices = []

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 10:
            continue
        px, py = bw * pad, bh * pad
        x1i = max(0, int(x1 - px))
        y1i = max(0, int(y1 - py))
        x2i = min(w, int(x2 + px))
        y2i = min(h, int(y2 + py))
        crop = frame_bgr[y1i:y2i, x1i:x2i]
        if crop.size == 0:
            continue
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensors.append(_clip_preprocess(pil_img))
        valid_indices.append(i)

    results: list[np.ndarray | None] = [None] * len(bboxes)
    if not tensors:
        return results

    batch = torch.stack(tensors).to(_clip_device)
    with torch.no_grad():
        feats = _clip_model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    feats_np = feats.cpu().numpy().astype(np.float32)

    for idx, vi in enumerate(valid_indices):
        results[vi] = feats_np[idx]

    return results


# ── Camera Motion Compensation ───────────────────────────────────────────────

_orb = cv2.ORB_create(nfeatures=500)
_bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def estimate_camera_motion(
    prev_frame: np.ndarray, curr_frame: np.ndarray
) -> np.ndarray | None:
    """Estimate 2D affine transform from prev_frame to curr_frame using ORB + RANSAC."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    h, w = prev_gray.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    if scale < 1.0:
        small_h, small_w = int(h * scale), int(w * scale)
        prev_small = cv2.resize(prev_gray, (small_w, small_h))
        curr_small = cv2.resize(curr_gray, (small_w, small_h))
    else:
        prev_small = prev_gray
        curr_small = curr_gray
        scale = 1.0

    kp1, des1 = _orb.detectAndCompute(prev_small, None)
    kp2, des2 = _orb.detectAndCompute(curr_small, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None

    matches = _bf_matcher.knnMatch(des1, des2, k=2)
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 6:
        return None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    if scale < 1.0:
        pts1 /= scale
        pts2 /= scale

    affine, inliers = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0
    )

    if affine is None or inliers is None:
        return None
    if np.sum(inliers) / len(inliers) < 0.3:
        return None
    return affine


def warp_bbox(bbox: list[float], affine: np.ndarray) -> list[float]:
    """Warp a [x1, y1, x2, y2] bbox through a 2×3 affine transform."""
    corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])
    warped = (affine @ pts.T).T
    return [float(warped[:, 0].min()), float(warped[:, 1].min()),
            float(warped[:, 0].max()), float(warped[:, 1].max())]


# ── Cost matrix helpers ──────────────────────────────────────────────────────

def _centroid(bbox: list[float]) -> np.ndarray:
    return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)


def _area(bbox: list[float]) -> float:
    return max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def _build_cost_matrix(
    tracks: list[Track],
    dets: list[dict],
    det_embeddings: list[np.ndarray | None],
    warped_bboxes: dict[int, list[float]],
    img_diag: float,
    use_clip: bool,
    w_clip: float = 0.20,
    w_centroid: float = 0.50,
    w_area: float = 0.20,
    w_iou: float = 0.10,
    max_centroid_ratio: float = 0.25,
    frame_w: int = 0,
) -> np.ndarray:
    """Build NxM cost matrix with label gating.  INF for label mismatches.

    Weights are spatial-dominant: for moving inventory (static objects +
    panning camera), post-CMC centroid proximity is the most reliable
    signal.  CLIP ViT-B/32 is category-level and can't distinguish
    same-model objects, so it serves only as a tiebreaker.

    max_centroid_ratio: if the post-CMC centroid distance exceeds this
    fraction of the image diagonal, the match is rejected (cost=INF).

    frame_w: frame width for opposite-edge gate.  If the track's RAW
    (un-warped) center was near one edge and the detection is near the
    opposite edge, the match is rejected.  This prevents a panning camera
    from merging distinct objects that enter/exit on opposite sides.
    """
    n, m = len(tracks), len(dets)
    cost = np.full((n, m), 1e5, dtype=np.float32)
    edge_pct = 0.15  # 15% from each edge

    for i, trk in enumerate(tracks):
        warped_bbox = warped_bboxes.get(trk.id)
        trk_bbox = warped_bbox if warped_bbox is not None else trk.last_bbox
        trk_cent = _centroid(trk_bbox)
        trk_area = _area(trk_bbox)
        # Raw (un-warped) bbox and center for edge gate + CMC fallback
        raw_bbox = trk.last_bbox
        raw_cent = _centroid(raw_bbox)
        raw_cent_x = float(raw_cent[0])

        for j, det in enumerate(dets):
            # ── Label gate ──
            if trk.label.lower() != det["label"].lower():
                continue  # cost stays at 1e5

            det_bbox = det["bbox"]
            det_cent = _centroid(det_bbox)
            det_area = _area(det_bbox)
            det_cent_x = float(det_cent[0])

            # ── Opposite-edge gate ──
            # If track was near one edge and detection is near the opposite
            # edge, this is almost certainly a different object entering the
            # frame as the tracked object exits (camera panning).
            if frame_w > 1:
                trk_left = raw_cent_x < frame_w * edge_pct
                trk_right = raw_cent_x > frame_w * (1 - edge_pct)
                det_left = det_cent_x < frame_w * edge_pct
                det_right = det_cent_x > frame_w * (1 - edge_pct)
                if (trk_left and det_right) or (trk_right and det_left):
                    continue  # cost stays at 1e5

            # ── Max centroid distance gate (with CMC fallback) ──
            # CMC only warps by the latest frame-pair affine.  For tracks
            # lost for several frames the warped position can be WRONG
            # (overshoots).  If the warped distance fails, fall back to
            # the raw (un-warped) centroid distance.
            cdist = float(np.linalg.norm(trk_cent - det_cent))
            use_raw_fallback = False
            if cdist > max_centroid_ratio * img_diag and warped_bbox is not None:
                # Warped position failed — try raw
                cdist_raw = float(np.linalg.norm(raw_cent - det_cent))
                if cdist_raw <= max_centroid_ratio * img_diag:
                    cdist = cdist_raw
                    use_raw_fallback = True
                else:
                    continue  # both warped and raw fail → reject
            elif cdist > max_centroid_ratio * img_diag:
                continue  # no warp, raw fails → reject

            # CLIP appearance cost
            if use_clip and trk.embedding is not None and det_embeddings[j] is not None:
                sim = float(np.dot(trk.embedding, det_embeddings[j]))
                clip_cost = 1.0 - sim
            else:
                clip_cost = 0.5  # neutral when unavailable

            # Centroid distance (normalised by image diagonal)
            cent_cost = min(cdist / max(img_diag, 1.0), 1.0)

            # Area ratio — use raw bbox areas when CMC fallback is active
            eff_area = _area(raw_bbox) if use_raw_fallback else trk_area
            mn, mx = min(eff_area, det_area), max(eff_area, det_area)
            area_cost = 1.0 - (mn / mx if mx > 0 else 0.0)

            # IoU — use raw bbox when CMC fallback is active
            eff_bbox = raw_bbox if use_raw_fallback else trk_bbox
            iou = compute_iou(eff_bbox, det_bbox)
            iou_cost = 1.0 - iou

            if use_clip:
                c = w_clip * clip_cost + w_centroid * cent_cost + w_area * area_cost + w_iou * iou_cost
            else:
                # No CLIP — redistribute weight to spatial signals
                c = 0.50 * cent_cost + 0.30 * (1.0 - iou) + 0.20 * area_cost

            cost[i, j] = c

    return cost


def _run_assignment(cost: np.ndarray, threshold: float) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment with threshold gating."""
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    unmatched_t = set(range(n))
    unmatched_d = set(range(m))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] < threshold:
            matches.append((r, c))
            unmatched_t.discard(r)
            unmatched_d.discard(c)

    return matches, sorted(unmatched_t), sorted(unmatched_d)


# ── Main tracker ─────────────────────────────────────────────────────────────


class ByteTracker:
    """
    CLIP-appearance tracker with two-stage Hungarian assignment.

    Replaces IoU-only matching with a fused cost dominated by CLIP
    visual similarity, making it robust to large camera motion at stride-30.
    Falls back to spatial-only matching if CLIP is unavailable.
    """

    def __init__(
        self,
        max_lost: int = 5,
        high_conf_thresh: float = 0.40,
        low_conf_thresh: float = 0.10,
        match_threshold: float = 0.55,
        match_threshold_low: float = 0.65,
    ):
        self.max_lost = max_lost
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.match_threshold = match_threshold
        self.match_threshold_low = match_threshold_low

    def track(
        self,
        frame_detections: list[list[dict]],
        frames: list[np.ndarray] | None = None,
    ) -> list[Track]:
        """
        Track objects across frames.
        Input:
            frame_detections: list of per-frame detections [{"label", "bbox", "confidence"}, ...]
            frames: BGR frames (required for CLIP, optional for spatial-only)
        Output: list of Track objects
        """
        Track._next_id = 1
        use_clip = _ensure_clip() and frames is not None
        use_cmc = frames is not None and len(frames) == len(frame_detections)

        active_tracks: list[Track] = []
        finished_tracks: list[Track] = []
        cmc_applied = 0
        clip_computed = 0

        # Image diagonal for centroid normalisation
        img_diag = 1.0
        if frames is not None and len(frames) > 0:
            h, w = frames[0].shape[:2]
            img_diag = float(np.sqrt(h ** 2 + w ** 2))

        for frame_idx, detections in enumerate(frame_detections):
            # ── CMC ───────────────────────────────────────────────────
            affine = None
            if use_cmc and frame_idx > 0 and active_tracks:
                affine = estimate_camera_motion(frames[frame_idx - 1], frames[frame_idx])
                if affine is not None:
                    cmc_applied += 1

            # Increment lost counter on all active tracks
            for trk in active_tracks:
                trk.lost_frames += 1

            # Warp track bboxes with CMC
            warped: dict[int, list[float]] = {}
            if affine is not None:
                for trk in active_tracks:
                    warped[trk.id] = warp_bbox(trk.last_bbox, affine)

            # ── Compute CLIP embeddings for ALL detections in this frame (batched) ──
            if use_clip and detections:
                all_bboxes = [d["bbox"] for d in detections]
                all_embeddings = _batch_crop_embeddings(frames[frame_idx], all_bboxes)
                clip_computed += sum(1 for e in all_embeddings if e is not None)
            else:
                all_embeddings = [None] * len(detections)

            # ── Split detections by confidence ────────────────────────
            high_indices = [i for i, d in enumerate(detections)
                           if d.get("confidence", 0) >= self.high_conf_thresh]
            low_indices = [i for i, d in enumerate(detections)
                          if self.low_conf_thresh <= d.get("confidence", 0) < self.high_conf_thresh]

            high_dets = [detections[i] for i in high_indices]
            high_embs = [all_embeddings[i] for i in high_indices]
            low_dets = [detections[i] for i in low_indices]
            low_embs = [all_embeddings[i] for i in low_indices]

            # ── STAGE 1: active tracks ↔ high-conf detections ────────
            if active_tracks and high_dets:
                cost = _build_cost_matrix(
                    active_tracks, high_dets, high_embs, warped, img_diag, use_clip
                )
                matches, unmatched_t, unmatched_d = _run_assignment(cost, self.match_threshold)

                for ti, di in matches:
                    active_tracks[ti].update(high_dets[di]["bbox"], frame_idx, high_embs[di])
                    active_tracks[ti].confidence = max(
                        active_tracks[ti].confidence, high_dets[di].get("confidence", 0)
                    )

                remaining_tracks = [active_tracks[i] for i in unmatched_t]
                remaining_high_dets = [(high_dets[i], high_embs[i]) for i in unmatched_d]
            else:
                remaining_tracks = list(active_tracks)
                remaining_high_dets = list(zip(high_dets, high_embs))

            # ── STAGE 2: remaining tracks ↔ low-conf detections ──────
            if remaining_tracks and low_dets:
                cost2 = _build_cost_matrix(
                    remaining_tracks, low_dets, low_embs, warped, img_diag, use_clip
                )
                matches2, unmatched_t2, _ = _run_assignment(cost2, self.match_threshold_low)

                for ti, di in matches2:
                    remaining_tracks[ti].update(low_dets[di]["bbox"], frame_idx, low_embs[di])

                still_unmatched = [remaining_tracks[i] for i in unmatched_t2]
            else:
                still_unmatched = list(remaining_tracks)

            # ── Create new tracks from unmatched high-conf detections ─
            for det, emb in remaining_high_dets:
                t = Track(det["label"], det["bbox"], frame_idx, det.get("confidence", 1.0))
                t.embedding = emb
                active_tracks.append(t)

            # ── Retire lost tracks ────────────────────────────────────
            new_active = []
            for trk in active_tracks:
                if trk.lost_frames > self.max_lost:
                    trk.is_active = False
                    finished_tracks.append(trk)
                else:
                    new_active.append(trk)
            # Also keep the still-unmatched (they incremented lost_frames already)
            for trk in still_unmatched:
                if trk not in new_active and trk.is_active:
                    new_active.append(trk)
            active_tracks = new_active

        all_tracks = finished_tracks + active_tracks
        cmc_msg = f", CMC {cmc_applied}/{max(len(frame_detections)-1,1)}" if use_cmc else ""
        clip_msg = f", CLIP {clip_computed} crops" if use_clip else " (spatial-only)"
        logger.info(
            f"ByteTrack: {len(all_tracks)} tracks from "
            f"{len(frame_detections)} frames{cmc_msg}{clip_msg}"
        )
        return all_tracks
