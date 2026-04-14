"""Mask-aware ByteTrack — two-stage tracker with Kalman filter and fused cost matrix.

Drop-in replacement for the simple ByteTracker when SAM3 is the detector,
using mask IoU + bbox IoU + centroid motion + area ratio for association.

Interface matches the existing ByteTracker:
    tracker.track(frame_detections, frames=None) -> list[Track]

Each detection dict must have: {"label", "bbox", "confidence"}
Optional: {"mask": np.ndarray (HxW bool)}
"""

from __future__ import annotations

from enum import Enum, auto

import cv2
import numpy as np
from loguru import logger


# ── Existing Track class (same interface as tracker.py) ───────────────────────

class Track:
    """A tracked object spanning multiple frames."""

    _next_id = 1

    def __init__(self, label: str, bbox: list[float], frame_idx: int,
                 confidence: float = 1.0, mask: np.ndarray | None = None):
        self.id = Track._next_id
        Track._next_id += 1
        self.label = label
        self.bboxes: list[tuple[int, list[float]]] = [(frame_idx, bbox)]
        self.confidence = confidence
        self.lost_frames = 0
        self.is_active = True
        self.masks: list[tuple[int, np.ndarray | None]] = [(frame_idx, mask)]

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

    def update(self, bbox: list[float], frame_idx: int, mask: np.ndarray | None = None) -> None:
        self.bboxes.append((frame_idx, bbox))
        self.masks.append((frame_idx, mask))
        self.lost_frames = 0
        self.is_active = True


# ── Kalman filter for bbox ────────────────────────────────────────────────

class KalmanBoxTracker:
    """Constant-velocity Kalman filter for [cx, cy, area, aspect_ratio]."""

    _F = np.eye(8, dtype=np.float32)
    _F[0, 4] = _F[1, 5] = _F[2, 6] = _F[3, 7] = 1.0

    _H = np.eye(4, 8, dtype=np.float32)

    def __init__(self, bbox_xyxy: np.ndarray):
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w = bbox_xyxy[2] - bbox_xyxy[0]
        h = bbox_xyxy[3] - bbox_xyxy[1]
        a = w * h
        r = w / max(h, 1e-6)

        self.x = np.array([cx, cy, a, r, 0, 0, 0, 0], dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        self.Q = np.eye(8, dtype=np.float32) * 1.0
        self.Q[4:, 4:] *= 0.01
        self.R = np.eye(4, dtype=np.float32) * 1.0

    def predict(self) -> np.ndarray:
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self.Q
        self.x[2] = max(self.x[2], 1.0)
        return self._state_to_bbox()

    def update(self, bbox_xyxy: np.ndarray) -> None:
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
        w = bbox_xyxy[2] - bbox_xyxy[0]
        h = bbox_xyxy[3] - bbox_xyxy[1]
        z = np.array([cx, cy, w * h, w / max(h, 1e-6)], dtype=np.float32)

        y = z - self._H @ self.x
        S = self._H @ self.P @ self._H.T + self.R
        K = self.P @ self._H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self._H) @ self.P

    def _state_to_bbox(self) -> np.ndarray:
        cx, cy, a, r = self.x[:4]
        a = max(a, 1.0)
        r = max(r, 0.01)
        w = np.sqrt(a * r)
        h = a / max(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)

    @property
    def predicted_bbox(self) -> np.ndarray:
        return self._state_to_bbox()

    @property
    def predicted_centroid(self) -> np.ndarray:
        return self.x[:2].copy()


# ── Internal track state ──────────────────────────────────────────────────

class TrackState(Enum):
    TENTATIVE = auto()
    CONFIRMED = auto()
    LOST = auto()
    REMOVED = auto()


class STrack:
    """Internal tracked object with Kalman state."""

    _next_id: int = 1

    def __init__(self, label: str, bbox: np.ndarray, confidence: float,
                 frame_idx: int, mask: np.ndarray | None = None, min_confirm: int = 2):
        self.track_id = STrack._next_id
        STrack._next_id += 1

        self.label = label
        self.state = TrackState.TENTATIVE
        self.kf = KalmanBoxTracker(bbox)
        self.min_confirm = min_confirm

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.last_bbox = bbox.copy()
        self.last_mask: np.ndarray | None = mask
        self.last_area = float(mask.sum()) if mask is not None else float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        self.last_confidence = confidence
        self.first_frame = frame_idx
        self.last_frame = frame_idx

        # History for output Track
        self.bbox_history: list[tuple[int, list[float]]] = [
            (frame_idx, [float(v) for v in bbox])
        ]
        self.mask_history: list[tuple[int, np.ndarray | None]] = [
            (frame_idx, mask)
        ]

    @classmethod
    def reset_id_counter(cls):
        cls._next_id = 1

    def predict(self) -> np.ndarray:
        self.age += 1
        self.time_since_update += 1
        return self.kf.predict()

    def update(self, bbox: np.ndarray, confidence: float, frame_idx: int,
               mask: np.ndarray | None = None) -> None:
        self.kf.update(bbox)
        self.last_bbox = bbox.copy()
        self.last_mask = mask
        self.last_area = float(mask.sum()) if mask is not None else float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        self.last_confidence = confidence
        self.last_frame = frame_idx
        self.hits += 1
        self.time_since_update = 0

        self.bbox_history.append((frame_idx, [float(v) for v in bbox]))
        self.mask_history.append((frame_idx, mask))

        if self.state == TrackState.TENTATIVE and self.hits >= self.min_confirm:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_lost(self):
        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.LOST

    def mark_removed(self):
        self.state = TrackState.REMOVED

    @property
    def bbox(self) -> np.ndarray:
        return self.kf.predicted_bbox

    @property
    def centroid(self) -> np.ndarray:
        return self.kf.predicted_centroid

    @property
    def is_active(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.TENTATIVE)

    def to_track(self) -> Track:
        """Convert to output Track object matching simple ByteTracker interface."""
        track = Track.__new__(Track)
        track.id = self.track_id
        track.label = self.label
        track.bboxes = list(self.bbox_history)
        track.confidence = self.last_confidence
        track.lost_frames = self.time_since_update
        track.is_active = self.is_active
        track.masks = list(self.mask_history)
        return track


# ── Association helpers (self-contained, no external deps) ────────────────

def _bbox_iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """NxM bbox IoU matrix."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    union = np.maximum(area_a + area_b - inter, 1e-6)
    return (inter / union).astype(np.float32)


def _mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def _mask_iou_batch(masks_a: list[np.ndarray | None], masks_b: list[np.ndarray | None]) -> np.ndarray:
    n, m = len(masks_a), len(masks_b)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        if masks_a[i] is None:
            continue
        for j in range(m):
            if masks_b[j] is None:
                continue
            iou_mat[i, j] = _mask_iou(masks_a[i], masks_b[j])
    return iou_mat


def _centroid_distance_batch(centroids_a: np.ndarray, centroids_b: np.ndarray,
                              img_diag: float) -> np.ndarray:
    diff = centroids_a[:, None, :] - centroids_b[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    if img_diag > 0:
        dist = dist / img_diag
    return dist.astype(np.float32)


def _area_ratio_batch(areas_a: np.ndarray, areas_b: np.ndarray) -> np.ndarray:
    a = areas_a[:, None]
    b = areas_b[None, :]
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    mx = np.maximum(mx, 1e-6)
    return (mn / mx).astype(np.float32)


def _compute_cost_matrix(
    track_boxes: np.ndarray,
    track_centroids: np.ndarray,
    track_areas: np.ndarray,
    track_masks: list[np.ndarray | None],
    det_boxes: np.ndarray,
    det_centroids: np.ndarray,
    det_areas: np.ndarray,
    det_masks: list[np.ndarray | None],
    img_diag: float,
    w_bbox: float = 0.25,
    w_mask: float = 0.45,
    w_motion: float = 0.20,
    w_area: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Build fused NxM cost matrix. Returns (cost, biou, miou)."""
    n, m = track_boxes.shape[0], det_boxes.shape[0]
    if n == 0 or m == 0:
        return np.empty((n, m), dtype=np.float32), np.empty((n, m), dtype=np.float32), None

    biou = _bbox_iou_batch(track_boxes, det_boxes)
    bbox_cost = 1.0 - biou

    has_masks = any(m is not None for m in track_masks)
    if has_masks:
        miou = _mask_iou_batch(track_masks, det_masks)
        mask_cost = 1.0 - miou
    else:
        miou = None
        mask_cost = np.ones((n, m), dtype=np.float32)
        w_bbox += w_mask
        w_mask = 0.0

    cdist = _centroid_distance_batch(track_centroids, det_centroids, img_diag)
    motion_cost = np.clip(cdist, 0.0, 1.0)

    aratio = _area_ratio_batch(track_areas, det_areas)
    area_cost = 1.0 - aratio

    cost = (w_bbox * bbox_cost + w_mask * mask_cost
            + w_motion * motion_cost + w_area * area_cost)

    return cost.astype(np.float32), biou, miou


def _gate_cost_matrix(cost: np.ndarray, biou: np.ndarray | None,
                       miou: np.ndarray | None,
                       gate_bbox_iou: float = 0.30,
                       gate_mask_iou: float = 0.10) -> np.ndarray:
    """Reject matches where bbox overlaps but mask doesn't."""
    if biou is None or miou is None:
        return cost
    bad = (biou > gate_bbox_iou) & (miou < gate_mask_iou)
    cost = cost.copy()
    cost[bad] = 1e5
    return cost


def _linear_assignment(cost: np.ndarray, threshold: float = 0.65):
    """Solve assignment problem with scipy."""
    from scipy.optimize import linear_sum_assignment

    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))

    row_idx, col_idx = linear_sum_assignment(cost)

    matches = []
    unmatched_tracks = set(range(cost.shape[0]))
    unmatched_dets = set(range(cost.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] > threshold:
            continue
        matches.append((r, c))
        unmatched_tracks.discard(r)
        unmatched_dets.discard(c)

    return matches, sorted(unmatched_tracks), sorted(unmatched_dets)


# ── Duplicate suppression ─────────────────────────────────────────────────

def _suppress_duplicates(dets: list[dict], bbox_iou_thresh: float = 0.80,
                          mask_iou_thresh: float = 0.70) -> list[dict]:
    """Remove near-duplicate detections on the same frame (same label, high overlap)."""
    if len(dets) <= 1:
        return dets
    sorted_dets = sorted(dets, key=lambda d: d.get("confidence", 0), reverse=True)
    keep = []
    suppressed = set()
    for i, di in enumerate(sorted_dets):
        if i in suppressed:
            continue
        keep.append(di)
        for j in range(i + 1, len(sorted_dets)):
            if j in suppressed:
                continue
            dj = sorted_dets[j]
            if di["label"].lower() != dj["label"].lower():
                continue
            biou = _bbox_iou_single(di["bbox"], dj["bbox"])
            if biou > bbox_iou_thresh:
                suppressed.add(j)
                continue
            mi = di.get("mask")
            mj = dj.get("mask")
            if mi is not None and mj is not None:
                m_iou = _mask_iou(mi, mj)
                if m_iou > mask_iou_thresh:
                    suppressed.add(j)
    return keep


def _bbox_iou_single(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ── Camera Motion Compensation (reused from existing tracker) ─────────────

_orb = cv2.ORB_create(nfeatures=500)
_bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def _estimate_camera_motion(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray | None:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape[:2]
    scale = min(1.0, 640.0 / max(h, w))
    if scale < 1.0:
        prev_small = cv2.resize(prev_gray, (int(w * scale), int(h * scale)))
        curr_small = cv2.resize(curr_gray, (int(w * scale), int(h * scale)))
    else:
        prev_small, curr_small = prev_gray, curr_gray
        scale = 1.0
    kp1, des1 = _orb.detectAndCompute(prev_small, None)
    kp2, des2 = _orb.detectAndCompute(curr_small, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return None
    matches = _bf_matcher.knnMatch(des1, des2, k=2)
    good = [m for m_pair in matches if len(m_pair) == 2
            for m, n in [m_pair] if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    if scale < 1.0:
        pts1 /= scale
        pts2 /= scale
    affine, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    if affine is None or inliers is None or np.sum(inliers) / len(inliers) < 0.3:
        return None
    return affine


def _warp_bbox(bbox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    corners = np.array([
        [bbox[0], bbox[1]], [bbox[2], bbox[1]],
        [bbox[2], bbox[3]], [bbox[0], bbox[3]],
    ], dtype=np.float32)
    ones = np.ones((4, 1), dtype=np.float32)
    pts = np.hstack([corners, ones])
    warped = (affine @ pts.T).T
    return np.array([warped[:, 0].min(), warped[:, 1].min(),
                     warped[:, 0].max(), warped[:, 1].max()], dtype=np.float32)


# ── Main tracker ──────────────────────────────────────────────────────────

class MaskAwareByteTracker:
    """
    Two-stage ByteTrack with mask-aware association and Kalman filter.

    Same interface as ByteTracker.track():
        track(frame_detections, frames=None) -> list[Track]

    Detection dicts should have: {"label", "bbox", "confidence", "mask" (optional)}
    """

    # Association weights
    W_BBOX = 0.25
    W_MASK = 0.45
    W_MOTION = 0.20
    W_AREA = 0.10

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 5,
        confidence_threshold: float = 0.40,
        low_confidence_threshold: float = 0.10,
        min_confirm: int = 2,
    ):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.confidence_threshold = confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.min_confirm = min_confirm

    def track(
        self,
        frame_detections: list[list[dict]],
        frames: list[np.ndarray] | None = None,
    ) -> list[Track]:
        """Track objects across frames using mask-aware two-stage association."""
        Track._next_id = 1
        STrack.reset_id_counter()

        active_tracks: list[STrack] = []
        lost_tracks: list[STrack] = []
        removed_tracks: list[STrack] = []

        use_cmc = frames is not None and len(frames) == len(frame_detections)
        cmc_applied = 0
        img_diag = 1.0

        if frames is not None and len(frames) > 0:
            h, w = frames[0].shape[:2]
            img_diag = np.sqrt(h**2 + w**2)

        for frame_idx, raw_dets in enumerate(frame_detections):
            # Suppress duplicate detections before tracking
            dets = _suppress_duplicates(raw_dets)

            # Camera motion compensation
            affine = None
            if use_cmc and frame_idx > 0 and (active_tracks or lost_tracks):
                affine = _estimate_camera_motion(frames[frame_idx - 1], frames[frame_idx])
                if affine is not None:
                    cmc_applied += 1

            # Predict all tracks
            all_stracks = active_tracks + lost_tracks
            for st in all_stracks:
                predicted = st.predict()
                if affine is not None:
                    # Warp to compensate camera motion
                    st.kf.x[:2] = _warp_bbox(predicted, affine)[:2]  # warp centroid

            # Split detections by confidence
            high_dets = [d for d in dets if d.get("confidence", 0) >= self.confidence_threshold]
            low_dets = [d for d in dets if self.low_confidence_threshold <= d.get("confidence", 0) < self.confidence_threshold]

            # ── STAGE 1: active + lost tracks ↔ high-conf detections ──
            if all_stracks and high_dets:
                track_boxes = np.array([st.bbox for st in all_stracks], dtype=np.float32)
                track_centroids = np.array([st.centroid for st in all_stracks], dtype=np.float32)
                track_areas = np.array([st.last_area for st in all_stracks], dtype=np.float32)
                track_masks = [st.last_mask for st in all_stracks]

                det_boxes = np.array([d["bbox"] for d in high_dets], dtype=np.float32)
                det_centroids = np.array([
                    [(d["bbox"][0]+d["bbox"][2])/2, (d["bbox"][1]+d["bbox"][3])/2]
                    for d in high_dets
                ], dtype=np.float32)
                det_areas = np.array([
                    float(d["mask"].sum()) if d.get("mask") is not None
                    else (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])
                    for d in high_dets
                ], dtype=np.float32)
                det_masks = [d.get("mask") for d in high_dets]

                cost, biou, miou = _compute_cost_matrix(
                    track_boxes, track_centroids, track_areas, track_masks,
                    det_boxes, det_centroids, det_areas, det_masks,
                    img_diag, self.W_BBOX, self.W_MASK, self.W_MOTION, self.W_AREA,
                )
                cost = _gate_cost_matrix(cost, biou, miou)

                # Label-aware gating: set cost to infinity for label mismatches
                for ti in range(len(all_stracks)):
                    for di in range(len(high_dets)):
                        if all_stracks[ti].label.lower() != high_dets[di]["label"].lower():
                            cost[ti, di] = 1e5

                matches, unmatched_t, unmatched_d = _linear_assignment(cost, threshold=0.65)

                for ti, di in matches:
                    d = high_dets[di]
                    all_stracks[ti].update(
                        np.array(d["bbox"], dtype=np.float32),
                        d.get("confidence", 0),
                        frame_idx,
                        d.get("mask"),
                    )

                remaining_tracks = [all_stracks[i] for i in unmatched_t]
                remaining_high_dets = [high_dets[i] for i in unmatched_d]
            else:
                remaining_tracks = list(all_stracks)
                remaining_high_dets = list(high_dets)

            # ── STAGE 2: remaining tracks ↔ low-conf detections ──
            if remaining_tracks and low_dets:
                track_boxes = np.array([st.bbox for st in remaining_tracks], dtype=np.float32)
                track_centroids = np.array([st.centroid for st in remaining_tracks], dtype=np.float32)
                track_areas = np.array([st.last_area for st in remaining_tracks], dtype=np.float32)
                track_masks = [st.last_mask for st in remaining_tracks]

                det_boxes = np.array([d["bbox"] for d in low_dets], dtype=np.float32)
                det_centroids = np.array([
                    [(d["bbox"][0]+d["bbox"][2])/2, (d["bbox"][1]+d["bbox"][3])/2]
                    for d in low_dets
                ], dtype=np.float32)
                det_areas = np.array([
                    float(d["mask"].sum()) if d.get("mask") is not None
                    else (d["bbox"][2]-d["bbox"][0])*(d["bbox"][3]-d["bbox"][1])
                    for d in low_dets
                ], dtype=np.float32)
                det_masks = [d.get("mask") for d in low_dets]

                cost2, biou2, miou2 = _compute_cost_matrix(
                    track_boxes, track_centroids, track_areas, track_masks,
                    det_boxes, det_centroids, det_areas, det_masks,
                    img_diag, self.W_BBOX, self.W_MASK, self.W_MOTION, self.W_AREA,
                )
                cost2 = _gate_cost_matrix(cost2, biou2, miou2)

                for ti in range(len(remaining_tracks)):
                    for di in range(len(low_dets)):
                        if remaining_tracks[ti].label.lower() != low_dets[di]["label"].lower():
                            cost2[ti, di] = 1e5

                matches2, unmatched_t2, _ = _linear_assignment(cost2, threshold=0.75)

                for ti, di in matches2:
                    d = low_dets[di]
                    remaining_tracks[ti].update(
                        np.array(d["bbox"], dtype=np.float32),
                        d.get("confidence", 0),
                        frame_idx,
                        d.get("mask"),
                    )

                still_unmatched = [remaining_tracks[i] for i in unmatched_t2]
            else:
                still_unmatched = list(remaining_tracks)

            # Handle unmatched tracks
            new_lost = []
            for st in still_unmatched:
                if st.time_since_update > self.max_lost:
                    st.mark_removed()
                    removed_tracks.append(st)
                else:
                    st.mark_lost()
                    new_lost.append(st)

            # Create new tracks from unmatched high-conf detections
            for d in remaining_high_dets:
                st = STrack(
                    label=d["label"],
                    bbox=np.array(d["bbox"], dtype=np.float32),
                    confidence=d.get("confidence", 0),
                    frame_idx=frame_idx,
                    mask=d.get("mask"),
                    min_confirm=self.min_confirm,
                )
                active_tracks.append(st)

            # Rebuild active/lost lists
            new_active = []
            seen_ids = set()
            # Keep tracks that were updated this frame
            for st in active_tracks + lost_tracks:
                if st.is_active and st.time_since_update == 0 and st.track_id not in seen_ids:
                    new_active.append(st)
                    seen_ids.add(st.track_id)
            # Keep active tracks that weren't updated but still active
            for st in active_tracks:
                if st.is_active and st.track_id not in seen_ids:
                    new_active.append(st)
                    seen_ids.add(st.track_id)
            active_tracks = new_active

            lost_tracks = [st for st in (lost_tracks + new_lost)
                          if st.state == TrackState.LOST
                          and st.time_since_update <= self.max_lost
                          and st.track_id not in seen_ids]

        # Collect all tracks and convert to output format
        all_stracks_final = active_tracks + lost_tracks + removed_tracks
        output_tracks = [st.to_track() for st in all_stracks_final
                        if st.hits >= self.min_confirm or st.state == TrackState.CONFIRMED]

        cmc_status = f", CMC on {cmc_applied}/{max(len(frame_detections)-1, 1)} transitions" if use_cmc else ""
        masks_used = sum(1 for st in all_stracks_final if st.last_mask is not None)
        logger.info(
            f"MaskAwareByteTrack: {len(output_tracks)} tracks from "
            f"{len(frame_detections)} frames, {masks_used} with masks{cmc_status}"
        )

        return output_tracks
