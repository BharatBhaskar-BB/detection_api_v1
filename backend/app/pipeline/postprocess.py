"""Post-processing: Track merge, CLIP visual Re-ID, key frame selection."""

import cv2
import numpy as np
import torch
from loguru import logger

from app.pipeline.tracker import Track, compute_iou


# ── Singleton CLIP model ─────────────────────────────────────────────────────

_clip_model = None
_clip_preprocess = None
_clip_device = None


def _ensure_clip_loaded():
    """Lazy-load CLIP ViT-B/32 (shared singleton across calls)."""
    global _clip_model, _clip_preprocess, _clip_device
    if _clip_model is not None:
        return

    try:
        import clip as clip_module

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        _clip_model, _clip_preprocess = clip_module.load("ViT-B/32", device=device)
        _clip_model.eval()
        _clip_device = device
        logger.info(f"CLIP ViT-B/32 loaded on {device} for visual Re-ID")
    except ImportError:
        logger.warning("clip package not installed — CLIP Re-ID disabled")
    except Exception as e:
        logger.warning(f"CLIP load failed — Re-ID disabled: {e}")


def _extract_crop_embedding(frame_bgr: np.ndarray, bbox: list[float], pad_ratio: float = 0.1) -> np.ndarray | None:
    """Crop an object from a frame and compute its CLIP embedding (512-d, L2-normalized)."""
    if _clip_model is None:
        return None

    from PIL import Image

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]

    # Pad the crop slightly for context
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_ratio, bh * pad_ratio
    x1i = max(0, int(x1 - px))
    y1i = max(0, int(y1 - py))
    x2i = min(w, int(x2 + px))
    y2i = min(h, int(y2 + py))

    crop = frame_bgr[y1i:y2i, x1i:x2i]
    if crop.size == 0:
        return None

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(crop_rgb)
    img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(_clip_device)

    with torch.no_grad():
        feat = _clip_model.encode_image(img_tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)

    return feat.cpu().numpy().flatten().astype(np.float32)


# ── Track Merging ────────────────────────────────────────────────────────────


def merge_fragmented_tracks(
    tracks: list[Track],
    iou_threshold: float = 0.3,
    max_gap_frames: int = 10,
) -> list[Track]:
    """
    Merge tracks that likely belong to the same object.
    Two tracks merge if:
      - Same label
      - Gap between them < max_gap_frames
      - Spatial IoU of last bbox of track A and first bbox of track B > threshold
    """
    if len(tracks) <= 1:
        return tracks

    # Sort by first frame
    tracks_sorted = sorted(tracks, key=lambda t: t.first_frame)
    merged = [tracks_sorted[0]]

    for track in tracks_sorted[1:]:
        did_merge = False
        for existing in merged:
            if (
                existing.label == track.label
                and track.first_frame - existing.last_frame <= max_gap_frames
                and compute_iou(existing.last_bbox, track.bboxes[0][1]) > iou_threshold
            ):
                # Merge: add all bboxes from track to existing
                existing.bboxes.extend(track.bboxes)
                did_merge = True
                break

        if not did_merge:
            merged.append(track)

    logger.info(f"Track merge: {len(tracks)} → {len(merged)} tracks")
    return merged


# ── CLIP Visual Re-ID ────────────────────────────────────────────────────────


def clip_visual_reid(
    tracks: list[Track],
    frames: list[np.ndarray],
    similarity_threshold: float = 0.8,
) -> list[Track]:
    """
    Merge re-appearing objects using CLIP crop embeddings.
    A merge requires:
      1. Same label
      2. High CLIP visual similarity (same-looking object)
      3. No co-visibility in the same frame (different physical objects)
    """
    if len(tracks) <= 1:
        return tracks

    _ensure_clip_loaded()
    if _clip_model is None:
        logger.info(f"CLIP Re-ID: {len(tracks)} tracks (CLIP unavailable — skipping)")
        return tracks

    # Compute a representative embedding for each track
    # Use the middle observation (camera settled, best view)
    track_embeddings: dict[int, np.ndarray] = {}
    for track in tracks:
        if not track.bboxes:
            continue
        # Pick the middle bbox observation
        mid_idx = len(track.bboxes) // 2
        frame_idx, bbox = track.bboxes[mid_idx]
        if frame_idx < len(frames):
            emb = _extract_crop_embedding(frames[frame_idx], bbox)
            if emb is not None:
                track_embeddings[track.id] = emb

    if len(track_embeddings) < 2:
        logger.info(f"CLIP Re-ID: not enough embeddings ({len(track_embeddings)}) — skipping")
        return tracks

    # Group tracks by label, then find merges within each group
    label_groups: dict[str, list[Track]] = {}
    for track in tracks:
        label_groups.setdefault(track.label, []).append(track)

    # Pre-compute frame sets for each track (for concurrent-detection check)
    track_frame_sets: dict[int, set[int]] = {}
    for track in tracks:
        track_frame_sets[track.id] = {frame_idx for frame_idx, _ in track.bboxes}

    merged_result: list[Track] = []
    total_merges = 0

    for label, group in label_groups.items():
        if len(group) <= 1:
            merged_result.extend(group)
            continue

        # Pairwise cosine similarity within the group
        consumed: set[int] = set()
        for i, t1 in enumerate(group):
            if t1.id in consumed:
                continue
            emb1 = track_embeddings.get(t1.id)
            if emb1 is None:
                merged_result.append(t1)
                continue

            for j in range(i + 1, len(group)):
                t2 = group[j]
                if t2.id in consumed:
                    continue
                emb2 = track_embeddings.get(t2.id)
                if emb2 is None:
                    continue

                # Never merge tracks that are visible in the same frame —
                # co-visible tracks are different physical objects
                frames_t1 = track_frame_sets.get(t1.id, set())
                frames_t2 = track_frame_sets.get(t2.id, set())
                if frames_t1 & frames_t2:
                    continue

                sim = float(np.dot(emb1, emb2))
                if sim >= similarity_threshold:
                    # Merge t2 into t1
                    t1.bboxes.extend(t2.bboxes)
                    t1.bboxes.sort(key=lambda x: x[0])
                    # Update t1's frame set to include t2's frames
                    track_frame_sets[t1.id] = frames_t1 | frames_t2
                    consumed.add(t2.id)
                    total_merges += 1

            merged_result.append(t1)

        # Add any tracks not consumed and not yet added
        for t in group:
            if t.id not in consumed and t not in merged_result:
                merged_result.append(t)

    logger.info(f"CLIP Re-ID: {len(tracks)} → {len(merged_result)} tracks ({total_merges} merges)")
    return merged_result


# ── Key Frame Selection ──────────────────────────────────────────────────────


def _embed_full_frame(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Compute CLIP embedding for a full frame. Returns L2-normalized (512,)."""
    if _clip_model is None:
        _ensure_clip_loaded()
    if _clip_model is None:
        return None

    from PIL import Image

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    tensor = _clip_preprocess(pil_img).unsqueeze(0).to(_clip_device)

    with torch.no_grad():
        emb = _clip_model.encode_image(tensor)
    emb = emb.cpu().numpy().astype(np.float32).flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


def select_key_frames(
    tracks: list[Track],
    total_frames: int,
    frames: list[np.ndarray] | None = None,
    min_frames: int = 3,
    max_frames: int = 12,
) -> list[int]:
    """
    Greedy set-cover key frame selection.

    Picks frames that maximize track coverage so the LLM verifier
    sees as many tracked objects as possible.
    """
    if not tracks:
        return [0] if total_frames > 0 else []

    # Build frame → set of track IDs mapping
    frame_tracks: dict[int, set[int]] = {}
    for track in tracks:
        for frame_idx, bbox in track.bboxes:
            frame_tracks.setdefault(frame_idx, set()).add(track.id)

    candidate_indices = sorted(frame_tracks.keys())
    budget = max(min_frames, min(max_frames, len(tracks) // 3 + 2))

    if len(candidate_indices) <= budget:
        logger.info(f"Key frames: all {len(candidate_indices)} candidates selected (≤ budget {budget})")
        return candidate_indices

    # ── Greedy set-cover ──────────────────────────────────────────────────
    uncovered = {t.id for t in tracks}
    selected: list[int] = []

    while uncovered and len(selected) < budget:
        best_frame = -1
        best_count = 0
        for frame_idx, track_ids in frame_tracks.items():
            if frame_idx in selected:
                continue
            count = len(track_ids & uncovered)
            if count > best_count:
                best_count = count
                best_frame = frame_idx
        if best_frame < 0:
            break
        selected.append(best_frame)
        uncovered -= frame_tracks.get(best_frame, set())

    selected.sort()
    logger.info(f"Key frames (set-cover): selected {len(selected)} from {total_frames} frames")
    return selected
