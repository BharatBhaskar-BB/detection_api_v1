"""Gallery-based CLIP tracker — BoT-SORT-style with never-forget Re-ID.

Instead of creating new tracks when objects reappear after going off-screen
and then trying to stitch fragments post-hoc, this tracker maintains a
permanent gallery of ALL retired track embeddings.  When a detection can't
match any active track, it searches the gallery and reactivates the old track
if a strong CLIP match is found (with co-visibility gating to prevent merging
distinct objects).

This eliminates:
  - merge_fragmented_tracks  (no fragments to merge)
  - clip_visual_reid          (re-ID is built into the tracker)
  - any counting formula hack (count = number of unique track IDs, period)

Uses the same Track class, CLIP helpers, and CMC from tracker.py.
"""

import numpy as np
from loguru import logger
from scipy.optimize import linear_sum_assignment

from app.pipeline.tracker import (
    Track,
    _batch_crop_embeddings,
    _build_cost_matrix,
    _ensure_clip,
    _run_assignment,
    compute_iou,
    estimate_camera_motion,
    warp_bbox,
)


class GalleryTracker:
    """
    BoT-SORT-style tracker with a persistent CLIP embedding gallery.

    Flow per frame:
      1. CMC — warp active track bboxes to compensate camera motion
      2. CLIP — batch-embed all detections
      3. Stage 1: match active tracks ↔ high-conf detections (Hungarian)
      4. Stage 2: match remaining active tracks ↔ low-conf detections
      5. Gallery search: unmatched high-conf detections ↔ retired tracks
         - Same label required
         - CLIP cosine similarity ≥ gallery_threshold
         - Co-visibility gate: never reactivate if the retired track was
           already matched to a *different* detection this frame
      6. Create truly new tracks only for detections that didn't match
         anything (active or gallery)
    """

    def __init__(
        self,
        max_lost: int = 5,
        high_conf_thresh: float = 0.40,
        low_conf_thresh: float = 0.10,
        match_threshold: float = 0.55,
        match_threshold_low: float = 0.65,
        gallery_sim_threshold: float = 0.78,
        min_track_length: int = 1,
    ):
        self.max_lost = max_lost
        self.high_conf_thresh = high_conf_thresh
        self.low_conf_thresh = low_conf_thresh
        self.match_threshold = match_threshold
        self.match_threshold_low = match_threshold_low
        self.gallery_sim_threshold = gallery_sim_threshold
        self.min_track_length = min_track_length

    def track(
        self,
        frame_detections: list[list[dict]],
        frames: list[np.ndarray] | None = None,
    ) -> list[Track]:
        """
        Track objects across frames using gallery-based Re-ID.

        Args:
            frame_detections: per-frame detections [{"label", "bbox", "confidence"}, ...]
            frames: BGR frames (required for CLIP)

        Returns:
            list of Track objects (unique object = unique track ID)
        """
        Track._next_id = 1
        use_clip = _ensure_clip() and frames is not None
        use_cmc = frames is not None and len(frames) == len(frame_detections)

        active_tracks: list[Track] = []
        gallery: list[Track] = []  # NEVER-FORGET: retired tracks stay here forever
        clip_computed = 0
        cmc_applied = 0
        gallery_reactivations = 0

        # Image diagonal for centroid normalisation
        img_diag = 1.0
        frame_w = 1
        if frames is not None and len(frames) > 0:
            h, w = frames[0].shape[:2]
            img_diag = float(np.sqrt(h ** 2 + w ** 2))
            frame_w = w

        for frame_idx, detections in enumerate(frame_detections):
            # ── CMC ──────────────────────────────────────────────────
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

            # ── Batch CLIP embeddings for all detections ─────────────
            if use_clip and detections:
                all_bboxes = [d["bbox"] for d in detections]
                all_embeddings = _batch_crop_embeddings(frames[frame_idx], all_bboxes)
                clip_computed += sum(1 for e in all_embeddings if e is not None)
            else:
                all_embeddings = [None] * len(detections)

            # ── Split by confidence ──────────────────────────────────
            high_indices = [i for i, d in enumerate(detections)
                           if d.get("confidence", 0) >= self.high_conf_thresh]
            low_indices = [i for i, d in enumerate(detections)
                          if self.low_conf_thresh <= d.get("confidence", 0) < self.high_conf_thresh]

            high_dets = [detections[i] for i in high_indices]
            high_embs = [all_embeddings[i] for i in high_indices]
            low_dets = [detections[i] for i in low_indices]
            low_embs = [all_embeddings[i] for i in low_indices]

            # Track IDs matched on this frame (for co-visibility gating)
            matched_track_ids_this_frame: set[int] = set()

            # ── STAGE 1: active tracks ↔ high-conf detections ───────
            if active_tracks and high_dets:
                cost = _build_cost_matrix(
                    active_tracks, high_dets, high_embs, warped, img_diag, use_clip,
                    frame_w=frame_w,
                )
                matches, unmatched_t, unmatched_d = _run_assignment(cost, self.match_threshold)

                for ti, di in matches:
                    active_tracks[ti].update(high_dets[di]["bbox"], frame_idx, high_embs[di])
                    active_tracks[ti].confidence = max(
                        active_tracks[ti].confidence, high_dets[di].get("confidence", 0)
                    )
                    matched_track_ids_this_frame.add(active_tracks[ti].id)

                remaining_tracks = [active_tracks[i] for i in unmatched_t]
                remaining_high_det_indices = list(unmatched_d)
            else:
                remaining_tracks = list(active_tracks)
                remaining_high_det_indices = list(range(len(high_dets)))

            # ── STAGE 2: remaining tracks ↔ low-conf detections ─────
            if remaining_tracks and low_dets:
                cost2 = _build_cost_matrix(
                    remaining_tracks, low_dets, low_embs, warped, img_diag, use_clip,
                    frame_w=frame_w,
                )
                matches2, unmatched_t2, _ = _run_assignment(cost2, self.match_threshold_low)

                for ti, di in matches2:
                    remaining_tracks[ti].update(low_dets[di]["bbox"], frame_idx, low_embs[di])
                    matched_track_ids_this_frame.add(remaining_tracks[ti].id)

            # ── STAGE 3: Gallery search for unmatched high-conf dets ─
            # Only search gallery with CLIP — spatial alone is useless
            # for objects that left and came back.
            newly_activated: list[Track] = []
            truly_new_det_indices: list[int] = []

            if use_clip and gallery and remaining_high_det_indices:
                # Filter gallery to those with embeddings
                gallery_with_emb = [
                    g for g in gallery
                    if g.embedding is not None
                ]

                if gallery_with_emb:
                    # Build similarity matrix: gallery × unmatched detections
                    for di in remaining_high_det_indices:
                        det = high_dets[di]
                        det_emb = high_embs[di]
                        if det_emb is None:
                            truly_new_det_indices.append(di)
                            continue

                        best_sim = -1.0
                        best_gallery_idx = -1

                        for gi, gtrk in enumerate(gallery_with_emb):
                            # Label gate
                            if gtrk.label.lower() != det["label"].lower():
                                continue

                            # Co-visibility gate: if this gallery track is
                            # already reactivated (matched) this frame, skip
                            if gtrk.id in matched_track_ids_this_frame:
                                continue

                            # Opposite-edge gate: if the track was last seen
                            # near one edge and detection is near the opposite
                            # edge, this is almost certainly a different object
                            # (camera panning brings new objects from the far side)
                            if frame_w > 1:
                                trk_cx = (gtrk.last_bbox[0] + gtrk.last_bbox[2]) / 2.0
                                det_cx = (det["bbox"][0] + det["bbox"][2]) / 2.0
                                edge_pct = 0.15
                                trk_left = trk_cx < frame_w * edge_pct
                                trk_right = trk_cx > frame_w * (1 - edge_pct)
                                det_left = det_cx < frame_w * edge_pct
                                det_right = det_cx > frame_w * (1 - edge_pct)
                                if (trk_left and det_right) or (trk_right and det_left):
                                    continue

                            sim = float(np.dot(gtrk.embedding, det_emb))
                            if sim > best_sim:
                                best_sim = sim
                                best_gallery_idx = gi

                        if best_sim >= self.gallery_sim_threshold and best_gallery_idx >= 0:
                            # REACTIVATE: pull from gallery → active
                            reactivated = gallery_with_emb[best_gallery_idx]
                            reactivated.update(det["bbox"], frame_idx, det_emb)
                            reactivated.confidence = max(
                                reactivated.confidence, det.get("confidence", 0)
                            )
                            reactivated.is_active = True
                            reactivated.lost_frames = 0
                            gallery.remove(reactivated)
                            newly_activated.append(reactivated)
                            matched_track_ids_this_frame.add(reactivated.id)
                            gallery_reactivations += 1
                        else:
                            truly_new_det_indices.append(di)
                else:
                    truly_new_det_indices = list(remaining_high_det_indices)
            else:
                truly_new_det_indices = list(remaining_high_det_indices)

            # ── Create NEW tracks (truly novel objects) ───────────────
            for di in truly_new_det_indices:
                det = high_dets[di]
                emb = high_embs[di]
                t = Track(det["label"], det["bbox"], frame_idx, det.get("confidence", 1.0))
                t.embedding = emb
                active_tracks.append(t)

            # Add reactivated tracks back to active
            active_tracks.extend(newly_activated)

            # ── Retire lost tracks → gallery (not deleted!) ──────────
            new_active: list[Track] = []
            for trk in active_tracks:
                if trk.lost_frames > self.max_lost:
                    trk.is_active = False
                    gallery.append(trk)  # → gallery, not trash
                else:
                    new_active.append(trk)
            active_tracks = new_active

        # ── Final: collect everything ────────────────────────────────
        all_tracks = active_tracks + gallery

        # Filter ghost tracks: single-detection AND narrow bbox (< 30px wide)
        before = len(all_tracks)
        filtered = []
        for t in all_tracks:
            if t.frame_count == 1:
                bb = t.last_bbox  # [x1, y1, x2, y2]
                w = bb[2] - bb[0]
                if w < 30:
                    continue  # tiny single-detection blip — noise
            filtered.append(t)
        all_tracks = filtered
        ghosts = before - len(all_tracks)
        if ghosts > 0:
            logger.info(f"GalleryTracker: filtered {ghosts} ghost tracks (single-det, <30px wide)")

        cmc_msg = f", CMC {cmc_applied}/{max(len(frame_detections)-1,1)}" if use_cmc else ""
        clip_msg = f", CLIP {clip_computed} crops" if use_clip else " (spatial-only)"
        logger.info(
            f"GalleryTracker: {len(all_tracks)} tracks from "
            f"{len(frame_detections)} frames, "
            f"{gallery_reactivations} gallery reactivations"
            f"{cmc_msg}{clip_msg}"
        )
        return all_tracks
