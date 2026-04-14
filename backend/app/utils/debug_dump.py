"""Debug artifact dump — saves per-step outputs for pipeline debugging."""

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

TESTING_DIR = Path(__file__).resolve().parent.parent.parent.parent / "testing"


def get_scan_dir(scan_id: str) -> Path:
    """Get (and create) testing/<scan_id>/ directory."""
    d = TESTING_DIR / scan_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_frames(scan_id: str, frames: list[np.ndarray], prefix: str = "frame") -> None:
    """Save all extracted frames as numbered JPEGs."""
    d = get_scan_dir(scan_id) / "01_frames"
    d.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(d / f"{prefix}_{i:04d}.jpg"), frame)
    logger.info(f"[debug] Dumped {len(frames)} frames → {d}")


def dump_scenes(scan_id: str, scenes: list[dict]) -> None:
    """Save scene representative frames + metadata."""
    d = get_scan_dir(scan_id) / "02_scenes"
    d.mkdir(exist_ok=True)
    meta = []
    for i, scene in enumerate(scenes):
        frame = scene.get("frame")
        if frame is not None:
            cv2.imwrite(str(d / f"scene_{i:02d}.jpg"), frame)
        meta.append({
            "scene_idx": i,
            "frame_range": scene.get("frame_range"),
            "middle_frame": scene.get("middle_frame"),
        })
    (d / "scenes_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info(f"[debug] Dumped {len(scenes)} scenes → {d}")


def dump_priming(
    scan_id: str,
    prompts: list[str],
    new_items: list[dict],
    usage: dict,
    count_hints: dict[str, int] | None = None,
    raw_llm_response: str | None = None,
    raw_pass2_response: str | None = None,
) -> None:
    """Save LLM priming results."""
    d = get_scan_dir(scan_id) / "03_priming"
    d.mkdir(exist_ok=True)
    data = {
        "prompts": prompts,
        "new_items": new_items,
        "count_hints": count_hints or {},
        "usage": usage,
    }
    (d / "priming_result.json").write_text(json.dumps(data, indent=2))
    if raw_llm_response:
        try:
            raw_parsed = json.loads(raw_llm_response)
            (d / "llm_raw_response.json").write_text(json.dumps(raw_parsed, indent=2))
        except json.JSONDecodeError:
            (d / "llm_raw_response.txt").write_text(raw_llm_response)
    if raw_pass2_response:
        try:
            raw_parsed = json.loads(raw_pass2_response)
            (d / "llm_pass2_cross_frame.json").write_text(json.dumps(raw_parsed, indent=2))
        except json.JSONDecodeError:
            (d / "llm_pass2_cross_frame.txt").write_text(raw_pass2_response)
    logger.info(f"[debug] Dumped priming ({len(prompts)} prompts, {len(new_items)} new, hints: {count_hints}) → {d}")


def dump_detections(
    scan_id: str,
    frames: list[np.ndarray],
    all_detections: list[list[dict]],
) -> None:
    """Save per-frame detections as annotated images + JSON."""
    d = get_scan_dir(scan_id) / "04_detections"
    d.mkdir(exist_ok=True)

    # JSON summary
    summary = []
    for i, dets in enumerate(all_detections):
        summary.append({
            "frame": i,
            "count": len(dets),
            "items": [{"label": det["label"], "confidence": round(det.get("confidence", 0), 3), "bbox": det.get("bbox")} for det in dets],
        })
    (d / "detections.json").write_text(json.dumps(summary, indent=2))

    # Distinct colors for up to 20 objects (BGR)
    _PALETTE = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 255), (255, 128, 0), (0, 128, 255), (128, 255, 0),
        (255, 0, 128), (0, 255, 128), (128, 128, 255), (255, 128, 128), (128, 255, 128),
        (64, 0, 192), (192, 64, 0), (0, 192, 64), (192, 0, 192), (64, 192, 192),
    ]

    # Check if any detection has a mask
    has_masks = any(
        det.get("mask") is not None
        for dets in all_detections for det in dets
    )

    # Annotated images: masks (colored overlay) when available, else bboxes
    for i, (frame, dets) in enumerate(zip(frames, all_detections)):
        if not dets:
            continue
        annotated = frame.copy()

        if has_masks:
            # Mask overlay mode — semi-transparent colored masks + label text
            overlay = annotated.copy()
            for j, det in enumerate(dets):
                color = _PALETTE[j % len(_PALETTE)]
                mask = det.get("mask")
                if mask is not None:
                    # Resize mask to frame if needed
                    h, w = annotated.shape[:2]
                    if mask.shape[:2] != (h, w):
                        mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                          interpolation=cv2.INTER_NEAREST).astype(bool)
                    overlay[mask] = color
                # Label at bbox top-left or mask centroid
                bbox = det.get("bbox")
                if bbox:
                    lx, ly = int(bbox[0]), max(int(bbox[1]) - 5, 12)
                else:
                    ly, lx = 20 + j * 18, 10
                label = f"{det['label']} {det.get('confidence', 0):.2f}"
                cv2.putText(annotated, label, (lx, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            # Blend overlay at 40% opacity
            cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
        else:
            # Bbox-only mode (GDINO)
            for det in dets:
                bbox = det.get("bbox")
                if bbox:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{det['label']} {det.get('confidence', 0):.2f}"
                    cv2.putText(annotated, label, (x1, max(y1 - 5, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(str(d / f"det_{i:04d}.jpg"), annotated)

    total_dets = sum(len(d) for d in all_detections)
    logger.info(f"[debug] Dumped detections ({total_dets} total) → {d}")


def dump_tracks(
    scan_id: str,
    tracks: list,
    stage: str = "raw",
) -> None:
    """Save track data as JSON. stage: 'raw', 'merged', 'reid'."""
    d = get_scan_dir(scan_id) / "05_tracking"
    d.mkdir(exist_ok=True)
    data = []
    for t in tracks:
        data.append({
            "track_id": t.id,
            "label": t.label,
            "confidence": round(t.confidence, 3),
            "frame_count": t.frame_count,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "bboxes": [(fi, [round(v, 1) for v in bb]) for fi, bb in t.bboxes],
        })
    (d / f"tracks_{stage}.json").write_text(json.dumps(data, indent=2))

    # Summary: label → count
    label_counts = {}
    for t in tracks:
        label_counts[t.label] = label_counts.get(t.label, 0) + 1
    (d / f"counts_{stage}.json").write_text(json.dumps(label_counts, indent=2, sort_keys=True))
    logger.info(f"[debug] Dumped {len(tracks)} tracks ({stage}) → {d}")


def dump_inventory(scan_id: str, inventory: list[dict], stage: str = "assembled") -> None:
    """Save inventory data. stage: 'assembled', 'verified'."""
    d = get_scan_dir(scan_id) / "06_inventory"
    d.mkdir(exist_ok=True)
    (d / f"inventory_{stage}.json").write_text(json.dumps(inventory, indent=2))
    logger.info(f"[debug] Dumped inventory ({len(inventory)} items, {stage}) → {d}")


def dump_key_frames(
    scan_id: str,
    frames: list[np.ndarray],
    indices: list[int],
) -> None:
    """Save key frames sent to LLM verification."""
    d = get_scan_dir(scan_id) / "07_key_frames"
    d.mkdir(exist_ok=True)
    for i, idx in enumerate(indices):
        if idx < len(frames):
            cv2.imwrite(str(d / f"keyframe_{i:02d}_fidx{idx:04d}.jpg"), frames[idx])
    (d / "key_frame_indices.json").write_text(json.dumps(indices))
    logger.info(f"[debug] Dumped {len(indices)} key frames → {d}")
