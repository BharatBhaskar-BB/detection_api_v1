"""Inventory assembly — fuse tracked counts + LLM-fused counts."""

from loguru import logger

from app.pipeline.tracker import Track


def assemble_inventory(
    tracks: list[Track],
    new_items: list[dict],
    count_hints: dict[str, int] | None = None,
    max_per_frame: dict[str, int] | None = None,
    llm_fused_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    Build inventory from tracked objects.

    If llm_fused_counts is provided (from LLM fusion step), use those directly.
    Otherwise use max(reid_count, max_per_frame) — the two provable signals.

    Args:
        tracks: ByteTrack tracks with labels and counts
        new_items: [NEW] items from LLM priming ({"name", "frame"})
        count_hints: LLM primer's max-per-frame counts (fallback only)
        max_per_frame: max simultaneous GDINO detections per label (fallback only)
        llm_fused_counts: LLM-reasoned unique counts per label (preferred)

    Returns:
        list of {"name", "count", "room_name", "source", "is_new", "frame_index"}
    """
    count_hints = count_hints or {}
    max_per_frame = max_per_frame or {}
    llm_fused_counts = llm_fused_counts or {}

    # Count unique objects by label from tracks (Re-ID output)
    reid_counts: dict[str, int] = {}
    for track in tracks:
        reid_counts[track.label] = reid_counts.get(track.label, 0) + 1

    # Merge labels: some may only exist in LLM fusion but not in tracks
    all_labels = set(reid_counts.keys())
    if llm_fused_counts:
        all_labels |= set(llm_fused_counts.keys())

    inventory: list[dict] = []
    for label in sorted(all_labels):
        reid_count = reid_counts.get(label, 0)
        fused = llm_fused_counts.get(label, 0)

        if fused > 0:
            # LLM fusion available — use it directly
            final_count = fused
            logger.info(
                f"Inventory [{label}]: llm_fused={fused}, reid={reid_count} → final={final_count}"
            )
        elif reid_count > 0:
            llm_hint = count_hints.get(label, 0)
            max_frame = max_per_frame.get(label, 0)

            # Two provable signals:
            # 1. reid_count: number of unique tracks (can undercount if
            #    CLIP merges identical-looking objects)
            # 2. max_frame: max simultaneously visible per frame after NMS
            #    (provable physical floor — those objects co-exist)
            final_count = max(reid_count, max_frame)

            logger.info(
                f"Inventory [{label}]: "
                f"reid={reid_count}, llm_hint={llm_hint}, max_frame={max_frame} "
                f"→ final={final_count}"
            )
        else:
            continue

        inventory.append({
            "name": label,
            "count": final_count,
            "room_name": "",
            "source": "detected",
            "is_new": False,
            "frame_index": None,
            "confidence": 1.0,
        })

    # Fallback injection: [NEW] items that GDINO couldn't detect
    detected_labels = set(reid_counts.keys()) | set(llm_fused_counts.keys())
    for item in new_items:
        name_lower = item["name"].lower()
        if not any(name_lower in d.lower() or d.lower() in name_lower for d in detected_labels):
            inventory.append({
                "name": item["name"],
                "count": item.get("count", 1),
                "room_name": "",
                "source": "llm_priming_fallback",
                "is_new": True,
                "frame_index": item.get("frame"),
                "confidence": 0.85,
            })
            logger.info(f"Fallback injection: {item['name']} (LLM saw it, GDINO missed)")

    logger.info(f"Inventory assembled: {len(inventory)} unique items")
    return inventory
