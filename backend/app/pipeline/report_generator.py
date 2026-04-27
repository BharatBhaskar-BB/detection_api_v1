"""Survey report generator — produces structured report data from pipeline output.

Combines inventory items with volume/weight estimates, evidence frames, 
packing materials, and special handling notes.
"""

from __future__ import annotations

import asyncio
import base64
import json
import cv2
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger

from app.config import get_settings
from app.pipeline.llm_inventory import InventoryItem, DraftInventory
from app.pipeline.frame_selector import SelectedFrame
from app.pipeline.volume_lookup import (
    lookup_volume_weight,
    estimate_packing_materials,
    estimate_truck_size,
)


@dataclass
class ReportItem:
    """A single item in the survey report with all computed fields."""
    name: str
    count: int
    room: str
    disposition: Optional[str]  # going|staying|scrap|haul|sell|None
    going: Optional[bool]       # DEPRECATED compat — derived from disposition
    notes: str
    size: str
    dimensions: Optional[dict]
    best_frame_ts: Optional[float]
    volume_cuft: float
    weight_lbs: float
    total_volume_cuft: float   # volume * count
    total_weight_lbs: float    # weight * count
    evidence_image_b64: Optional[str] = None  # JPEG base64 of best frame
    special_handling: list[str] = field(default_factory=list)


@dataclass
class SurveyReport:
    """Complete survey report data."""
    # Metadata
    generated_at: str
    video_duration_s: float
    survey_mode: str  # "voice_assisted" or "visual_only"
    num_frames_analyzed: int

    # Inventory
    items: list[ReportItem]
    rooms: dict  # room_name -> list of ReportItem indices

    # Totals
    total_items: int
    total_volume_cuft: float
    total_weight_lbs: float
    truck_recommendation: str

    # Packing
    packing_materials: dict

    # Special handling
    special_handling_notes: list[str]

    # LLM usage
    usage: dict


# ── Centroid estimation (separate from inventory — display only) ──

async def _estimate_centroids(
    items: list[InventoryItem],
    frames: list[SelectedFrame],
    frame_indices: list[int],
) -> list[Optional[list[int]]]:
    """Estimate centroid [y, x] (0-1000 scale) for each item in its evidence frame.

    This is a DISPLAY-ONLY step — completely decoupled from inventory counting.
    Batches items by frame to keep per-call payloads small and reliable.

    Args:
        items: Inventory items (name used for locating).
        frames: All selected frames.
        frame_indices: Index into `frames` for each item's evidence frame.

    Returns:
        List of centroids (one per item), None for items that couldn't be located.
    """
    if not items or not frames:
        return [None] * len(items)

    settings = get_settings()
    api_key = settings.GEMINI_API_KEY
    if not api_key:
        logger.debug("No Gemini API key — skipping centroid estimation")
        return [None] * len(items)

    try:
        from google import genai
        from google.genai import types
        from PIL import Image
        from app.pipeline.llm_inventory import _parse_centroid

        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.warning(f"Centroid setup failed: {e}")
        return [None] * len(items)

    # Group items by their assigned frame index
    frame_to_items: dict[int, list[int]] = {}
    for i, fidx in enumerate(frame_indices):
        frame_to_items.setdefault(fidx, []).append(i)

    result: list[Optional[list[int]]] = [None] * len(items)

    thinking_config = None
    model_name = settings.GEMINI_MODEL
    if "2.5" in model_name:
        thinking_config = types.ThinkingConfig(thinking_budget=256)

    # Process each frame's items in a single call (max ~30 items per frame typically)
    for fidx, item_ids in frame_to_items.items():
        if fidx >= len(frames):
            continue

        # Build item list for this frame
        item_entries = []
        for local_i, global_i in enumerate(item_ids):
            item_entries.append(f'  {local_i}: "{items[global_i].name}"')

        prompt = (
            "Look at this image. For each item below, find it and return its center "
            "as [y, x] on a 0-1000 scale (0,0 = top-left, 1000,1000 = bottom-right).\n"
            "If you cannot find the item, return null.\n\n"
            "Items:\n" + "\n".join(item_entries) + "\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"centroids": [[y, x], [y, x], null, ...]}\n'
            f"Array must have exactly {len(item_ids)} entries.\n"
        )

        try:
            frame_rgb = cv2.cvtColor(frames[fidx].frame, cv2.COLOR_BGR2RGB)
            parts: list = [prompt, Image.fromarray(frame_rgb)]

            gen_config = types.GenerateContentConfig(
                max_output_tokens=2000,
                temperature=0.1,
                response_mime_type="application/json",
                thinking_config=thinking_config,
            )

            response = await client.aio.models.generate_content(
                model=model_name,
                contents=parts,
                config=gen_config,
            )

            text = (response.text or "").strip()
            data = json.loads(text)
            raw_centroids = data.get("centroids", [])

            for local_i, global_i in enumerate(item_ids):
                if local_i < len(raw_centroids):
                    result[global_i] = _parse_centroid(raw_centroids[local_i])

        except Exception as e:
            logger.warning(f"Centroid estimation failed for frame {fidx} ({len(item_ids)} items): {e}")
            # Items in this frame get None centroids — no vignette, just raw frame

    n_found = sum(1 for c in result if c is not None)
    logger.info(f"Centroid estimation: {n_found}/{len(items)} items located")
    return result


def _extract_evidence_frame(
    item: InventoryItem,
    all_frames: list[SelectedFrame],
) -> Optional[str]:
    """Extract the best evidence frame for an item as base64 JPEG.
    
    Uses best_frame_ts from LLM to find the closest frame.
    Returns base64-encoded JPEG string, or None.
    """
    if not all_frames:
        return None

    ts = item.best_frame_ts
    if ts is None:
        # Fallback: pick the middle frame (likely best overall view)
        best_frame = all_frames[len(all_frames) // 2]
    else:
        # Find the frame closest to the target timestamp
        best_frame = min(all_frames, key=lambda f: abs(f.timestamp_s - ts))

    # Encode as JPEG
    _, buf = cv2.imencode(".jpg", best_frame.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode("utf-8")


def _assign_frame_indices(
    items: list,
    all_frames: list[SelectedFrame],
) -> list[int]:
    """Assign each item a frame index based on the LLM's best_frame_ts.

    Uses the timestamp where the LLM reported seeing the item to find the
    closest selected frame. Never redistributes items to unrelated frames —
    it's better to have multiple items share a frame than show wrong evidence.

    Returns a list of frame indices (one per item).
    """
    if not all_frames:
        return [0] * len(items)

    n_frames = len(all_frames)
    frame_timestamps = [f.timestamp_s for f in all_frames]

    assignments: list[int] = []
    for item in items:
        ts = item.best_frame_ts
        if ts is None:
            # No timestamp — pick the middle frame as a neutral fallback
            assignments.append(n_frames // 2)
        else:
            best_idx = min(range(n_frames), key=lambda i: abs(frame_timestamps[i] - ts))
            assignments.append(best_idx)

    return assignments


def _encode_evidence_frames(
    items: list,
    all_frames: list[SelectedFrame],
    frame_indices: list[int],
    centroids: list[Optional[list[int]]],
) -> list[Optional[str]]:
    """Encode evidence frames as base64 JPEG, applying vignette where centroid is available."""
    if not all_frames:
        return [None] * len(items)

    from app.pipeline.evidence_renderer import render_evidence

    results: list[Optional[str]] = []
    raw_frame_cache: dict[int, str] = {}
    for i, idx in enumerate(frame_indices):
        centroid = centroids[i] if i < len(centroids) else None

        if centroid is not None:
            annotated = render_evidence(
                all_frames[idx].frame, centroid_1000=centroid, item_name=items[i].name
            )
            _, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80]
            )
            results.append(base64.b64encode(buf).decode("utf-8"))
        else:
            if idx not in raw_frame_cache:
                _, buf = cv2.imencode(
                    ".jpg", all_frames[idx].frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                raw_frame_cache[idx] = base64.b64encode(buf).decode("utf-8")
            results.append(raw_frame_cache[idx])

    return results


def _detect_special_handling(item: InventoryItem) -> list[str]:
    """Detect special handling requirements from item properties."""
    flags = []
    name = item.name.lower()
    notes = (item.notes or "").lower()

    # Fragile
    fragile_items = {"tv", "television", "monitor", "lamp", "mirror", "glass",
                     "chandelier", "vase", "artwork", "painting", "sculpture"}
    if any(f in name for f in fragile_items):
        flags.append("fragile — professional packing recommended")

    # Heavy / oversized
    heavy_items = {"refrigerator", "safe", "gun safe", "piano", "pool table",
                   "vending machine", "washing machine", "dryer", "chest freezer"}
    if any(h in name for h in heavy_items):
        flags.append("heavy — may require additional crew")

    # Disassembly
    disassemble_items = {"bunk bed", "bed frame", "jungle gym", "trampoline",
                         "sectional sofa", "crib", "shelving unit"}
    if any(d in name for d in disassemble_items):
        flags.append("disassembly/reassembly required")

    # From notes
    if "disassembl" in notes:
        flags.append("disassembly mentioned")
    if "original box" in notes:
        flags.append("has original packaging")
    if "fragile" in notes:
        flags.append("flagged fragile")
    if "staying" in notes and item.disposition != "staying":
        flags.append("check staying status")

    return flags


async def generate_report(
    inventory: DraftInventory,
    selected_frames: list[SelectedFrame],
    video_duration_s: float,
    has_audio: bool = False,
) -> dict:
    """Generate a complete survey report from pipeline output.
    
    Returns a JSON-serializable dict for the frontend.
    """
    report_items: list[dict] = []
    rooms: dict[str, list[int]] = {}
    total_vol = 0.0
    total_wt = 0.0
    total_count = 0
    all_special = []

    # Step 1: Assign frame indices (pure logic, no LLM)
    frame_indices = _assign_frame_indices(inventory.items, selected_frames)

    # Step 2: Estimate centroids in a separate LLM call (display-only)
    centroids = await _estimate_centroids(
        inventory.items, selected_frames, frame_indices
    )

    # Step 3: Encode evidence frames with optional vignette overlay
    evidence_images = _encode_evidence_frames(
        inventory.items, selected_frames, frame_indices, centroids
    )

    for idx, item in enumerate(inventory.items):
        # Volume/weight lookup
        vol, wt = lookup_volume_weight(
            name=item.name,
            size=item.size,
            dimensions=item.dimensions,
            count=item.count,
        )
        item_total_vol = round(vol * item.count, 1)
        item_total_wt = round(wt * item.count, 1)
        total_vol += item_total_vol
        total_wt += item_total_wt
        total_count += item.count

        # Evidence frame (from batch assignment)
        evidence_b64 = evidence_images[idx]

        # Special handling
        special = _detect_special_handling(item)
        if special:
            for flag in special:
                all_special.append(f"{item.name} (x{item.count}): {flag}")

        room_name = item.room or "unknown"
        if room_name not in rooms:
            rooms[room_name] = []
        rooms[room_name].append(idx)

        report_items.append({
            "name": item.name,
            "count": item.count,
            "room": room_name,
            "going": item.going,
            "disposition": item.disposition,
            "notes": item.notes,
            "size": item.size,
            "dimensions": item.dimensions,
            "best_frame_ts": item.best_frame_ts,
            "volume_cuft": vol,
            "weight_lbs": wt,
            "total_volume_cuft": item_total_vol,
            "total_weight_lbs": item_total_wt,
            "evidence_image": evidence_b64,
            "special_handling": special,
        })

    # Packing materials
    packing = estimate_packing_materials([
        {"name": item.name, "count": item.count, "room": item.room, "notes": item.notes}
        for item in inventory.items
    ])

    # Truck
    truck = estimate_truck_size(total_vol)

    report = {
        "generated_at": datetime.now().isoformat(),
        "video_duration_s": round(video_duration_s, 1),
        "survey_mode": "voice_assisted" if has_audio else "visual_only",
        "num_frames_analyzed": len(selected_frames),
        "items": report_items,
        "rooms": rooms,
        "summary": {
            "total_items": total_count,
            "total_item_types": len(report_items),
            "total_volume_cuft": round(total_vol, 1),
            "total_weight_lbs": round(total_wt, 1),
            "truck_recommendation": truck,
        },
        "packing_materials": packing,
        "special_handling": all_special,
        "usage": inventory.usage,
    }

    logger.info(f"Report: {total_count} items, {total_vol:.0f} cu ft, "
                f"{total_wt:.0f} lbs, truck: {truck}")

    return report
