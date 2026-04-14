"""LLM Fusion — ask GPT-4o to determine final unique counts from per-frame observations + tracking."""

import json

import httpx
from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.pipeline.tracker import Track


async def llm_fuse_counts(
    tracks: list[Track],
    per_frame_llm: list[dict],
    all_detections: list[list[dict]],
) -> tuple[dict[str, int], dict]:
    """
    Use GPT-4o to reason about unique item counts given:
      - LLM per-frame observations (what and how many per scene frame)
      - ByteTrack/Re-ID track data (which detections correspond across frames)

    Returns:
        (final_counts, usage) where final_counts = {"office chair": 6, ...}
    """
    settings = get_settings()
    if not settings.OPENAI_API_KEY:
        logger.warning("No OpenAI key — falling back to max(per_frame)")
        return _fallback_max(per_frame_llm), {"tokens_in": 0, "tokens_out": 0, "model": "none"}

    # ── Build per-label track summary ──────────────────────────────────────
    # For each label, list tracks and which frames they span
    label_tracks: dict[str, list[dict]] = {}
    for track in tracks:
        frames = sorted(set(fi for fi, _ in track.bboxes))
        entry = {"track_id": track.id, "frames": frames}
        label_tracks.setdefault(track.label, []).append(entry)

    # ── Build per-label per-frame LLM counts ──────────────────────────────
    # Scene frames are a subset of all frames (e.g. 8 out of 24).
    # The LLM only saw scene frames, so we label them scene 1..N.
    label_llm_counts: dict[str, dict[int, int]] = {}
    for frame_data in per_frame_llm:
        fnum = frame_data.get("frame", 0)
        for item in frame_data.get("items", []):
            name = item.get("name", "").strip().lower()
            count = item.get("count", 1)
            if name:
                label_llm_counts.setdefault(name, {})[fnum] = count

    # ── Determine which labels need fusion ─────────────────────────────────
    all_labels = sorted(set(list(label_tracks.keys()) + list(label_llm_counts.keys())))

    # Build the payload for the LLM
    items_payload = []
    for label in all_labels:
        llm_obs = label_llm_counts.get(label, {})
        trks = label_tracks.get(label, [])

        # Skip labels with no tracking data and no LLM observations
        if not llm_obs and not trks:
            continue

        item_info = {
            "label": label,
            "llm_per_scene_frame": {f"scene_{k}": v for k, v in sorted(llm_obs.items())},
            "tracks": [
                {"id": t["track_id"], "active_frames": t["frames"]}
                for t in trks
            ],
            "track_count": len(trks),
        }
        items_payload.append(item_info)

    if not items_payload:
        return {}, {"tokens_in": 0, "tokens_out": 0, "model": "none"}

    prompt = (
        "You are counting unique movable items in a room from a walkthrough video.\n\n"
        "For each item type below, you receive:\n"
        "1. **LLM per-scene-frame counts**: How many of this item were visible in each scene frame "
        "(scene frames are sampled snapshots — not every video frame).\n"
        "2. **Tracking data**: Object tracks from a visual tracker. Each track has an ID and the "
        "video frames where it was detected. Tracks ideally represent individual objects, but the "
        "tracker may over-fragment (same object gets multiple track IDs) or under-fragment "
        "(different objects share a track ID).\n\n"
        "Your task: determine how many UNIQUE physical items of each type exist in the room.\n\n"
        "Reasoning guidelines:\n"
        "- The LLM per-frame counts are reliable for how many are visible at once.\n"
        "- If tracks overlap in time (active in the same frames), they represent DIFFERENT objects.\n"
        "- If tracks don't overlap in time, they MAY be the same object seen at different times, "
        "OR different objects — use the LLM counts to decide.\n"
        "- If max(LLM per-frame count) = N and there are tracks that don't overlap with those N, "
        "those tracks likely represent additional objects.\n"
        "- The total unique count is usually >= max(LLM per-frame) but can be higher if objects "
        "appear in non-overlapping frames.\n"
        "- Ignore track IDs that seem like tracker noise (very short, single-frame tracks with "
        "labels that don't match any LLM observation).\n\n"
        f"Items data:\n{json.dumps(items_payload, indent=2)}\n\n"
        'Return JSON: {"counts": {"<item_name>": <unique_count>, ...}}\n'
        "Include ALL items from the data above."
    )

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        http_client=httpx.AsyncClient(verify=False),
    )

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=0,
        response_format={"type": "json_object"},
    )

    text = response.choices[0].message.content or ""
    usage = {
        "tokens_in": response.usage.prompt_tokens if response.usage else 0,
        "tokens_out": response.usage.completion_tokens if response.usage else 0,
        "model": settings.OPENAI_MODEL,
    }

    logger.info(f"LLM Fusion response:\n{text}")
    logger.info(f"Fusion usage: {usage}")

    try:
        data = json.loads(text)
        counts = {k.lower(): int(v) for k, v in data.get("counts", {}).items()}
    except (json.JSONDecodeError, ValueError):
        logger.error(f"Failed to parse fusion response, falling back to max(per_frame)")
        counts = _fallback_max(per_frame_llm)

    return counts, usage


def _fallback_max(per_frame_llm: list[dict]) -> dict[str, int]:
    """Fallback: just take max(per_frame) as before."""
    max_counts: dict[str, int] = {}
    for frame_data in per_frame_llm:
        for item in frame_data.get("items", []):
            name = item.get("name", "").strip().lower()
            count = item.get("count", 1)
            if name:
                max_counts[name] = max(max_counts.get(name, 0), count)
    return max_counts
