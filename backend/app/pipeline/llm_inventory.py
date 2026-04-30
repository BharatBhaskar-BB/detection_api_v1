"""Step 4: LLM Inventory Draft — time-batched multimodal LLM calls with optional transcript."""

import asyncio
import base64
import json
import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings
from app.pipeline.checklist import get_checklist_text
from app.pipeline.frame_selector import SelectedFrame
from app.pipeline.transcriber import TranscriptionResult


# Valid disposition values
DISPOSITION_VALUES = {"going", "staying", "scrap", "haul", "sell"}


@dataclass
class InventoryItem:
    name: str
    count: int
    room: str = ""
    disposition: Optional[str] = None  # going|staying|scrap|haul|sell|None(undecided)
    going: Optional[bool] = None  # DEPRECATED — kept for report compat
    notes: str = ""
    size: str = "medium"        # small, medium, large
    dimensions: Optional[dict] = None  # {"length_in": 48, "width_in": 36, "height_in": 30}
    best_frame_ts: Optional[float] = None  # timestamp (s) where item is best visible
    best_frame_idx: Optional[int] = None  # direct index into selected_frames[]
    centroid: Optional[list[int]] = None  # Gemini centroid [y, x] on 0-1000 scale
    confidence: float = 0.9
    source: str = "llm_draft"


@dataclass
class DraftInventory:
    items: list[InventoryItem] = field(default_factory=list)
    detection_prompts: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"tokens_in": 0, "tokens_out": 0, "model": ""})
    raw_responses: list[str] = field(default_factory=list)

    @property
    def going_items(self) -> list[InventoryItem]:
        return [item for item in self.items if item.disposition == "going"]

    @property
    def staying_items(self) -> list[InventoryItem]:
        return [item for item in self.items if item.disposition == "staying"]

    @property
    def unknown_items(self) -> list[InventoryItem]:
        return [item for item in self.items if item.disposition is None]


_INVENTORY_JSON_SCHEMA = """{
  "items": [
    {
      "name": "dining chair",
      "count": 4,
      "room": "kitchen",
      "disposition": "going",
      "size": "medium",
      "dimensions_approx": {"length_in": 18, "width_in": 20, "height_in": 34},
      "best_frame_ts": 5.0,
      "best_frame_idx": 3,
      "notes": "white wood chairs around round table"
    },
    {
      "name": "refrigerator",
      "count": 1,
      "room": "kitchen",
      "disposition": "staying",
      "size": "large",
      "dimensions_approx": {"length_in": 36, "width_in": 30, "height_in": 70},
      "best_frame_ts": 12.0,
      "best_frame_idx": 7,
      "notes": "owner says fridge is staying"
    }
  ]
}"""

_INVENTORY_JSON_SCHEMA_VISUAL = """{
  "items": [
    {
      "name": "office chair",
      "count": 6,
      "room": "office",
      "size": "medium",
      "dimensions_approx": {"length_in": 22, "width_in": 22, "height_in": 38},
      "best_frame_ts": 5.0,
      "best_frame_idx": 3,
      "notes": "black rolling chairs around conference table"
    },
    {
      "name": "laptop",
      "count": 2,
      "room": "office",
      "size": "small",
      "dimensions_approx": {"length_in": 14, "width_in": 10, "height_in": 1},
      "best_frame_ts": 8.3,
      "best_frame_idx": 5,
      "notes": "open on desk"
    }
  ]
}"""

_MERGE_JSON_SCHEMA = """{
  "items": [
    {
      "name": "dining chair",
      "count": 4,
      "room": "kitchen",
      "disposition": "going",
      "size": "medium",
      "dimensions_approx": {"length_in": 18, "width_in": 20, "height_in": 34},
      "best_frame_ts": 5.0,
      "notes": "white wood chairs around table"
    }
  ]
}"""

_MERGE_JSON_SCHEMA_VISUAL = """{
  "items": [
    {
      "name": "office chair",
      "count": 6,
      "room": "office",
      "size": "medium",
      "dimensions_approx": {"length_in": 22, "width_in": 22, "height_in": 38},
      "best_frame_ts": 5.0,
      "notes": "black rolling chairs around conference table"
    }
  ]
}"""


def _parse_centroid(raw: object) -> Optional[list[int]]:
    """Validate and return a Gemini-format centroid [y, x] (0-1000 scale).

    Returns None if the input is missing, malformed, or out of range.
    """
    if raw is None or not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    try:
        vals = [int(v) for v in raw]
    except (TypeError, ValueError):
        return None
    y, x = vals
    if not (0 <= y <= 1000 and 0 <= x <= 1000):
        return None
    return vals


class LLMInventoryDrafter:
    """Create a comprehensive inventory draft using LLM with frames + optional transcript."""

    def __init__(self):
        settings = get_settings()
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.gemini_model = settings.GEMINI_MODEL
        self.openai_api_key = settings.OPENAI_API_KEY
        self.openai_model = settings.OPENAI_MODEL
        self._batch_size = getattr(settings, "LLM_BATCH_FRAME_COUNT", 15)
        self._provider = self._select_provider()

    def _select_provider(self) -> str:
        if self.gemini_api_key:
            return "gemini"
        if self.openai_api_key:
            return "openai"
        return "none"

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def _format_timestamp(self, seconds: float) -> str:
        mm = int(seconds // 60)
        ss = int(seconds % 60)
        return f"{mm}:{ss:02d}"

    # ── Prompt Building ──

    def _build_batch_prompt(
        self,
        batch_frames: list[SelectedFrame],
        transcript: TranscriptionResult | None,
        batch_start_s: float,
        batch_end_s: float,
        total_duration_s: float,
        batch_index: int = 0,
        num_batches: int = 1,
    ) -> str:
        has_transcript = transcript is not None and transcript.has_speech

        prompt = (
            "You are a PROFESSIONAL MOVING ESTIMATOR reviewing a home video walkthrough.\n\n"
            f"This is batch {batch_index + 1} of {num_batches}, covering "
            f"{self._format_timestamp(batch_start_s)} - "
            f"{self._format_timestamp(batch_end_s)} of a "
            f"{self._format_timestamp(total_duration_s)} video.\n"
        )
        if num_batches > 1:
            prompt += (
                f"Other batches cover different parts of the video and may show "
                f"additional rooms you cannot see here.\n"
            )
        prompt += "\n"
        prompt += (
            "TASK: List EVERY item in these frames — both items being MOVED and items\n"
            "that are STAYING. A complete estimate requires knowing ALL items.\n"
            "For each item provide:\n"
            "- name: specific item name (lowercase)\n"
            "- count: how many you see\n"
            "- room: which room it's in. NUMBER duplicate room types! If the home\n"
            "  has two bedrooms, use 'bedroom 1', 'bedroom 2'. If two bathrooms,\n"
            "  use 'bathroom 1', 'bathroom 2'. Distinguish by visual cues like\n"
            "  decor, size, or contents (e.g., master bedroom with dark accent wall\n"
            "  = 'bedroom 1', guest room with plaid bedding = 'bedroom 2').\n"
            "- disposition: ONLY set this if the homeowner EXPLICITLY states the item's fate\n"
            "  in the audio. Valid values: 'going', 'staying', 'scrap', 'haul', 'sell'.\n"
            "  If the item is NOT discussed in the audio, OMIT this field entirely.\n"
            "- size: 'small', 'medium', or 'large' relative to typical furniture\n"
            "- dimensions_approx: estimated dimensions in inches {length_in, width_in, height_in}\n"
            "- best_frame_ts: REQUIRED — the timestamp (seconds) of the frame where this\n"
            "  item is MOST visible. Must match one of the provided frame timestamps.\n"
            "  This field is CRITICAL for evidence images — do NOT omit it.\n"
            "- best_frame_idx: REQUIRED — the frame NUMBER (integer) of the frame where\n"
            "  this item is most visible. Must match one of the 'Frame N' labels above.\n"
            "  This is the integer N from 'Frame N (at M:SS)'. This is CRITICAL.\n"
            "- notes: any relevant details (color, size, brand, special handling)\n\n"
        )

        if has_transcript:
            prompt += (
                "YOU HAVE TWO SOURCES — USE BOTH EQUALLY:\n\n"
                "SOURCE 1 — AUDIO TRANSCRIPT (for going/staying & mentioned items):\n"
                "The transcript below is from a live video call between the homeowner and\n"
                "a moving estimator. They walk through each room discussing which items\n"
                "are going vs staying.\n\n"
                "SOURCE 2 — VIDEO FRAMES (for visual-only items):\n"
                "The frames show the actual rooms. Many items are VISIBLE but NEVER\n"
                "discussed in the conversation. You MUST identify these too.\n\n"
                "RULES FOR TRANSCRIPT ITEMS:\n"
                "1. List EVERY item mentioned in the transcript, even if not clearly visible\n"
                "   in the frames. If they say 'rocking chair' or 'nightstand', add it.\n"
                "2. Set 'disposition' ONLY when the homeowner EXPLICITLY states an item's\n"
                "   fate. Map their words to one of these values:\n"
                "   - 'going': moving to new home. Signals include: 'take this', 'pack this',\n"
                "     'this is going', 'ship this', 'move this', 'we need this', 'box this up',\n"
                "     'wrap this', 'this comes with us'\n"
                "   - 'staying': left behind. Signals: 'stays', 'leave it', 'not taking',\n"
                "     'keep it here', 'stays with the house', 'don't pack'\n"
                "   - 'scrap': dispose/trash. Signals: 'throw away', 'trash it', 'dump it',\n"
                "     'get rid of', 'junk', 'toss it'\n"
                "   - 'haul': professional hauling/removal. Signals: 'haul away', 'junk removal',\n"
                "     'have someone take it', 'call junk guys'\n"
                "   - 'sell': marketplace/sell. Signals: 'sell this', 'put it on marketplace',\n"
                "     'list it', 'sell online', 'give away'\n"
                "   If the homeowner does NOT discuss an item at all, OMIT the disposition\n"
                "   field entirely. Do NOT guess or assume.\n"
                "3. The transcript may be in ANY LANGUAGE (English, Hindi, Spanish, etc.)\n"
                "   or code-switched (e.g., Hindi-English mix). Interpret the speaker's\n"
                "   intent regardless of language. Examples:\n"
                "   - Hindi: 'yeh le jaana hai' / 'isko pack karo' → going\n"
                "   - Hindi: 'yeh rahega' / 'isko rehne do' → staying\n"
                "   - Hindi: 'phenk do' / 'kabad mein daal do' → scrap\n"
                "   - Spanish: 'esto se va' → going, 'esto se queda' → staying\n"
                "   Always output item NAMES in English regardless of transcript language.\n"
                "4. Use room names from the conversation (e.g., 'bedroom number two').\n"
                "5. Include items from closets, entries, hallways, and quick pass-through\n"
                "   areas — these are often mentioned in audio but barely shown.\n"
                "6. Include luggage, suitcases, musical instruments, and cases mentioned.\n"
                "7. If audio mentions items like 'toys', 'outdoor toys', 'scooter',\n"
                "   'tricycle', list each separately — do NOT lump them together.\n"
                "8. Include appliances and tools mentioned: vacuum cleaners, tool chests,\n"
                "   toaster ovens, blenders, etc.\n"
                "9. Note special handling from audio: original boxes, disassembly needed,\n"
                "   weight concerns, etc.\n\n"
                "RULES FOR VISUAL-ONLY ITEMS (NOT mentioned in transcript):\n"
                "10. Carefully examine EVERY frame. If you see an item that is NEVER\n"
                "    discussed in the transcript, you MUST still list it.\n"
                "11. Common visual-only items: exercise/gym equipment (elliptical, bench,\n"
                "    weights, treadmill), monitors, printers, mirrors, dressers, toaster\n"
                "    ovens, changing tables, children's play equipment, wall art, lamps.\n"
                "12. For visual-only items, OMIT the disposition field and add\n"
                "    notes='not discussed in video call'.\n"
                "13. Use SPECIFIC item names: 'elliptical' not 'exercise equipment',\n"
                "    'computer monitor' not 'electronics', 'full-length mirror' not\n"
                "    'furniture'. Name the exact item you see.\n"
                "14. Do NOT create a single vague entry like 'workout room' or 'closet'.\n"
                "    Instead list each individual item you can see in the frame.\n\n"
                "TRANSCRIPT (full video):\n"
                f"{transcript.formatted_transcript()}\n\n"
            )
        else:
            prompt += (
                "NO AUDIO IS AVAILABLE — USE ONLY VISUAL INFORMATION.\n\n"
                "You must carefully examine EVERY frame and list ALL objects you can see.\n"
                "For each frame, scan systematically: left to right, foreground to background.\n\n"
                "RULES:\n"
                "1. List EVERY distinct item visible in the frames — furniture, appliances,\n"
                "   electronics, equipment, containers, decorations, etc.\n"
                "2. Use SPECIFIC names: 'coffee machine' not 'appliance', 'ottoman' not\n"
                "   'seat', 'water dispenser' not 'machine'.\n"
                "3. Count carefully: if you see 4 chairs across multiple frames in the same\n"
                "   area, count=4. Do not double-count items seen from different angles.\n"
                "4. Identify the room/area type from context (office, break room, living room,\n"
                "   kitchen, bedroom, etc.). NUMBER duplicate room types! If there are\n"
                "   two bedrooms, use 'bedroom 1', 'bedroom 2'. Distinguish by visual\n"
                "   cues (decor, size, contents).\n"
                "5. Do NOT return the example JSON — analyze the ACTUAL frames provided.\n"
                "6. Do NOT include a 'going' field in your JSON output — without audio\n"
                "   we cannot determine whether items are being moved or staying. The JSON\n"
                "   schema below does NOT have a 'going' field — follow it exactly.\n"
                "7. For each item, estimate its SIZE ('small', 'medium', 'large') and\n"
                "   approximate DIMENSIONS in inches (length, width, height).\n"
                "8. Set best_frame_ts to the timestamp (in seconds) of the frame where\n"
                "   the item is MOST clearly visible. This is REQUIRED for every item\n"
                "   and must match one of the provided frame timestamps.\n"
                "9. Set best_frame_idx to the frame NUMBER (integer N from 'Frame N') where\n"
                "   the item is MOST clearly visible. This is REQUIRED for every item.\n\n"
            )

        prompt += (
            "COUNTING RULES (CRITICAL — get this right):\n"
            "- These frames are SEQUENTIAL from a MOVING CAMERA in the same space.\n"
            "  The same physical object WILL appear in MULTIPLE frames from different\n"
            "  angles. You MUST recognize this and count each physical object ONCE.\n"
            "- ACCURACY over completeness — it is WORSE to over-count (same object\n"
            "  counted twice from different angles) than to slightly under-count.\n"
            "- To count correctly: find the SINGLE FRAME that shows the MOST instances\n"
            "  of an item type (e.g., the widest shot showing all chairs). Use THAT\n"
            "  frame's count. Do NOT add counts across frames.\n"
            "- If 4 chairs are visible in frame A and 3 in frame B, and they are in\n"
            "  the same area, the count is 4 (NOT 4+3=7, NOT 5).\n"
            "- Be specific: 'queen bed' not just 'bed', 'nightstand' not 'table',\n"
            "  'laptop' not 'electronics', 'elliptical' not 'workout room'.\n"
            "- Include small items on desks/tables: laptops, monitors, keyboards, phones.\n\n"
            "COMPLETENESS — DO NOT MISS THESE:\n"
            "- Small items on counters/shelves (toaster, blender, lamp, clock)\n"
            "- Entry/hallway furniture (hall tree, coat rack, console table)\n"
            "- Closet contents (suitcases, luggage, guitar cases, instrument cases)\n"
            "- Outdoor items (grill, fire pit, patio furniture, bikes, scooters, tricycles)\n"
            "- Garage items (tool chest, workbench, shelving, bins, holiday decorations)\n"
            "- Gym/fitness equipment (elliptical, treadmill, weight bench, dumbbells,\n"
            "  barbell rack, resistance bands, yoga mats, exercise ball)\n"
            "- Play equipment (jungle gym, crash pads, easel, slide, trampoline)\n"
            "- Office equipment (computer monitor, printer, scanner, desk lamp)\n"
            "- Drawers, filing cabinets, storage cabinets, desk drawers, pedestals\n"
            "- Bedroom items (nightstand, dresser, mirror, jewelry box, hamper)\n"
            "- Nursery items (changing table, baby monitor, mobile, rocking chair)\n"
            "- Items mentioned in audio but hard to see (the audio is RELIABLE)\n"
            "- Items VISIBLE in frames but NOT mentioned in audio — scan each frame!\n\n"
            "FRAMES: Each frame below is labeled with its timestamp.\n\n"
            f"Respond with ONLY valid JSON:\n"
            f"{_INVENTORY_JSON_SCHEMA if has_transcript else _INVENTORY_JSON_SCHEMA_VISUAL}\n"
        )
        return prompt

    def _build_merge_prompt(
        self,
        batch_results: list[str],
        has_transcript: bool = False,
        batch_time_ranges: list[tuple[float, float]] | None = None,
    ) -> str:
        prompt = (
            "You are a moving estimator. Below are partial inventory lists from different "
            "TIME SEGMENTS of the same home video walkthrough. The camera walks through "
            "the home sequentially — each batch covers a different portion of the video.\n\n"
            "TASK: Merge them into ONE deduplicated, complete inventory.\n\n"
        )

        # ── CRITICAL: Room renumbering instructions ──
        prompt += (
            "CRITICAL — ROOM IDENTITY ACROSS BATCHES:\n"
            "Each batch was processed INDEPENDENTLY. This means different batches may have\n"
            "labeled DIFFERENT physical rooms with the SAME name (e.g., both call their\n"
            "bedroom 'bedroom 1' because each batch only saw one bedroom).\n\n"
            "To detect this, compare items across batches that share a room name:\n"
            "- If 'bedroom 1' in batch 1 has a queen bed + nightstand (timestamps ~0-80s)\n"
            "  and 'bedroom 1' in batch 2 has a twin bed + desk (timestamps ~80-160s),\n"
            "  these are DIFFERENT rooms. Renumber: keep batch 1 as 'bedroom 1', rename\n"
            "  batch 2 to 'bedroom 2'.\n"
            "- Same logic applies to bathrooms, closets, and any room type that might\n"
            "  appear multiple times in a home.\n"
            "- Clues that rooms are DIFFERENT: completely different item sets, very different\n"
            "  timestamps (items from separate time segments), different descriptions/notes.\n"
            "- Clues that rooms are the SAME: overlapping items, similar timestamps, same\n"
            "  distinguishing features in notes.\n"
            "- When in doubt, treat them as DIFFERENT rooms (over-splitting is better\n"
            "  than merging two real rooms into one).\n\n"
        )

        prompt += (
            "MERGE RULES:\n"
            "- If the same item appears in multiple batches from the SAME physical room,\n"
            "  keep it ONCE with the HIGHEST count.\n"
            "- If the same item type appears in DIFFERENT rooms, keep separate entries "
            "(e.g., 'nightstand' in 'bedroom 1' AND 'nightstand' in 'bedroom 2').\n"
            "- Preserve room assignments and notes.\n"
        )

        if has_transcript:
            prompt += (
                "- Preserve disposition values (going/staying/scrap/haul/sell) from audio.\n"
                "- KEEP ALL STAYING ITEMS (disposition='staying'). Do NOT remove items just because "
                "they are staying — they are important for the estimate.\n"
                "- If an item has no disposition (not discussed in audio), leave it without a disposition field.\n"
            )
        else:
            prompt += (
                "- Do NOT include a 'disposition' field — this is a visual-only survey with no audio.\n"
            )

        prompt += (
            "- Do NOT drop any items unless they are clear duplicates of the same item "
            "in the same room. When in doubt, KEEP the item.\n"
            "- Do NOT add items not mentioned in any batch.\n"
            "- Keep notes SHORT (under 10 words each) to stay within output limits.\n"
            "- Omit dimensions_approx from the merged output to save space.\n"
            "- PRESERVE best_frame_ts for every item — pick the best one if a merged\n"
            "  item appears in multiple batches. This field is CRITICAL.\n"
            "- RENUMBER room names sequentially after merging (bedroom 1, bedroom 2, etc.).\n\n"
        )

        for i, result in enumerate(batch_results):
            time_label = ""
            if batch_time_ranges and i < len(batch_time_ranges):
                s, e = batch_time_ranges[i]
                time_label = f" (video {self._format_timestamp(s)} – {self._format_timestamp(e)})"
            prompt += f"--- BATCH {i+1}{time_label} ---\n{result}\n\n"

        prompt += (f"Respond with ONLY valid JSON:\n"
                  f"{_MERGE_JSON_SCHEMA if has_transcript else _MERGE_JSON_SCHEMA_VISUAL}\n")
        return prompt

    # ── API Calls ──

    async def _call_gemini(self, prompt: str, frames: list[SelectedFrame] | None = None, max_output_tokens: int = 16000, model_override: str | None = None) -> tuple[str, dict]:
        """Call Gemini with text + optional images. Returns (response_text, usage).

        Retries up to 4 times with exponential backoff on 429/RESOURCE_EXHAUSTED.
        """
        from google import genai
        from google.genai import types
        from PIL import Image

        client = genai.Client(api_key=self.gemini_api_key)
        model_name = model_override or self.gemini_model

        parts: list = [prompt]
        if frames:
            for sf in frames:
                ts = self._format_timestamp(sf.timestamp_s)
                parts.append(f"Frame {sf.frame_index} (at {ts}):")
                frame_rgb = cv2.cvtColor(sf.frame, cv2.COLOR_BGR2RGB)
                parts.append(Image.fromarray(frame_rgb))

        # For 2.5 models, constrain thinking budget to avoid consuming output tokens
        thinking_config = None
        if "2.5" in model_name:
            thinking_config = types.ThinkingConfig(thinking_budget=128)

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=0.0,
            response_mime_type="application/json",
            thinking_config=thinking_config,
        )

        max_retries = 4
        for attempt in range(max_retries + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=parts,
                    config=gen_config,
                )
                break
            except Exception as exc:
                exc_str = str(exc)
                if ("429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str) and attempt < max_retries:
                    delay = 2 ** (attempt + 1)  # 2, 4, 8, 16 seconds
                    logger.warning(f"Gemini rate limited (attempt {attempt+1}/{max_retries+1}), "
                                   f"retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise

        text = response.text or ""
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {
            "tokens_in": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            "tokens_out": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
            "model": model_name,
        }
        return text, usage

    async def _call_openai(self, prompt: str, frames: list[SelectedFrame] | None = None) -> tuple[str, dict]:
        """Call OpenAI with text + optional images. Returns (response_text, usage)."""
        import httpx
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.openai_api_key,
            http_client=httpx.AsyncClient(verify=False),
        )

        content: list[dict] = [{"type": "text", "text": prompt}]
        if frames:
            for sf in frames:
                ts = self._format_timestamp(sf.timestamp_s)
                b64 = self._frame_to_base64(sf.frame)
                content.append({"type": "text", "text": f"Frame {sf.frame_index} (at {ts}):"})
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                })

        response = await client.chat.completions.create(
            model=self.openai_model,
            messages=[{"role": "user", "content": content}],
            max_tokens=8000,
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or ""
        usage = {
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "model": self.openai_model,
        }
        return text, usage

    async def _call_llm(self, prompt: str, frames: list[SelectedFrame] | None = None, max_output_tokens: int = 16000, model_override: str | None = None) -> tuple[str, dict]:
        """Route to the configured LLM provider."""
        if self._provider == "gemini":
            return await self._call_gemini(prompt, frames, max_output_tokens=max_output_tokens, model_override=model_override)
        elif self._provider == "openai":
            return await self._call_openai(prompt, frames)
        else:
            return "{\"items\": []}", {"tokens_in": 0, "tokens_out": 0, "model": "none"}

    # ── Timestamp restoration after LLM merge ──

    def _restore_timestamps(
        self,
        merged_items: list[InventoryItem],
        batch_texts: list[str],
        frame_timestamps: list[float],
        batch_time_ranges: list[tuple[float, float]] | None = None,
    ) -> None:
        """Restore best_frame_ts using batch time ranges as ground truth.

        The merge LLM frequently divides timestamps by ~100 (e.g., 40s → 0.4s).
        These corrupted values often pass simple validation because they land
        near frame 0 at 0.0s.

        Strategy: use deterministic batch time ranges to detect corruption.
        Each item came from a specific batch with a known time range. If the
        merged timestamp falls outside that batch's range, it's corrupted and
        we restore the original batch timestamp.

        Mutates merged_items in place.
        """
        if not frame_timestamps or not batch_time_ranges or not batch_texts:
            return

        # Parse each batch → build lookup: (name, room) → list of (batch_idx, timestamp)
        # Store ALL candidates per key (not just last) so we can pick the best batch.
        batch_item_candidates: dict[tuple[str, str], list[tuple[int, float]]] = {}
        name_batch_lookup: dict[str, list[tuple[int, float]]] = {}

        for batch_idx, text in enumerate(batch_texts):
            batch_items = self._parse_inventory_json(text)
            for item in batch_items:
                if item.best_frame_ts is None:
                    continue
                key = (item.name.lower().strip(), (item.room or "").lower().strip())
                batch_item_candidates.setdefault(key, []).append(
                    (batch_idx, item.best_frame_ts)
                )
                name_batch_lookup.setdefault(
                    item.name.lower().strip(), []
                ).append((batch_idx, item.best_frame_ts))

        def _ts_in_batch_range(ts: float, batch_idx: int) -> bool:
            """Check if timestamp falls within the batch's time range (with tolerance)."""
            if batch_idx >= len(batch_time_ranges):
                return False
            start, end = batch_time_ranges[batch_idx]
            tolerance = 5.0  # seconds of slack
            return (start - tolerance) <= ts <= (end + tolerance)

        def _snap_to_nearest_frame(ts: float) -> float:
            """Find the closest real frame timestamp."""
            return min(frame_timestamps, key=lambda ft: abs(ft - ts))

        def _pick_best_candidate(
            candidates: list[tuple[int, float]],
            merged_ts: float | None,
        ) -> tuple[int, float] | None:
            """Pick the candidate whose batch range best matches the merged timestamp."""
            if not candidates:
                return None
            if len(candidates) == 1:
                return candidates[0]
            # Prefer candidate whose batch range contains the merged timestamp
            if merged_ts is not None:
                for cand_batch_idx, cand_ts in candidates:
                    if _ts_in_batch_range(merged_ts, cand_batch_idx):
                        return (cand_batch_idx, cand_ts)
            # Fallback: pick the candidate whose batch timestamp is in its own batch range
            for cand_batch_idx, cand_ts in candidates:
                if _ts_in_batch_range(cand_ts, cand_batch_idx):
                    return (cand_batch_idx, cand_ts)
            # Last resort: pick the last candidate (latest batch)
            return candidates[-1]

        n_fixed = 0
        n_snapped = 0
        for item in merged_items:
            # Look up which batch this item came from
            key = (item.name.lower().strip(), (item.room or "").lower().strip())
            candidates = batch_item_candidates.get(key, [])

            # Fallback: name-only lookup
            if not candidates:
                name_key = item.name.lower().strip()
                candidates = name_batch_lookup.get(name_key, [])

            source = _pick_best_candidate(candidates, item.best_frame_ts)

            if source is None:
                # Item not found in any batch — can't validate, keep as-is
                continue

            src_batch_idx, batch_ts = source

            # Check: is the merged timestamp within the correct batch range?
            if item.best_frame_ts is not None and _ts_in_batch_range(
                item.best_frame_ts, src_batch_idx
            ):
                # Merged timestamp is in the right range — keep it
                continue

            # Merged timestamp is OUTSIDE the batch range — it's corrupted.
            # Try the batch's original timestamp first.
            if _ts_in_batch_range(batch_ts, src_batch_idx):
                old_ts = item.best_frame_ts
                item.best_frame_ts = _snap_to_nearest_frame(batch_ts)
                n_fixed += 1
                logger.debug(
                    f"Restored ts for '{item.name}' [{item.room}]: "
                    f"{old_ts} → {item.best_frame_ts} "
                    f"(batch {src_batch_idx+1} range "
                    f"{batch_time_ranges[src_batch_idx][0]:.0f}-"
                    f"{batch_time_ranges[src_batch_idx][1]:.0f}s)"
                )
            else:
                # Both timestamps are bad — snap to batch midpoint
                start, end = batch_time_ranges[src_batch_idx]
                mid = (start + end) / 2.0
                old_ts = item.best_frame_ts
                item.best_frame_ts = _snap_to_nearest_frame(mid)
                n_snapped += 1
                logger.debug(
                    f"Snapped ts for '{item.name}' [{item.room}]: "
                    f"{old_ts} → {item.best_frame_ts} (batch {src_batch_idx+1} midpoint)"
                )

        total_fixed = n_fixed + n_snapped
        if total_fixed:
            logger.info(
                f"Timestamp restoration: fixed {n_fixed}, snapped {n_snapped} "
                f"/ {len(merged_items)} items"
            )
        else:
            logger.info(
                f"Timestamp restoration: all {len(merged_items)} items "
                f"within correct batch ranges"
            )

    # ── Programmatic merge fallback ──

    def _programmatic_merge(
        self, batch_texts: list[str], has_transcript: bool
    ) -> list[InventoryItem]:
        """Combine batch results without LLM — simple dedup by (name, room).

        When items share the same name+room, keep the entry with the highest
        count and merge notes.  Runs in <1ms, never truncates.
        """
        # Parse all batches
        all_items: list[InventoryItem] = []
        for text in batch_texts:
            all_items.extend(self._parse_inventory_json(text))

        if not all_items:
            return []

        # Dedup by (name, room) — keep highest count, merge notes/disposition
        merged: dict[tuple[str, str], InventoryItem] = {}
        for item in all_items:
            key = (item.name, item.room)
            if key not in merged:
                merged[key] = item
            else:
                existing = merged[key]
                if item.count > existing.count:
                    existing.count = item.count
                if item.disposition and not existing.disposition:
                    existing.disposition = item.disposition
                    existing.going = item.going
                if item.notes and item.notes not in (existing.notes or ""):
                    existing.notes = (
                        f"{existing.notes}; {item.notes}" if existing.notes
                        else item.notes
                    )

        result = list(merged.values())
        logger.info(f"Programmatic merge: {len(all_items)} batch items → "
                    f"{len(result)} merged items")
        return result

    # ── Programmatic merge v2 — fuzzy name matching + frame_idx preservation ──

    def _programmatic_merge_v2(
        self,
        batch_texts: list[str],
        has_transcript: bool,
        frame_timestamps: list[float] | None = None,
        selected_frames: list[SelectedFrame] | None = None,
    ) -> list[InventoryItem]:
        """Merge batch results programmatically with fuzzy name matching.

        Improvements over v1:
        - Fuzzy name matching via SequenceMatcher (handles 'king bed' vs 'king size bed')
        - Always preserves best_frame_idx from original batch (never trusts merge LLM)
        - Snaps best_frame_ts to match frame_idx if available
        - Logs detailed merge decisions for debugging
        """
        import re
        from difflib import SequenceMatcher

        # Parse all batches, tagging each item with its source batch
        all_items: list[tuple[int, InventoryItem]] = []
        for batch_idx, text in enumerate(batch_texts):
            for item in self._parse_inventory_json(text):
                all_items.append((batch_idx, item))

        if not all_items:
            return []

        def _normalize_name(name: str) -> str:
            """Normalize item name for comparison."""
            name = name.lower().strip()
            # Remove common size/color prefixes that don't change identity
            name = re.sub(r'\b(small|medium|large|big|little)\b', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
            return name

        def _names_match(a: str, b: str, threshold: float = 0.82) -> bool:
            """Check if two item names refer to the same item."""
            na, nb = _normalize_name(a), _normalize_name(b)
            if na == nb:
                return True
            # Token-set similarity (handles word reordering)
            tokens_a, tokens_b = set(na.split()), set(nb.split())
            if tokens_a and tokens_b:
                jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                if jaccard >= 0.75:
                    return True
            return SequenceMatcher(None, na, nb).ratio() >= threshold

        # Group by room, then fuzzy-dedup within each room
        room_items: dict[str, list[tuple[int, InventoryItem]]] = {}
        for batch_idx, item in all_items:
            room = (item.room or "").lower().strip()
            room_items.setdefault(room, []).append((batch_idx, item))

        merged: list[InventoryItem] = []
        merge_log: list[str] = []

        for room, items_in_room in room_items.items():
            # For each item in this room, try to match with existing merged items
            room_merged: list[InventoryItem] = []

            for batch_idx, item in items_in_room:
                matched = False
                for existing in room_merged:
                    if _names_match(existing.name, item.name):
                        # Duplicate — merge: keep higher count, better frame ref
                        old_count = existing.count
                        if item.count > existing.count:
                            existing.count = item.count
                        # Prefer the entry with a valid best_frame_idx
                        if item.best_frame_idx is not None and existing.best_frame_idx is None:
                            existing.best_frame_idx = item.best_frame_idx
                            existing.best_frame_ts = item.best_frame_ts
                        if item.disposition and not existing.disposition:
                            existing.disposition = item.disposition
                            existing.going = item.going
                        if item.notes and item.notes not in (existing.notes or ""):
                            existing.notes = (
                                f"{existing.notes}; {item.notes}" if existing.notes
                                else item.notes
                            )
                        merge_log.append(
                            f"  DEDUP [{room}]: '{item.name}' (batch {batch_idx+1}, "
                            f"count={item.count}) merged into '{existing.name}' "
                            f"(count {old_count}→{existing.count})"
                        )
                        matched = True
                        break

                if not matched:
                    room_merged.append(item)

            merged.extend(room_merged)

        # Snap best_frame_ts to match frame_idx using SelectedFrame.frame_index lookup
        if selected_frames:
            fidx_to_ts = {f.frame_index: f.timestamp_s for f in selected_frames}
            for item in merged:
                if item.best_frame_idx is not None and item.best_frame_idx in fidx_to_ts:
                    item.best_frame_ts = fidx_to_ts[item.best_frame_idx]

        n_input = len(all_items)
        n_output = len(merged)
        logger.info(f"Programmatic merge v2: {n_input} batch items → {n_output} merged items "
                    f"({n_input - n_output} deduped)")
        if merge_log:
            for line in merge_log:
                logger.debug(line)

        return merged

    # ── Room disambiguation (text-only LLM call) ──

    async def _disambiguate_rooms(
        self,
        batch_texts: list[str],
        batch_time_ranges: list[tuple[float, float]],
    ) -> tuple[list[str], dict]:
        """Disambiguate room names across batches using a lightweight text-only LLM call.

        Each batch independently names rooms, so the same room name in different
        batches may refer to different physical rooms (e.g., two different bedrooms
        both called "bedroom 1").  This method asks the LLM to identify such
        collisions by comparing item sets across batches.

        Returns (corrected_batch_texts, usage_dict).
        On any failure, returns the original batch_texts unchanged with zero usage.
        """
        zero_usage = {"tokens_in": 0, "tokens_out": 0}
        # Parse items per batch to build the summary
        batch_items: list[list[InventoryItem]] = []
        for text in batch_texts:
            batch_items.append(self._parse_inventory_json(text))

        # Collect all room names across batches, keyed by (room, batch_idx)
        room_batches: dict[str, list[int]] = {}
        for b_idx, items in enumerate(batch_items):
            for item in items:
                room = (item.room or "").strip().lower()
                if room:
                    room_batches.setdefault(room, [])
                    if b_idx not in room_batches[room]:
                        room_batches[room].append(b_idx)

        # Only rooms appearing in 2+ batches need disambiguation
        collision_rooms = {r for r, batches in room_batches.items() if len(batches) >= 2}
        if not collision_rooms:
            logger.info("Room disambiguation: no cross-batch room collisions — skipping")
            return batch_texts, zero_usage

        logger.info(f"Room disambiguation: {len(collision_rooms)} room(s) appear in multiple batches: "
                    f"{sorted(collision_rooms)}")

        # Build a compact text summary for the LLM
        summary_parts = []
        for b_idx, items in enumerate(batch_items):
            if not items:
                continue
            s, e = batch_time_ranges[b_idx] if b_idx < len(batch_time_ranges) else (0, 0)
            time_label = f"{self._format_timestamp(s)}–{self._format_timestamp(e)}"

            # Group items by room
            rooms_in_batch: dict[str, list[str]] = {}
            for item in items:
                room = (item.room or "unknown").strip().lower()
                rooms_in_batch.setdefault(room, [])
                label = item.name
                if item.count > 1:
                    label += f" ×{item.count}"
                rooms_in_batch[room].append(label)

            lines = [f"Batch {b_idx + 1} ({time_label}):"]
            for room, item_names in sorted(rooms_in_batch.items()):
                lines.append(f"  {room}: {', '.join(item_names)}")
            summary_parts.append("\n".join(lines))

        batch_summary = "\n\n".join(summary_parts)

        # Collect all existing room names to avoid collisions when renaming
        all_room_names = sorted(room_batches.keys())

        prompt = (
            "You are analyzing a walkthrough video of a home. The video was processed in "
            f"{len(batch_items)} time-based batches. Each batch independently assigned room "
            "names to items it detected.\n\n"
            "The SAME room name in DIFFERENT batches might refer to DIFFERENT physical rooms "
            "(each batch doesn't know what other batches named rooms).\n\n"
            "Your task: Identify cases where the same room name across batches actually refers to "
            "different physical rooms, and provide corrected room names.\n\n"
            "RULES:\n"
            "- If items OVERLAP significantly between batches (same bed type, same key furniture), "
            "it's the SAME room — do NOT rename.\n"
            "- If items are COMPLETELY DIFFERENT (queen bed vs bunk bed, different furniture), "
            "it's likely DIFFERENT rooms — rename the later batch's room.\n"
            "- Adjacent time batches with the same room name are MORE LIKELY the same room "
            "(continuous walkthrough).\n"
            "- When UNSURE, do NOT rename. Keeping rooms merged is safer than incorrect splitting.\n"
            "- Only provide room corrections. Do not rename items.\n"
            "- When renaming, ensure the new name doesn't collide with any existing room name "
            f"in any batch. Existing rooms: {all_room_names}\n\n"
            f"{batch_summary}\n\n"
            "Respond with ONLY valid JSON:\n"
            '{"corrections": [\n'
            '  {"batch": <1-based batch number>, "original_room": "<current name>", '
            '"new_room": "<corrected name>"},\n'
            "  ...\n"
            "]}\n"
            "Return empty corrections [] if no rooms need renaming.\n"
        )

        try:
            text, usage = await self._call_llm(prompt, max_output_tokens=2000)
            logger.debug(f"Room disambiguation raw response: {text[:500]}")

            data = json.loads(text)
            corrections = data.get("corrections", [])

            if not corrections:
                logger.info("Room disambiguation: LLM found no rooms to rename")
                return batch_texts, usage

            # Validate and apply corrections
            valid_corrections: list[dict] = []
            for corr in corrections:
                batch_num = corr.get("batch")
                orig_room = (corr.get("original_room") or "").strip().lower()
                new_room = (corr.get("new_room") or "").strip().lower()

                if not isinstance(batch_num, int) or batch_num < 1 or batch_num > len(batch_texts):
                    logger.warning(f"Room disambiguation: invalid batch number {batch_num}, skipping")
                    continue
                if not orig_room or not new_room or orig_room == new_room:
                    continue

                # Check that orig_room actually exists in that batch
                b_idx = batch_num - 1
                batch_rooms = {(item.room or "").strip().lower() for item in batch_items[b_idx]}
                if orig_room not in batch_rooms:
                    logger.warning(f"Room disambiguation: room '{orig_room}' not found in batch {batch_num}, skipping")
                    continue

                valid_corrections.append({
                    "batch_idx": b_idx,
                    "original_room": orig_room,
                    "new_room": new_room,
                })

            if not valid_corrections:
                logger.info("Room disambiguation: no valid corrections after validation")
                return batch_texts, usage

            # Apply corrections to the raw batch JSON texts
            corrected_texts = list(batch_texts)
            for corr in valid_corrections:
                b_idx = corr["batch_idx"]
                corrected_texts[b_idx] = self._apply_room_correction(
                    corrected_texts[b_idx], corr["original_room"], corr["new_room"]
                )
                logger.info(
                    f"Room disambiguation: batch {b_idx + 1}: "
                    f"'{corr['original_room']}' → '{corr['new_room']}'"
                )

            return corrected_texts, usage

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Room disambiguation: failed to parse LLM response ({e}), "
                           "using original room names")
            return batch_texts, zero_usage
        except Exception as e:
            logger.warning(f"Room disambiguation: unexpected error ({e}), "
                           "using original room names")
            return batch_texts, zero_usage

    @staticmethod
    def _apply_room_correction(batch_json: str, old_room: str, new_room: str) -> str:
        """Replace a room name in a batch JSON response string.

        Operates on the parsed JSON to ensure only the 'room' fields are changed,
        then re-serializes.
        """
        try:
            data = json.loads(batch_json)
        except json.JSONDecodeError:
            # Try extracting from markdown code blocks
            import re as _re
            match = _re.search(r"```(?:json)?\s*(.*?)```", batch_json, _re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    return batch_json
            else:
                return batch_json

        items_list = data.get("items", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        changed = 0
        for item in items_list:
            if isinstance(item, dict):
                room_val = (item.get("room") or "").strip().lower()
                if room_val == old_room:
                    item["room"] = new_room
                    changed += 1

        if changed == 0:
            return batch_json

        if isinstance(data, dict):
            data["items"] = items_list
        return json.dumps(data, indent=2)

    # ── Merge comparison logging ──

    def _compare_merge_results(
        self,
        llm_items: list[InventoryItem],
        prog_items: list[InventoryItem],
        frame_timestamps: list[float],
    ) -> None:
        """Log a detailed comparison of LLM merge vs programmatic merge results.

        This runs during the A/B testing phase so we can decide which approach
        to keep. Compares item counts, names, rooms, and frame assignments.
        """
        llm_set = {(i.name, i.room) for i in llm_items}
        prog_set = {(i.name, i.room) for i in prog_items}

        only_llm = llm_set - prog_set
        only_prog = prog_set - llm_set
        common = llm_set & prog_set

        logger.info(
            f"=== MERGE COMPARISON: LLM={len(llm_items)} items, "
            f"PROG={len(prog_items)} items ==="
        )
        logger.info(
            f"  Common: {len(common)}, Only in LLM: {len(only_llm)}, "
            f"Only in PROG: {len(only_prog)}"
        )

        if only_llm:
            logger.info(f"  Items ONLY in LLM merge: {sorted(only_llm)}")
        if only_prog:
            logger.info(f"  Items ONLY in PROG merge: {sorted(only_prog)}")

        # Compare frame assignments for common items
        llm_lookup = {(i.name, i.room): i for i in llm_items}
        prog_lookup = {(i.name, i.room): i for i in prog_items}

        frame_mismatches = []
        for key in sorted(common):
            li = llm_lookup[key]
            pi = prog_lookup[key]

            # Compare frame indices
            llm_frame = li.best_frame_ts
            prog_frame = pi.best_frame_idx
            prog_ts = pi.best_frame_ts

            if llm_frame is not None and prog_ts is not None:
                if abs(llm_frame - prog_ts) > 5.0:  # >5s difference
                    frame_mismatches.append(
                        f"  '{key[0]}' [{key[1]}]: LLM ts={llm_frame:.1f}s, "
                        f"PROG idx={prog_frame} ts={prog_ts:.1f}s"
                    )

        if frame_mismatches:
            logger.info(f"  Frame assignment mismatches (>5s diff): {len(frame_mismatches)}")
            for line in frame_mismatches:
                logger.info(line)
        else:
            logger.info("  Frame assignments: all within 5s agreement")

        # Count comparison
        count_diffs = []
        for key in sorted(common):
            li = llm_lookup[key]
            pi = prog_lookup[key]
            if li.count != pi.count:
                count_diffs.append(
                    f"  '{key[0]}' [{key[1]}]: LLM count={li.count}, PROG count={pi.count}"
                )
        if count_diffs:
            logger.info(f"  Count differences: {len(count_diffs)}")
            for line in count_diffs:
                logger.info(line)

    # ── Parsing ──

    def _parse_inventory_json(self, text: str) -> list[InventoryItem]:
        """Parse JSON inventory response into InventoryItem list."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON: {text[:300]}")
                    return []
            else:
                logger.error(f"Failed to parse JSON: {text[:300]}")
                return []

        items_data = data.get("items", []) if isinstance(data, dict) else data if isinstance(data, list) else []
        items = []
        for item in items_data:
            name = item.get("name", "").strip().lower()
            if not name:
                continue
            count = item.get("count", 1)
            if isinstance(count, str):
                count = int("".join(c for c in count if c.isdigit()) or "1")
            count = max(1, count)

            dims = item.get("dimensions_approx")
            if isinstance(dims, dict):
                dims = {
                    "length_in": dims.get("length_in", 0),
                    "width_in": dims.get("width_in", 0),
                    "height_in": dims.get("height_in", 0),
                }
            else:
                dims = None

            # Parse disposition (new) and compute going (compat)
            raw_disp = item.get("disposition")
            disposition = raw_disp if raw_disp in DISPOSITION_VALUES else None
            # Also check legacy "going" field from older prompts
            if disposition is None and item.get("going") is not None:
                disposition = "going" if item["going"] else "staying"
            # Compute compat going flag
            going = True if disposition == "going" else (False if disposition == "staying" else None)

            items.append(InventoryItem(
                name=name,
                count=count,
                room=(item.get("room") or "").strip().lower(),
                disposition=disposition,
                going=going,
                notes=(item.get("notes") or "").strip(),
                size=(item.get("size") or "medium").strip().lower(),
                dimensions=dims,
                best_frame_ts=item.get("best_frame_ts"),
                best_frame_idx=item.get("best_frame_idx"),
                centroid=None,  # centroid is computed separately in report_generator
            ))
        return items

    # ── Main Draft Pipeline ──

    async def draft(
        self,
        selected_frames: list[SelectedFrame],
        transcript: TranscriptionResult | None = None,
        duration_s: float = 0,
    ) -> DraftInventory:
        """
        Create a comprehensive inventory draft using time-batched LLM calls.

        Args:
            selected_frames: Quality-filtered, diverse frames with timestamps
            transcript: Optional audio transcript
            duration_s: Total video duration

        Returns:
            DraftInventory with items, detection prompts, and usage stats
        """
        result = DraftInventory()

        if not selected_frames:
            return result

        if duration_s <= 0:
            duration_s = selected_frames[-1].timestamp_s + 30

        # ── Split frames into time-based batches ──
        batch_size = self._batch_size
        num_batches = max(1, math.ceil(len(selected_frames) / batch_size))
        batch_duration = duration_s / num_batches

        batches: list[tuple[list[SelectedFrame], float, float]] = []
        for b in range(num_batches):
            batch_start = b * batch_duration
            batch_end = (b + 1) * batch_duration
            batch_frames = [
                f for f in selected_frames
                if batch_start <= f.timestamp_s < batch_end
            ]
            # Handle frames at the very end
            if b == num_batches - 1:
                batch_frames = [
                    f for f in selected_frames
                    if f.timestamp_s >= batch_start
                ]
            if batch_frames:
                batches.append((batch_frames, batch_start, batch_end))

        logger.info(f"Split {len(selected_frames)} frames into {len(batches)} batches "
                    f"({batch_size} frames/batch)")

        # ── Run batch LLM calls in parallel ──
        async def process_batch(batch_frames, start_s, end_s, batch_idx):
            # Stagger requests to avoid hitting rate limits
            if batch_idx > 0:
                await asyncio.sleep(batch_idx * 0.5)
            prompt = self._build_batch_prompt(
                batch_frames, transcript, start_s, end_s, duration_s,
                batch_index=batch_idx, num_batches=len(batches),
            )
            logger.info(f"Batch {batch_idx+1}/{len(batches)}: "
                        f"{len(batch_frames)} frames, "
                        f"{self._format_timestamp(start_s)}-{self._format_timestamp(end_s)}")
            text, usage = await self._call_llm(prompt, batch_frames)
            return text, usage

        batch_tasks = [
            process_batch(frames, start, end, i)
            for i, (frames, start, end) in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        # Accumulate usage
        total_usage = {"tokens_in": 0, "tokens_out": 0, "model": self._provider}
        raw_responses = []
        for text, usage in batch_results:
            total_usage["tokens_in"] += usage.get("tokens_in", 0)
            total_usage["tokens_out"] += usage.get("tokens_out", 0)
            total_usage["model"] = usage.get("model", self._provider)
            raw_responses.append(text)

        # ── Merge if multiple batches ──
        if len(batch_results) == 1:
            # Single batch — parse directly
            all_items = self._parse_inventory_json(batch_results[0][0])
        else:
            has_transcript = transcript is not None and transcript.has_speech
            batch_time_ranges = [(s, e) for (_, s, e) in batches]
            frame_ts = [f.timestamp_s for f in selected_frames]

            # ── Approach A: LLM merge (existing) ──
            merge_prompt = self._build_merge_prompt(
                raw_responses, has_transcript=has_transcript,
                batch_time_ranges=batch_time_ranges,
            )
            merge_text, merge_usage = await self._call_llm(
                merge_prompt, max_output_tokens=32000
            )
            total_usage["tokens_in"] += merge_usage.get("tokens_in", 0)
            total_usage["tokens_out"] += merge_usage.get("tokens_out", 0)
            raw_responses.append(f"--- MERGE ---\n{merge_text}")
            llm_items = self._parse_inventory_json(merge_text)

            # Fallback: if merge JSON was truncated/invalid, combine batch results
            if not llm_items:
                logger.warning("LLM merge returned no items — falling back to "
                               "programmatic batch combination")
                llm_items = self._programmatic_merge(raw_responses[:-1], has_transcript)

            # Restore timestamps that the merge LLM may have corrupted
            self._restore_timestamps(
                llm_items, raw_responses[:-1], frame_ts,
                batch_time_ranges=batch_time_ranges,
            )

            # ── Approach B: Programmatic merge v2 with room disambiguation ──
            # Step 1: Disambiguate room names across batches (text-only LLM call)
            batch_only_responses = raw_responses[:-1]  # exclude merge response
            disambiguated_responses, disambig_usage = await self._disambiguate_rooms(
                batch_only_responses, batch_time_ranges,
            )
            total_usage["tokens_in"] += disambig_usage.get("tokens_in", 0)
            total_usage["tokens_out"] += disambig_usage.get("tokens_out", 0)

            # Step 2: Run programmatic merge on disambiguated batch texts
            prog_items = self._programmatic_merge_v2(
                disambiguated_responses, has_transcript,
                frame_timestamps=frame_ts, selected_frames=selected_frames,
            )

            # ── Compare and log both approaches ──
            self._compare_merge_results(llm_items, prog_items, frame_ts)

            # Use programmatic merge (preserves frame_idx from batch originals)
            all_items = prog_items

        n_going = len([i for i in all_items if i.disposition == "going"])
        n_staying = len([i for i in all_items if i.disposition == "staying"])
        n_other = len([i for i in all_items if i.disposition in ("scrap", "haul", "sell")])
        n_undecided = len([i for i in all_items if i.disposition is None])
        logger.info(f"Draft inventory: {len(all_items)} items "
                    f"({n_going} going, {n_staying} staying, {n_other} scrap/haul/sell, "
                    f"{n_undecided} undecided)")

        # If no transcript, force disposition=None — LLM may still hallucinate
        has_transcript = transcript is not None and transcript.has_speech
        if not has_transcript:
            for item in all_items:
                item.disposition = None
                item.going = None

        # ── Build detection prompts (all items except staying) ──
        prompts = list(set(item.name for item in all_items if item.disposition != "staying"))

        result.items = all_items
        result.detection_prompts = prompts
        result.usage = total_usage
        result.raw_responses = raw_responses

        return result

    async def draft_with_model(
        self,
        model_name: str,
        selected_frames: list[SelectedFrame],
        transcript: TranscriptionResult | None = None,
        duration_s: float = 0,
    ) -> DraftInventory:
        """Run the same draft pipeline but with a specific model override.

        Used for background comparison runs — same frames, different model.
        """
        result = DraftInventory()
        if not selected_frames:
            return result
        if duration_s <= 0:
            duration_s = selected_frames[-1].timestamp_s + 30

        batch_size = self._batch_size
        num_batches = max(1, math.ceil(len(selected_frames) / batch_size))
        batch_duration = duration_s / num_batches

        batches: list[tuple[list[SelectedFrame], float, float]] = []
        for b in range(num_batches):
            batch_start = b * batch_duration
            batch_end = (b + 1) * batch_duration
            batch_frames = [
                f for f in selected_frames
                if batch_start <= f.timestamp_s < batch_end
            ]
            if b == num_batches - 1:
                batch_frames = [f for f in selected_frames if f.timestamp_s >= batch_start]
            if batch_frames:
                batches.append((batch_frames, batch_start, batch_end))

        async def process_batch(batch_frames, start_s, end_s, batch_idx):
            if batch_idx > 0:
                await asyncio.sleep(batch_idx * 0.5)
            prompt = self._build_batch_prompt(
                batch_frames, transcript, start_s, end_s, duration_s,
                batch_index=batch_idx, num_batches=len(batches),
            )
            text, usage = await self._call_llm(prompt, batch_frames, model_override=model_name)
            return text, usage

        batch_tasks = [
            process_batch(frames, start, end, i)
            for i, (frames, start, end) in enumerate(batches)
        ]
        batch_results = await asyncio.gather(*batch_tasks)

        total_usage = {"tokens_in": 0, "tokens_out": 0, "model": model_name}
        raw_responses = []
        for text, usage in batch_results:
            total_usage["tokens_in"] += usage.get("tokens_in", 0)
            total_usage["tokens_out"] += usage.get("tokens_out", 0)
            raw_responses.append(text)

        if len(batch_results) == 1:
            all_items = self._parse_inventory_json(batch_results[0][0])
        else:
            has_transcript = transcript is not None and transcript.has_speech
            batch_time_ranges = [(s, e) for (_, s, e) in batches]
            frame_ts = [f.timestamp_s for f in selected_frames]
            merge_prompt = self._build_merge_prompt(
                raw_responses, has_transcript=has_transcript,
                batch_time_ranges=batch_time_ranges,
            )
            merge_text, merge_usage = await self._call_llm(
                merge_prompt, max_output_tokens=32000, model_override=model_name
            )
            total_usage["tokens_in"] += merge_usage.get("tokens_in", 0)
            total_usage["tokens_out"] += merge_usage.get("tokens_out", 0)
            raw_responses.append(f"--- MERGE ---\n{merge_text}")
            llm_items = self._parse_inventory_json(merge_text)
            if not llm_items:
                llm_items = self._programmatic_merge(raw_responses[:-1], has_transcript)
            self._restore_timestamps(
                llm_items, raw_responses[:-1], frame_ts,
                batch_time_ranges=batch_time_ranges,
            )
            # Disambiguate room names before programmatic merge
            disambiguated_responses, disambig_usage = await self._disambiguate_rooms(
                raw_responses[:-1], batch_time_ranges,
            )
            total_usage["tokens_in"] += disambig_usage.get("tokens_in", 0)
            total_usage["tokens_out"] += disambig_usage.get("tokens_out", 0)
            prog_items = self._programmatic_merge_v2(
                disambiguated_responses, has_transcript,
                frame_timestamps=frame_ts, selected_frames=selected_frames,
            )
            self._compare_merge_results(llm_items, prog_items, frame_ts)
            all_items = prog_items

        has_transcript = transcript is not None and transcript.has_speech
        if not has_transcript:
            for item in all_items:
                item.disposition = None
                item.going = None

        prompts = list(set(item.name for item in all_items if item.disposition != "staying"))
        result.items = all_items
        result.detection_prompts = prompts
        result.usage = total_usage
        result.raw_responses = raw_responses
        return result

    # ── Move Summary ──

    async def generate_move_summary(
        self,
        inventory: DraftInventory,
        transcript: TranscriptionResult | None = None,
    ) -> dict:
        """Generate a structured move summary from inventory + transcript.

        Returns a JSON-serializable dict with overall summary, per-room breakdowns,
        and special instructions. This is a text-only LLM call (no frames).
        """
        has_audio = transcript is not None and transcript.has_speech

        # Build item list by room
        rooms: dict[str, list[InventoryItem]] = {}
        for item in inventory.items:
            room = item.room or "unknown"
            rooms.setdefault(room, []).append(item)

        items_text = ""
        for room, room_items in rooms.items():
            items_text += f"\n{room}:\n"
            for it in room_items:
                disp = f" [{it.disposition}]" if it.disposition else ""
                items_text += f"  - {it.name} x{it.count}{disp}"
                if it.notes:
                    items_text += f" ({it.notes})"
                items_text += "\n"

        prompt = (
            "You are a PROFESSIONAL MOVING ESTIMATOR. Based on the inventory and "
            "any available audio transcript, produce a MOVE SUMMARY.\n\n"
            f"INVENTORY ({len(inventory.items)} item types across {len(rooms)} rooms):\n"
            f"{items_text}\n"
        )

        if has_audio:
            prompt += (
                "AUDIO TRANSCRIPT (full):\n"
                f"{transcript.formatted_transcript()}\n\n"
                "The transcript is from a live call between the homeowner and a moving\n"
                "estimator walking through each room. It may be in any language.\n\n"
            )
        else:
            prompt += "NO AUDIO WAS AVAILABLE — this was a visual-only survey.\n\n"

        prompt += (
            "Produce a JSON summary with this EXACT structure:\n"
            "{\n"
            '  "has_audio": true/false,\n'
            '  "audio_summary": "Brief one-liner about what the audio covered",\n'
            '  "overall": "2-3 sentence summary of the entire move — property type, '
            'scale, key highlights",\n'
            '  "rooms": [\n'
            "    {\n"
            '      "name": "room name",\n'
            '      "summary": "1-2 sentences about this room\'s contents and decisions",\n'
            '      "instructions": ["any packing/handling notes specific to this room"]\n'
            "    }\n"
            "  ],\n"
            '  "special_instructions": [\n'
            '    "Important notes about the move: fragile items, heavy items requiring '
            'extra crew, disassembly required, items needing special crating, parking '
            'or access concerns mentioned in audio, tight staircases, elevator-only, etc."\n'
            "  ]\n"
            "}\n\n"
            "RULES:\n"
            "- Keep the overall summary concise but informative (2-3 sentences max).\n"
            "- For each room, summarize what items are there and any key decisions.\n"
            "- Only include special_instructions that are genuinely important for the "
            "moving crew — do not pad with generic advice.\n"
            "- If there is no audio, the audio_summary should say 'No speech detected "
            "— visual-only survey' and room summaries should focus on what was seen.\n"
            "- If the transcript is in a non-English language, still write the summary "
            "in English.\n"
            "- Respond with ONLY valid JSON.\n"
        )

        try:
            text, usage = await self._call_llm(prompt)
            data = json.loads(text)
            # Ensure required keys
            data.setdefault("has_audio", has_audio)
            data.setdefault("audio_summary",
                            "No speech detected — visual-only survey" if not has_audio
                            else "Audio walkthrough recorded")
            data.setdefault("overall", "")
            data.setdefault("rooms", [])
            data.setdefault("special_instructions", [])
            logger.info(f"Move summary generated: {len(data.get('rooms', []))} rooms, "
                        f"{len(data.get('special_instructions', []))} instructions")
            return data
        except Exception as e:
            logger.warning(f"Failed to generate move summary: {e}")
            # Return a minimal fallback
            n_going = len([i for i in inventory.items if i.disposition == "going"])
            n_staying = len([i for i in inventory.items if i.disposition == "staying"])
            return {
                "has_audio": has_audio,
                "audio_summary": (
                    "No speech detected — visual-only survey" if not has_audio
                    else "Audio walkthrough recorded"
                ),
                "overall": (
                    f"Survey found {len(inventory.items)} item types across "
                    f"{len(rooms)} rooms."
                    + (f" {n_going} items going, {n_staying} staying." if has_audio else "")
                ),
                "rooms": [
                    {"name": r, "summary": f"{len(items)} items catalogued.", "instructions": []}
                    for r, items in rooms.items()
                ],
                "special_instructions": [],
            }
