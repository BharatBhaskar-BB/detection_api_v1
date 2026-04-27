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
      "notes": "black rolling chairs around conference table"
    },
    {
      "name": "laptop",
      "count": 2,
      "room": "office",
      "size": "small",
      "dimensions_approx": {"length_in": 14, "width_in": 10, "height_in": 1},
      "best_frame_ts": 8.3,
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
    ) -> str:
        has_transcript = transcript is not None and transcript.has_speech

        prompt = (
            "You are a PROFESSIONAL MOVING ESTIMATOR reviewing a home video walkthrough.\n\n"
            f"This batch covers {self._format_timestamp(batch_start_s)} - "
            f"{self._format_timestamp(batch_end_s)} of a "
            f"{self._format_timestamp(total_duration_s)} video.\n\n"
            "TASK: List EVERY item in these frames — both items being MOVED and items\n"
            "that are STAYING. A complete estimate requires knowing ALL items.\n"
            "For each item provide:\n"
            "- name: specific item name (lowercase)\n"
            "- count: how many you see\n"
            "- room: which room it's in (e.g., 'living room', 'kitchen', 'master bedroom')\n"
            "- disposition: ONLY set this if the homeowner EXPLICITLY states the item's fate\n"
            "  in the audio. Valid values: 'going', 'staying', 'scrap', 'haul', 'sell'.\n"
            "  If the item is NOT discussed in the audio, OMIT this field entirely.\n"
            "- size: 'small', 'medium', or 'large' relative to typical furniture\n"
            "- dimensions_approx: estimated dimensions in inches {length_in, width_in, height_in}\n"
            "- best_frame_ts: the timestamp (seconds) of the frame where this item is MOST visible\n"
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
                "   kitchen, bedroom, etc.).\n"
                "5. Do NOT return the example JSON — analyze the ACTUAL frames provided.\n"
                "6. Do NOT include a 'going' field in your JSON output — without audio\n"
                "   we cannot determine whether items are being moved or staying. The JSON\n"
                "   schema below does NOT have a 'going' field — follow it exactly.\n"
                "7. For each item, estimate its SIZE ('small', 'medium', 'large') and\n"
                "   approximate DIMENSIONS in inches (length, width, height).\n"
                "8. Set best_frame_ts to the timestamp (in seconds) of the frame where\n"
                "   the item is MOST clearly visible.\n\n"
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

    def _build_merge_prompt(self, batch_results: list[str], has_transcript: bool = False) -> str:
        prompt = (
            "You are a moving estimator. Below are partial inventory lists from different "
            "sections of the same home video walkthrough.\n\n"
            "TASK: Merge them into ONE deduplicated, complete inventory.\n\n"
            "RULES:\n"
            "- If the same item appears in multiple batches from the same room, keep it ONCE "
            "with the HIGHEST count.\n"
            "- If the same item type appears in DIFFERENT rooms, keep separate entries "
            "(e.g., 'nightstand' in 'master bedroom' AND 'nightstand' in 'guest bedroom').\n"
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
            "- Omit dimensions_approx from the merged output to save space.\n\n"
        )

        for i, result in enumerate(batch_results):
            prompt += f"--- BATCH {i+1} ---\n{result}\n\n"

        prompt += (f"Respond with ONLY valid JSON:\n"
                  f"{_MERGE_JSON_SCHEMA if has_transcript else _MERGE_JSON_SCHEMA_VISUAL}\n")
        return prompt

    # ── API Calls ──

    async def _call_gemini(self, prompt: str, frames: list[SelectedFrame] | None = None, max_output_tokens: int = 16000) -> tuple[str, dict]:
        """Call Gemini with text + optional images. Returns (response_text, usage).

        Retries up to 4 times with exponential backoff on 429/RESOURCE_EXHAUSTED.
        """
        from google import genai
        from google.genai import types
        from PIL import Image

        client = genai.Client(api_key=self.gemini_api_key)

        parts: list = [prompt]
        if frames:
            for sf in frames:
                ts = self._format_timestamp(sf.timestamp_s)
                parts.append(f"Frame at {ts}:")
                frame_rgb = cv2.cvtColor(sf.frame, cv2.COLOR_BGR2RGB)
                parts.append(Image.fromarray(frame_rgb))

        # For 2.5 models, constrain thinking budget to avoid consuming output tokens
        thinking_config = None
        if "2.5" in self.gemini_model:
            thinking_config = types.ThinkingConfig(thinking_budget=1024)

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
                    model=self.gemini_model,
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
            "model": self.gemini_model,
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
                content.append({"type": "text", "text": f"Frame at {ts}:"})
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

    async def _call_llm(self, prompt: str, frames: list[SelectedFrame] | None = None, max_output_tokens: int = 16000) -> tuple[str, dict]:
        """Route to the configured LLM provider."""
        if self._provider == "gemini":
            return await self._call_gemini(prompt, frames, max_output_tokens=max_output_tokens)
        elif self._provider == "openai":
            return await self._call_openai(prompt, frames)
        else:
            return "{\"items\": []}", {"tokens_in": 0, "tokens_out": 0, "model": "none"}

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

        items_data = data.get("items", [])
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
                await asyncio.sleep(batch_idx * 1.5)
            prompt = self._build_batch_prompt(
                batch_frames, transcript, start_s, end_s, duration_s
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
            # Multiple batches — LLM merge (with higher token limit for large inventories)
            has_transcript = transcript is not None and transcript.has_speech
            merge_prompt = self._build_merge_prompt(raw_responses, has_transcript=has_transcript)
            merge_text, merge_usage = await self._call_llm(
                merge_prompt, max_output_tokens=32000
            )
            total_usage["tokens_in"] += merge_usage.get("tokens_in", 0)
            total_usage["tokens_out"] += merge_usage.get("tokens_out", 0)
            raw_responses.append(f"--- MERGE ---\n{merge_text}")
            all_items = self._parse_inventory_json(merge_text)

            # Fallback: if merge JSON was truncated/invalid, combine batch results
            if not all_items:
                logger.warning("LLM merge returned no items — falling back to "
                               "programmatic batch combination")
                all_items = self._programmatic_merge(raw_responses[:-1], has_transcript)

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
