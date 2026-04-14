"""LLM Checklist Priming - send scene frames + checklist to GPT-4o (JSON per-frame)."""

import base64
import json

import cv2
import numpy as np
import httpx
from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.pipeline.checklist import get_checklist_text

_JSON_SCHEMA_EXAMPLE = r"""{
  "frames": [
    {
      "frame": 1,
      "items": [
        {"name": "dining chair", "count": 3, "size": "M", "fragility": "LOW"},
        {"name": "coffee machine", "count": 2, "size": "M", "fragility": "MEDIUM"}
      ]
    },
    {
      "frame": 2,
      "items": [
        {"name": "dining chair", "count": 2, "size": "M", "fragility": "LOW"},
        {"name": "vending machine", "count": 1, "size": "L", "fragility": "LOW"},
        {"name": "custom widget", "count": 1, "size": "S", "fragility": "HIGH", "new": true}
      ]
    }
  ]
}"""


class LLMPrimer:
    """Use GPT-4o to identify which items from the checklist are present in scene frames."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
        self.raw_response: str = ""
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            http_client=httpx.AsyncClient(verify=False),
        ) if self.api_key else None

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    async def prime(self, scenes: list[dict]) -> tuple[list[str], list[dict], dict, dict[str, int]]:
        """
        Send scene frames + checklist to GPT-4o.
        Returns (prompts, new_items, usage, count_hints).
        """
        if not scenes:
            return [], [], {"tokens_in": 0, "tokens_out": 0, "model": self.model}, {}

        if not self.api_key or not self.client:
            fallback = [
                "sofa", "armchair", "coffee table", "dining table", "dining chair",
                "TV", "TV stand", "bookshelf", "bed", "nightstand", "dresser",
                "wardrobe", "desk", "office chair", "floor lamp", "table lamp",
                "refrigerator", "oven", "microwave", "dishwasher", "washing machine",
                "area rug", "mirror", "curtain", "plant pot", "fan",
                "stool", "bench", "cabinet", "shoe rack",
            ]
            logger.warning(f"No OpenAI API key - using {len(fallback)} common items as GDINO prompts")
            return list(fallback), [], {"tokens_in": 0, "tokens_out": 0, "model": "none"}, {}

        checklist_text = get_checklist_text()

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "You are an expert moving inventory surveyor. Your job is to identify "
                    "EVERY physical object that movers would need to pack, carry, or transport "
                    "from this property.\n\n"
                    "TASK:\n"
                    "For EACH frame, list every movable item you see and how many are visible "
                    "IN THAT FRAME.\n"
                    "Do NOT merge across frames. Report only what is visible in the current frame.\n\n"
                    "---\n\n"
                    "HOW TO SCAN (STRICT PROCESS):\n\n"
                    "Scan each frame in this order:\n"
                    "1. Large furniture (beds, sofas, tables, cabinets)\n"
                    "2. Medium movable items (chairs, appliances, suitcases)\n"
                    "3. ALL surfaces (tables, counters, shelves, beds)\n"
                    "4. Floor and corners\n"
                    "5. Walls (anything mounted, hanging, or decorative)\n"
                    "6. Overhead (fans, hanging items if movable)\n\n"
                    "IMPORTANT:\n"
                    "Small items on surfaces are HIGH PRIORITY and must NOT be skipped.\n\n"
                    "---\n\n"
                    "THINK LIKE A MOVER:\n\n"
                    "If it needs to be packed, wrapped, carried, or loaded onto a truck — list it.\n\n"
                    "ALWAYS INCLUDE:\n"
                    "- Bags, backpacks, handbags, helmets\n"
                    "- Electronics (monitors, laptops, chargers, remotes)\n"
                    "- Items on tables/counters/shelves\n"
                    "- Kitchen items and loose utensils\n"
                    "- Decorative items (frames, vases, lamps, rugs, cushions)\n\n"
                    "DO NOT INCLUDE:\n"
                    "- Built-in or fixed infrastructure (doors, windows, wall sockets, "
                    "built-in cabinets, walls, ceilings, floors, columns)\n\n"
                    "---\n\n"
                    "SMALL ITEM RULE:\n\n"
                    "If multiple small similar items exist:\n"
                    '- Group them (e.g., "books", "toys", "clothes")\n'
                    "- BUT do NOT ignore them\n\n"
                    "If a small object is clearly visible (e.g., helmet, bag, speaker), "
                    "list it separately.\n\n"
                    "---\n\n"
                    "UNCERTAINTY RULE:\n\n"
                    "If you are not 100% sure what an object is:\n"
                    "- Make your BEST guess\n"
                    '- Use a generic label if needed (e.g., "bag", "container", "device")\n'
                    "- Do NOT skip visible objects\n\n"
                    "---\n\n"
                    "ITEM NAMING:\n\n"
                    "Use a name from the CHECKLIST if possible.\n"
                    "When an item could match multiple checklist names, choose the most "
                    "specific match.\n"
                    "If no checklist name fits:\n"
                    "- Use a clear, specific lowercase name\n"
                    '- Set "new": true\n\n'
                    "Do NOT rename checklist items.\n\n"
                    "---\n\n"
                    "COUNTING (PER FRAME ONLY):\n\n"
                    "Count every visible instance in the current frame.\n"
                    "Include partially visible or occluded objects.\n"
                    "Do NOT reason across frames.\n"
                    "When in doubt, COUNT IT.\n\n"
                    "---\n\n"
                    "SIZE GUIDELINES:\n"
                    "S = fits in a box (books, small electronics, decor)\n"
                    "M = can be carried by one person (chair, suitcase, microwave)\n"
                    "L = requires two people or special handling (sofa, bed, large cabinet)\n\n"
                    "FRAGILITY:\n"
                    "HIGH = glass, screens, delicate items\n"
                    "MEDIUM = electronics, appliances\n"
                    "LOW = solid furniture, bags, non-breakables\n\n"
                    "---\n\n"
                    f"CHECKLIST:\n{checklist_text}\n\n"
                    "Respond with ONLY valid JSON matching this schema:\n"
                    f"{_JSON_SCHEMA_EXAMPLE}\n\n"
                    "Rules:\n"
                    "- One entry per frame, listing all items visible in that frame.\n"
                    '- "name" must be lowercase.\n'
                    '- "new": true only for items not in the checklist. Omit the field otherwise.\n'
                    '- "size": one of "S", "M", "L".\n'
                    '- "fragility": one of "HIGH", "MEDIUM", "LOW".'
                ),
            }
        ]

        for i, scene in enumerate(scenes):
            content.append({"type": "text", "text": f"Frame {i + 1}:"})
            b64 = self._frame_to_base64(scene["frame"])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=4000,
            temperature=0,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or ""
        self.raw_response = text
        usage_data = {
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "model": self.model,
        }
        logger.info(f"LLM Priming response:\n{text}")
        logger.info(f"Priming usage: {usage_data}")

        prompts, new_items, count_hints = self._parse_json_response(text)
        logger.info(f"Priming: {len(prompts)} prompts, {len(new_items)} new, hints: {count_hints}")
        if self._item_meta:
            logger.info(f"Item metadata: {self._item_meta}")

        return prompts, new_items, usage_data, count_hints

    def _parse_json_response(self, text: str) -> tuple[list[str], list[dict], dict[str, int]]:
        """Parse JSON per-frame response. Compute max-per-frame as count_hints."""
        self._item_meta: dict[str, dict] = {}
        self._per_frame_data: list[dict] = []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from LLM response: {text[:200]}")
            return [], [], {}

        frames = data.get("frames", [])
        if not frames:
            logger.warning("LLM returned empty frames array")
            return [], [], {}

        self._per_frame_data = frames

        max_counts: dict[str, int] = {}
        new_item_names: set[str] = set()

        for frame_data in frames:
            for item in frame_data.get("items", []):
                name = item.get("name", "").strip().lower()
                if not name:
                    continue
                count = item.get("count", 1)
                if isinstance(count, str):
                    count = int("".join(c for c in count if c.isdigit()) or "1")

                if name not in max_counts or count > max_counts[name]:
                    max_counts[name] = count

                if name not in self._item_meta:
                    size = item.get("size", "M").upper()
                    if size not in ("S", "M", "L"):
                        size = "M"
                    fragility = item.get("fragility", "LOW").upper()
                    if fragility not in ("HIGH", "MEDIUM", "LOW"):
                        fragility = "LOW"
                    self._item_meta[name] = {"size": size, "fragility": fragility}

                if item.get("new"):
                    new_item_names.add(name)

        prompts = list(max_counts.keys())
        count_hints = dict(max_counts)
        new_items = [
            {"name": name, "count": max_counts[name], "frame": 0,
             **self._item_meta.get(name, {"size": "M", "fragility": "LOW"})}
            for name in new_item_names
        ]

        return prompts, new_items, count_hints
