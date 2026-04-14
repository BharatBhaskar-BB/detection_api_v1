"""LLM Checklist Priming via Google Gemini (JSON per-frame)."""

import base64
import json

import cv2
import numpy as np
from loguru import logger

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


class GeminiPrimer:
    """Use Gemini to identify which items from the checklist are present in scene frames."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    async def prime(self, scenes: list[dict]) -> tuple[list[str], list[dict], dict, dict[str, int]]:
        """
        Send scene frames + checklist to Gemini.
        Returns (prompts, new_items, usage, count_hints).
        """
        if not scenes:
            return [], [], {"tokens_in": 0, "tokens_out": 0, "model": self.model}, {}

        if not self.api_key:
            logger.warning("No Gemini API key - falling back to common items")
            fallback = [
                "sofa", "armchair", "coffee table", "dining table", "dining chair",
                "TV", "TV stand", "bookshelf", "bed", "nightstand", "dresser",
                "wardrobe", "desk", "office chair", "floor lamp", "table lamp",
                "refrigerator", "oven", "microwave", "dishwasher", "washing machine",
                "area rug", "mirror", "curtain", "plant pot", "fan",
                "stool", "bench", "cabinet", "shoe rack",
            ]
            return list(fallback), [], {"tokens_in": 0, "tokens_out": 0, "model": "none"}, {}

        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        checklist_text = get_checklist_text()

        prompt_text = (
            "You are an expert moving inventory surveyor.\n\n"
            "TASK: For EACH frame, list every movable item you see and how many are visible IN THAT FRAME.\n"
            "Do NOT merge across frames. Just report what you see in each individual frame.\n\n"
            "ITEM NAMING:\n"
            "- ALWAYS use a name from the checklist below if the item matches or closely resembles one.\n"
            "  Example: a padded footstool = 'ottoman', a screen on a stand = 'TV' or 'monitor'.\n"
            "- Only set \"new\": true for items that genuinely have NO match in the checklist.\n"
            "- Do NOT rename checklist items.\n\n"
            "COUNTING (per frame only):\n"
            "- Count every distinct item instance visible in the current frame.\n"
            "- If you see 2 coffee machines side by side in one frame, count = 2.\n"
            "- If you see 4 chairs around a table, count = 4.\n"
            "- Do NOT think about other frames. Just count what is visible NOW.\n\n"
            "SIZE categories: S (small, fits in a box), M (medium, needs 1 person), L (large, needs 2+ people)\n"
            "FRAGILITY: HIGH (glass/screens/delicate), MEDIUM (electronics/appliances), LOW (solid furniture/bags)\n\n"
            f"CHECKLIST:\n{checklist_text}\n\n"
            "Respond with ONLY valid JSON matching this schema:\n"
            f"{_JSON_SCHEMA_EXAMPLE}\n\n"
            "Rules:\n"
            "- One entry per frame, listing all items visible in that frame.\n"
            "- \"name\" must be lowercase.\n"
            "- \"new\": true only for items not in the checklist. Omit the field otherwise.\n"
            "- \"size\": one of \"S\", \"M\", \"L\".\n"
            "- \"fragility\": one of \"HIGH\", \"MEDIUM\", \"LOW\"."
        )

        parts: list = [prompt_text]
        for i, scene in enumerate(scenes):
            parts.append(f"Frame {i + 1}:")
            frame_bgr = scene["frame"]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            parts.append(pil_img)

        response = await model.generate_content_async(
            parts,
            generation_config=genai.GenerationConfig(
                max_output_tokens=4000,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        text = response.text or ""
        usage_meta = getattr(response, "usage_metadata", None)
        usage_data = {
            "tokens_in": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            "tokens_out": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
            "model": self.model,
        }
        logger.info(f"Gemini Priming response:\n{text}")
        logger.info(f"Gemini Priming usage: {usage_data}")

        prompts, new_items, count_hints = self._parse_json_response(text)
        logger.info(f"Gemini Priming: {len(prompts)} prompts, {len(new_items)} new, hints: {count_hints}")
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
            logger.error(f"Failed to parse JSON from Gemini response: {text[:200]}")
            return [], [], {}

        frames = data.get("frames", [])
        if not frames:
            logger.warning("Gemini returned empty frames array")
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
