"""LLM Checklist Priming — send scene frames + 150-item checklist to GPT-4o."""

import base64
import re

import cv2
import numpy as np
import httpx
from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings
from app.pipeline.checklist import get_checklist_text


class LLMPrimer:
    """Use GPT-4o to identify which items from the checklist are present in scene frames."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.OPENAI_API_KEY
        self.model = settings.OPENAI_MODEL
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
        Returns (confirmed_prompts, new_items, usage, count_hints).
        Falls back to full checklist if no OpenAI API key is configured.
        """
        if not scenes:
            return [], [], {"tokens_in": 0, "tokens_out": 0, "model": self.model}, {}

        # Fallback: use common items as prompts when no API key
        if not self.api_key or not self.client:
            fallback = [
                "sofa", "armchair", "coffee table", "dining table", "dining chair",
                "TV", "TV stand", "bookshelf", "bed", "nightstand", "dresser",
                "wardrobe", "desk", "office chair", "floor lamp", "table lamp",
                "refrigerator", "oven", "microwave", "dishwasher", "washing machine",
                "area rug", "mirror", "curtain", "plant pot", "fan",
                "stool", "bench", "cabinet", "shoe rack",
            ]
            logger.warning(f"No OpenAI API key — using {len(fallback)} common items as GDINO prompts")
            return list(fallback), [], {"tokens_in": 0, "tokens_out": 0, "model": "none"}, {}

        checklist_text = get_checklist_text()

        # Build multimodal messages
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "You are an inventory surveyor for a moving company. "
                    "Your goal is to create a complete and conservative moving inventory from walkthrough video frames.\n\n"
                    "You will be shown multiple frames from the same property walkthrough. "
                    "The camera may pan across the room, so the same object can appear in more than one frame.\n\n"
                    "TASK:\n"
                    "Identify all furniture, appliances, and household items visible in these frames that would need to be packed and moved.\n\n"
                    "COUNTING RULES (VERY IMPORTANT):\n"
                    "1. First count all CLEARLY VISIBLE item instances in the frames.\n"
                    "2. If multiple items of the same type are visible at the same time or in clearly different positions, count ALL of them.\n"
                    "3. Only merge sightings across frames when it is VERY LIKELY they are the exact same physical object.\n"
                    "4. Do NOT reduce a count unless duplicate identity across frames is strong and obvious.\n"
                    "5. If 6 separate chairs are clearly visible across the frames, report 6 chairs — do not lower the count unless some are obviously repeat views of the same chair.\n"
                    "6. When uncertain between two nearby counts, prefer the higher count if the extra item is clearly visible.\n"
                    "7. For checklist items, use exact counts when visibility is clear. Use approximate counts only if visibility is partial or obstructed.\n\n"
                    f"CHECKLIST:\n"
                    f"Report ONLY checklist items that you can confirm are present in at least one frame.\n\n"
                    f"{checklist_text}\n\n"
                    "NEW ITEMS:\n"
                    "If you see any movable household item that is not on the checklist, include it with prefix [NEW].\n"
                    "For [NEW] items, include count and first frame where it is clearly seen.\n\n"
                    "OUTPUT RULES:\n"
                    "- Report only items that are visually confirmed.\n"
                    "- Do not mention items that are speculative.\n"
                    "- Prefer exact integer counts for confirmed items.\n"
                    "- Use (~N) only when the count is genuinely uncertain due to occlusion or partial visibility.\n\n"
                    "RESPONSE FORMAT (nothing else):\n"
                    "CONFIRMED: item1 (N), item2 (N), item3 (~N)\n"
                    "NEW: item_name (N, frame F), item_name (~N, frame F)"
                ),
            }
        ]

        for i, scene in enumerate(scenes):
            b64 = self._frame_to_base64(scene["frame"])
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1000,
            temperature=0.1,
        )

        text = response.choices[0].message.content or ""
        usage_data = {
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "model": self.model,
        }
        logger.info(f"LLM Priming response:\n{text}")
        logger.info(f"Priming usage: {usage_data}")

        confirmed, new_items, count_hints = self._parse_response(text)
        logger.info(f"Priming: {len(confirmed)} confirmed, {len(new_items)} new, hints: {count_hints}")

        # Build GDINO prompts
        prompts = confirmed + [item["name"] for item in new_items]
        # Add new_item counts to hints
        for item in new_items:
            if item.get("count", 0) > 0:
                count_hints[item["name"]] = item["count"]
        return prompts, new_items, usage_data, count_hints

    @staticmethod
    def _split_items(text: str) -> list[str]:
        """Split comma-separated items, respecting parenthesised groups."""
        parts: list[str] = []
        depth = 0
        current: list[str] = []
        for ch in text:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth = max(depth - 1, 0)
                current.append(ch)
            elif ch == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    def _parse_response(self, text: str) -> tuple[list[str], list[dict], dict[str, int]]:
        """Parse CONFIRMED and NEW lines from LLM response, extracting count hints."""
        confirmed: list[str] = []
        new_items: list[dict] = []
        count_hints: dict[str, int] = {}

        for line in text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("CONFIRMED:"):
                items_str = line.split(":", 1)[1].strip()
                for item in self._split_items(items_str):
                    if not item:
                        continue
                    # Extract count hint: "office chair (~6)" → name="office chair", count=6
                    count_match = re.search(r"\(~?(\d+)\)", item)
                    count = int(count_match.group(1)) if count_match else 1
                    # Remove the (~N) part and any leading quantity
                    cleaned = re.sub(r"\s*\(~?\d+\)", "", item).strip()
                    cleaned = re.sub(r"^\d+\s+", "", cleaned).strip()
                    # Singularize simple cases
                    if cleaned.endswith("s") and not cleaned.endswith("ss"):
                        singular = cleaned[:-1]
                    else:
                        singular = cleaned
                    name = singular if singular else cleaned
                    confirmed.append(name)
                    count_hints[name] = count

            elif line.upper().startswith("NEW:"):
                items_str = line.split(":", 1)[1].strip()
                for item in self._split_items(items_str):
                    if not item:
                        continue
                    # Parse "item_name (~N, frame F)" or "item_name (frame N)"
                    match = re.match(r"(.+?)\s*\(~?(\d+),?\s*frame\s*(\d+)\)", item, re.IGNORECASE)
                    if match:
                        new_items.append({"name": match.group(1).strip(), "count": int(match.group(2)), "frame": int(match.group(3))})
                    else:
                        match2 = re.match(r"(.+?)\s*\(frame\s*(\d+)\)", item, re.IGNORECASE)
                        if match2:
                            new_items.append({"name": match2.group(1).strip(), "count": 1, "frame": int(match2.group(2))})
                        else:
                            new_items.append({"name": item.strip(), "count": 1, "frame": 0})

        return confirmed, new_items, count_hints
