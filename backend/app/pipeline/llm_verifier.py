"""LLM Verification — GPT-4o confirms final inventory against key frames."""

import base64

import cv2
import numpy as np
import httpx
from loguru import logger
from openai import AsyncOpenAI

from app.config import get_settings


class LLMVerifier:
    """Final verification step: send key frames + draft inventory to GPT-4o."""

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

    async def verify(
        self,
        inventory: list[dict],
        key_frames: list[np.ndarray],
    ) -> tuple[list[dict], dict]:
        """
        Send key frames + draft inventory to GPT-4o for verification.
        Falls back to passing inventory through unchanged if no API key.
        """
        if not key_frames or not inventory:
            return inventory, {"tokens_in": 0, "tokens_out": 0, "model": self.model}

        # Fallback: skip verification when no API key
        if not self.api_key or not self.client:
            logger.warning("No OpenAI API key — skipping LLM verification (passthrough)")
            return inventory, {"tokens_in": 0, "tokens_out": 0, "model": "none"}

        # Build inventory text
        inv_text = "\n".join(
            f"- {item['name']}: {item['count']}× (room: {item.get('room_name', '?')})"
            for item in inventory
        )

        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    "You are an inventory surveyor for a moving company, verifying "
                    "an automatically generated list of items to be moved.\n\n"
                    "Below is the draft inventory detected by our AI system using "
                    "frame-by-frame object detection and CLIP-appearance tracking. "
                    "The tracker uses visual embeddings to identify unique objects "
                    "across ALL video frames — including frames NOT shown here.\n"
                    "You will see only a few key frames from the property walkthrough.\n\n"
                    f"DRAFT INVENTORY:\n{inv_text}\n\n"
                    "TASK:\n"
                    "Review each item against the key frames.\n"
                    "- REMOVE items you are CERTAIN are false positives "
                    "(e.g., a wall pattern detected as 'painting', or a shadow detected as an object).\n"
                    "- LOWER a count ONLY if you can see ALL instances of that item type "
                    "in a single wide shot AND the count is clearly wrong. "
                    "For example, if the draft says 3 vending machines but a wide shot "
                    "shows the entire wall with only 1, lower it.\n"
                    "- Do NOT lower counts for items that could exist outside the key frames. "
                    "The tracker processed MANY more frames than shown here. "
                    "Backpacks, bags, small items, and furniture may appear in frames you cannot see.\n"
                    "- Do NOT lower a count just because you see fewer in these key frames — "
                    "these are a SMALL SUBSET of all frames.\n"
                    "- Do NOT increase any count above what the draft says.\n"
                    "- Do NOT add new items not in the draft.\n"
                    "- When in doubt, KEEP the item and its count unchanged.\n\n"
                    "RESPONSE FORMAT (JSON array):\n"
                    '[{"name": "item", "count": N, "room_name": "room"}, ...]'
                ),
            }
        ]

        for frame in key_frames:
            b64 = self._frame_to_base64(frame)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "auto"},
            })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=1500,
            temperature=0,
        )

        text = response.choices[0].message.content or ""
        usage_data = {
            "tokens_in": response.usage.prompt_tokens if response.usage else 0,
            "tokens_out": response.usage.completion_tokens if response.usage else 0,
            "model": self.model,
        }
        logger.info(f"LLM Verification response:\n{text}")
        logger.info(f"Verification usage: {usage_data}")

        # Parse JSON response
        try:
            import json
            # Extract JSON from potential markdown code block
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            verified = json.loads(text.strip())
            if isinstance(verified, list):
                # Safety: never let verifier increase counts above draft
                draft_counts = {item["name"].lower(): item["count"] for item in inventory}
                for item in verified:
                    name_lower = item.get("name", "").lower()
                    draft_count = draft_counts.get(name_lower, 0)
                    if draft_count > 0 and item.get("count", 0) > draft_count:
                        logger.warning(
                            f"Verifier tried to increase {item['name']}: "
                            f"{draft_count}→{item['count']} — capping at {draft_count}"
                        )
                        item["count"] = draft_count
                    elif item.get("count", 0) < draft_count:
                        logger.info(
                            f"Verifier lowered {item['name']}: "
                            f"{draft_count}→{item['count']}"
                        )
                return verified, usage_data
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM verification response — using draft inventory")

        return inventory, usage_data
