"""LLM Verification via Google Gemini — confirms final inventory against key frames."""

import base64
import json

import cv2
import numpy as np
from loguru import logger

from app.config import get_settings


class GeminiVerifier:
    """Final verification step: send key frames + draft inventory to Gemini."""

    def __init__(self):
        settings = get_settings()
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL

    def _frame_to_base64(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    async def verify(
        self,
        inventory: list[dict],
        key_frames: list[np.ndarray],
    ) -> tuple[list[dict], dict]:
        """
        Send key frames + draft inventory to Gemini for verification.
        Falls back to passing inventory through unchanged if no API key.
        """
        if not key_frames or not inventory:
            return inventory, {"tokens_in": 0, "tokens_out": 0, "model": self.model}

        if not self.api_key:
            logger.warning("No Gemini API key — skipping verification (passthrough)")
            return inventory, {"tokens_in": 0, "tokens_out": 0, "model": "none"}

        import google.generativeai as genai
        from PIL import Image

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        # Build inventory text
        inv_text = "\n".join(
            f"- {item['name']}: {item['count']}× (room: {item.get('room_name', '?')})"
            for item in inventory
        )

        prompt_text = (
            "You are a strict verification surveyor for a moving company.\n\n"
            "An AI system generated the draft list of items to be moved. Your job is to "
            "CHECK EVERY ITEM against the key frames and AGGRESSIVELY remove "
            "anything that is NOT clearly visible in the property.\n\n"
            f"DRAFT INVENTORY:\n{inv_text}\n\n"
            "CRITICAL VERIFICATION RULES:\n"
            "1. For EACH item in the draft, look at the frames — can you clearly see it?\n"
            "2. If you cannot clearly see an item in any frame, REMOVE IT.\n"
            "3. Be very strict about identification — common AI mistakes include:\n"
            "   - Calling chair bases/wheels a 'TV stand'\n"
            "   - Confusing monitors with TVs\n"
            "   - Inventing items that should logically be in a room but aren't visible\n"
            "4. Correct counts — if you can clearly see more items than listed, increase the count.\n"
            "5. Only add missing items if they are OBVIOUS and clearly visible.\n"
            "6. It is better to under-report than to include false positives.\n\n"
            "RESPONSE FORMAT (JSON array, nothing else):\n"
            '[{"name": "item", "count": N, "room_name": "room"}, ...]'
        )

        # Build parts: text + images
        parts: list = [prompt_text]
        for frame in key_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            parts.append(pil_img)

        response = await model.generate_content_async(
            parts,
            generation_config=genai.GenerationConfig(
                max_output_tokens=1500,
                temperature=0.1,
            ),
        )

        text = response.text or ""
        usage_meta = getattr(response, "usage_metadata", None)
        usage_data = {
            "tokens_in": getattr(usage_meta, "prompt_token_count", 0) if usage_meta else 0,
            "tokens_out": getattr(usage_meta, "candidates_token_count", 0) if usage_meta else 0,
            "model": self.model,
        }
        logger.info(f"Gemini Verification response:\n{text}")
        logger.info(f"Gemini Verification usage: {usage_data}")

        # Parse JSON response
        try:
            clean = text.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            verified = json.loads(clean.strip())
            if isinstance(verified, list):
                return verified, usage_data
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse Gemini verification response — using draft inventory")

        return inventory, usage_data
