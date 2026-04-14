"""Grounding-DINO detection — text-conditioned object detection with prompt batching."""

import time

import numpy as np
import torch
from loguru import logger


# Default config/weights filenames
_GDINO_CONFIG = "GroundingDINO_SwinT_OGC.py"
_GDINO_WEIGHTS = "groundingdino_swint_ogc.pth"


class GDINODetector:
    """
    Grounding-DINO object detector.
    Processes frames in batches of prompts for text-conditioned detection.

    Requires: groundingdino-py==0.4.0, SwinT model weights (~593MB)
    Falls back to stub mode if package/weights are unavailable.
    """

    PROMPT_BATCH_SIZE = 20

    def __init__(self, weights_path: str = "", batch_size: int = 20):
        self.weights_path = weights_path
        self.PROMPT_BATCH_SIZE = batch_size
        self._model = None
        self._device: str = ""
        self._stub_mode = False

    def _load_model(self):
        """Lazy-load GDINO model."""
        if self._model is not None or self._stub_mode:
            return

        try:
            import os
            import groundingdino
            from groundingdino.util.inference import load_model

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"  # GDINO doesn't fully support MPS

            # Resolve config path from the installed package
            pkg_dir = os.path.dirname(groundingdino.__file__)
            config_path = os.path.join(pkg_dir, "config", _GDINO_CONFIG)

            # Resolve weights path
            weights = self.weights_path
            if not weights or not os.path.isfile(weights):
                weights = _GDINO_WEIGHTS
            if not os.path.isfile(weights):
                raise FileNotFoundError(
                    f"GDINO weights not found at: {self.weights_path} or {_GDINO_WEIGHTS}. "
                    f"Download from: https://github.com/IDEA-Research/GroundingDINO/"
                    f"releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
                )

            t0 = time.time()
            self._model = load_model(config_path, weights, device=self._device)
            logger.info(
                f"GDINO loaded on {self._device} in {time.time() - t0:.1f}s (weights: {weights})"
            )
        except ImportError:
            logger.warning("groundingdino-py not installed — using stub mode")
            self._stub_mode = True
        except FileNotFoundError as e:
            logger.warning(f"GDINO weights missing — using stub mode: {e}")
            self._stub_mode = True
        except Exception as e:
            logger.warning(f"GDINO load failed — using stub mode: {e}")
            self._stub_mode = True

    def _build_caption(self, prompts: list[str]) -> str:
        """Convert prompt list to GDINO caption format: 'chair . table . sofa'."""
        return " . ".join(p.strip().lower() for p in prompts)

    @staticmethod
    def _match_prompt(phrase: str, prompts: list[str]) -> str:
        """Match a GDINO output phrase to the closest original prompt."""
        phrase_lower = phrase.lower().strip()
        for p in prompts:
            if p.lower().strip() == phrase_lower:
                return p
        for p in prompts:
            if phrase_lower in p.lower() or p.lower() in phrase_lower:
                return p
        return phrase

    def detect_frame(
        self, frame: np.ndarray, prompts: list[str], threshold: float = 0.25
    ) -> list[dict]:
        """
        Detect objects in a single frame using given prompts.
        Returns list of {"label": str, "bbox": [x1,y1,x2,y2], "confidence": float}.

        Prompts are batched internally (PROMPT_BATCH_SIZE).
        """
        self._load_model()

        if self._stub_mode or self._model is None:
            return []

        import groundingdino.datasets.transforms as T
        from PIL import Image
        import cv2

        # Convert BGR→RGB→PIL and pre-transform (done once, reused across batches)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_transformed, _ = transform(pil_image, None)

        h, w = frame.shape[:2]

        # Batch prompts for reliable phrase matching
        all_dets: list[dict] = []
        bs = self.PROMPT_BATCH_SIZE
        for batch_start in range(0, len(prompts), bs):
            batch_prompts = prompts[batch_start:batch_start + bs]
            dets = self._detect_batch(image_transformed, batch_prompts, w, h, threshold)
            all_dets.extend(dets)

        return all_dets

    def _detect_batch(
        self,
        image_tensor: torch.Tensor,
        batch_prompts: list[str],
        img_w: int,
        img_h: int,
        threshold: float,
    ) -> list[dict]:
        """Run one GDINO predict call for a batch of prompts."""
        from groundingdino.util.inference import predict

        caption = self._build_caption(batch_prompts)

        with torch.no_grad():
            boxes, logits, phrases = predict(
                model=self._model,
                image=image_tensor,
                caption=caption,
                box_threshold=threshold,
                text_threshold=max(0.15, threshold - 0.10),
                device=self._device,
            )

        detections: list[dict] = []
        if boxes is None or len(boxes) == 0:
            return detections

        boxes_np = boxes.cpu().numpy()
        logits_np = logits.cpu().numpy()

        for i in range(len(boxes_np)):
            cx, cy, bw, bh = boxes_np[i]
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h

            conf = float(logits_np[i])
            if conf < threshold:
                continue

            # Skip tiny detections
            box_area = (x2 - x1) * (y2 - y1)
            if box_area < 100:
                continue

            phrase = phrases[i].strip().lower()
            label = self._match_prompt(phrase, batch_prompts)

            detections.append({
                "label": label,
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(x2), 1), round(float(y2), 1)],
                "confidence": round(conf, 4),
            })

        return detections

    def detect_batch(
        self, frames: list[np.ndarray], prompts: list[str], threshold: float = 0.25
    ) -> list[list[dict]]:
        """Detect objects across multiple frames."""
        return [self.detect_frame(frame, prompts, threshold) for frame in frames]
