"""SAM3 detector wrapper — drop-in replacement for GDINODetector.

Exposes the same `detect_frame()` interface:
    detect_frame(frame, prompts, threshold) -> list[dict]

Each detection dict has:
    {"label": str, "bbox": [x1,y1,x2,y2], "confidence": float,
     "mask": np.ndarray | None}

The mask field is added for SAM3 (GDINO doesn't produce masks).
Downstream code that doesn't use masks simply ignores it.
"""

import os
import sys
import time

import numpy as np
import torch
from loguru import logger


class SAM3Detector:
    """
    SAM3 semantic (text-prompted) segmentation detector.
    Processes frames with prompt batching, returns bbox + mask per detection.

    Requires: sam3.pt weights (~3.45 GB), SAM3 framework in sam3_framework/
    Falls back to stub mode if framework/weights unavailable.
    """

    PROMPT_BATCH_SIZE = 20

    def __init__(self, weights_path: str = "", batch_size: int = 20, imgsz: int = 512):
        self.weights_path = weights_path
        self.PROMPT_BATCH_SIZE = batch_size
        self.imgsz = imgsz
        self._predictor = None
        self._device: str = ""
        self._stub_mode = False

    def _load_model(self):
        """Lazy-load SAM3 predictor."""
        if self._predictor is not None or self._stub_mode:
            return

        try:
            # Add sam3_framework to sys.path so its internal absolute imports work
            framework_dir = os.path.join(
                os.path.dirname(__file__), "sam3_framework"
            )
            if framework_dir not in sys.path:
                sys.path.insert(0, framework_dir)

            from predict import SAM3SemanticPredictor

            # Determine device
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

            use_half = torch.cuda.is_available()

            # Resolve weights path
            weights = self.weights_path
            if not weights or not os.path.isfile(weights):
                weights = "sam3.pt"
            if not os.path.isfile(weights):
                raise FileNotFoundError(
                    f"SAM3 weights not found at: {self.weights_path} or sam3.pt. "
                    f"Expected ~3.45 GB file."
                )

            t0 = time.time()
            self._predictor = SAM3SemanticPredictor(overrides=dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=weights,
                half=use_half,
                save=False,
                imgsz=self.imgsz,
                device=self._device,
            ))
            logger.info(
                f"SAM3 loaded on {self._device} in {time.time() - t0:.1f}s "
                f"(weights: {weights}, imgsz: {self.imgsz})"
            )

        except ImportError as e:
            logger.warning(f"SAM3 framework not available — using stub mode: {e}")
            self._stub_mode = True
        except FileNotFoundError as e:
            logger.warning(f"SAM3 weights missing — using stub mode: {e}")
            self._stub_mode = True
        except Exception as e:
            logger.warning(f"SAM3 load failed — using stub mode: {e}")
            self._stub_mode = True

    def detect_frame(
        self, frame: np.ndarray, prompts: list[str], threshold: float = 0.25
    ) -> list[dict]:
        """
        Detect objects in a single frame using given text prompts.
        Returns list of {"label": str, "bbox": [x1,y1,x2,y2], "confidence": float, "mask": ndarray|None}.

        Prompts are batched internally (PROMPT_BATCH_SIZE).
        """
        self._load_model()

        if self._stub_mode or self._predictor is None:
            return []

        predictor = self._predictor
        h, w = frame.shape[:2]

        # SAM3 expects BGR, which is what OpenCV read gives us
        predictor.set_image(frame)

        all_dets: list[dict] = []
        bs = self.PROMPT_BATCH_SIZE

        for batch_start in range(0, len(prompts), bs):
            batch_prompts = prompts[batch_start : batch_start + bs]

            with torch.no_grad():
                results = predictor(text=batch_prompts)

            if not results:
                continue

            for result in results:
                boxes_data = result.boxes
                masks_data = result.masks
                names = result.names

                if boxes_data is None or len(boxes_data) == 0:
                    continue

                boxes_np = boxes_data.data.cpu().numpy()
                masks_np = (
                    masks_data.data.cpu().numpy()
                    if masks_data is not None
                    else None
                )

                for j, row in enumerate(boxes_np):
                    x1, y1, x2, y2, conf, cls_raw = row[:6]
                    conf = float(conf)

                    if conf < threshold:
                        continue

                    cls_idx = int(cls_raw)
                    class_name = (
                        names[cls_idx] if cls_idx < len(names) else f"cls_{cls_idx}"
                    )

                    # Match back to original prompt
                    label = self._match_prompt(class_name, batch_prompts)

                    # Filter tiny boxes
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < 100:
                        continue

                    # Extract mask if available
                    mask = None
                    if masks_np is not None and j < masks_np.shape[0]:
                        mask_bin = masks_np[j].astype(bool)
                        if mask_bin.sum() >= 50:  # skip tiny mask fragments
                            mask = mask_bin

                    all_dets.append({
                        "label": label,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": conf,
                        "mask": mask,
                    })

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_dets

    @staticmethod
    def _match_prompt(phrase: str, prompts: list[str]) -> str:
        """Match a SAM3 output class name to the closest original prompt."""
        phrase_lower = phrase.lower().strip()
        for p in prompts:
            if p.lower().strip() == phrase_lower:
                return p
        for p in prompts:
            if phrase_lower in p.lower() or p.lower() in phrase_lower:
                return p
        return phrase
