"""Evidence frame renderer — Soft radial vignette for item identification.

Takes a full video frame and a centroid point (from Gemini) and produces
an annotated evidence image with a soft circular glow centered on the item.
The bright area fades smoothly into a darkened surround — no hard edges,
no precise markers that could look wrong if the centroid is slightly off.

If no centroid is provided, returns the original frame unchanged.

This module is fully self-contained. Removing it only requires deleting the
single call site in report_generator.py (the _assign_evidence_frames function).
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from loguru import logger


def render_evidence(
    frame: np.ndarray,
    centroid_1000: Optional[list[int]] = None,
    item_name: str = "",
    glow_fraction: float = 0.35,
    darken_strength: float = 0.55,
    label_font_scale: float = 0.6,
    label_thickness: int = 2,
) -> np.ndarray:
    """Render a soft radial vignette on an evidence frame centered on the item.

    The centroid area stays bright while the rest of the frame darkens
    gradually via a smooth Gaussian-blurred circular mask.

    Args:
        frame: BGR numpy array (original video frame).
        centroid_1000: Gemini-format centroid [y, x] on 0-1000 scale.
                       If None, returns the frame unchanged.
        item_name: Item name for a subtle label (optional).
        glow_fraction: Radius of the bright region as a fraction of
                       frame diagonal (0.35 = 35% of diagonal).
        darken_strength: How dark the edges get (0=invisible, 1=fully black).
        label_font_scale: Font scale for the item name label.
        label_thickness: Thickness for the label text.

    Returns:
        Annotated BGR frame (new array — original is not mutated).
    """
    if centroid_1000 is None:
        return frame

    # Validate centroid
    if not isinstance(centroid_1000, (list, tuple)) or len(centroid_1000) != 2:
        logger.debug(f"Invalid centroid format for '{item_name}': {centroid_1000}")
        return frame

    try:
        cy_1000, cx_1000 = [int(v) for v in centroid_1000]
    except (TypeError, ValueError):
        logger.debug(f"Non-numeric centroid for '{item_name}': {centroid_1000}")
        return frame

    if not (0 <= cy_1000 <= 1000 and 0 <= cx_1000 <= 1000):
        logger.debug(f"Out-of-range centroid for '{item_name}': {centroid_1000}")
        return frame

    h, w = frame.shape[:2]

    # Convert from 0-1000 scale to pixel coordinates
    cx = int(cx_1000 * w / 1000)
    cy = int(cy_1000 * h / 1000)

    # Clamp to frame bounds
    cx = max(0, min(cx, w - 1))
    cy = max(0, min(cy, h - 1))

    # ── 1. Build soft radial mask ──
    # Create a distance map from the centroid
    diag = np.sqrt(h * h + w * w)
    radius = glow_fraction * diag

    # Coordinate grids
    ys = np.arange(h, dtype=np.float32)
    xs = np.arange(w, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")

    # Distance from centroid
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Smooth falloff: 0 at center (no darkening), 1 at edges (full darkening)
    # Using a smooth sigmoid-like curve for natural falloff
    mask = np.clip((dist - radius * 0.5) / (radius * 0.8), 0.0, 1.0)

    # Apply a slight Gaussian smooth for extra softness
    kernel_size = max(31, int(diag * 0.04) | 1)  # ensure odd
    mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    # Scale by darken strength
    mask = (mask * darken_strength).astype(np.float32)

    # ── 2. Apply vignette ──
    # Darken: result = frame * (1 - mask)
    mask_3ch = np.stack([mask, mask, mask], axis=-1)
    result = (frame.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

    # ── 3. Item name label (bottom of bright region) ──
    if item_name:
        label = item_name.capitalize()
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, label_font_scale, label_thickness
        )

        # Position label below the centroid, within the bright zone
        label_x = max(4, cx - text_w // 2)
        label_y = min(h - 8, cy + int(radius * 0.4) + text_h)

        # Keep label within frame bounds
        label_x = min(label_x, w - text_w - 8)
        label_x = max(4, label_x)

        # Semi-transparent background for readability
        pad = 5
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (label_x - pad, label_y - text_h - pad),
            (label_x + text_w + pad, label_y + baseline + pad),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)

        cv2.putText(
            result, label, (label_x, label_y),
            font, label_font_scale, (255, 255, 255), label_thickness, cv2.LINE_AA,
        )

    return result
