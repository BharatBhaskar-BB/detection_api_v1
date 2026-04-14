# MIT License - Copyright (c) 2025 SAM3 Project Contributors
"""
Minimal data utilities: format constants and polygon helpers.
"""
from __future__ import annotations

import numpy as np

# Supported image and video file extensions
IMG_FORMATS = frozenset({
    "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm", "heic",
})
VID_FORMATS = frozenset({
    "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm",
})
FORMATS_HELP_MSG = (
    f"Supported image formats: {', '.join(sorted(IMG_FORMATS))}. "
    f"Supported video formats: {', '.join(sorted(VID_FORMATS))}."
)


def merge_multi_segment(segments: list[np.ndarray]) -> np.ndarray:
    """Merge multiple polygon segments into one contiguous array.

    Connects each segment to the next via the closest endpoints, producing a
    single polygon suitable for mask generation.

    Args:
        segments: List of (N, 2) arrays, each a polygon's xy coordinates.

    Returns:
        (np.ndarray): Single merged (M, 2) polygon array.
    """
    if not segments:
        return np.zeros((0, 2), dtype=np.float32)
    if len(segments) == 1:
        return segments[0]

    def _nearest_point_idx(seg_a: np.ndarray, seg_b: np.ndarray):
        """Return indices of the closest pair of endpoints between two segments."""
        dists = np.linalg.norm(seg_a[[0, -1], None] - seg_b[None, [0, -1]], axis=-1)  # (2, 2)
        ai, bi = np.unravel_index(dists.argmin(), dists.shape)
        return ai, bi

    merged = segments[0]
    for seg in segments[1:]:
        ai, bi = _nearest_point_idx(merged, seg)
        # Rotate merged so the join endpoint is last
        if ai == 0:
            merged = merged[::-1]
        # Rotate seg so the join endpoint is first
        if bi == 1:
            seg = seg[::-1]
        merged = np.concatenate([merged, seg], axis=0)

    return merged
