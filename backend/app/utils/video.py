"""Video utilities — frame extraction using OpenCV."""

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


def extract_frames(video_path: str, stride: int = 30, max_width: int = 480) -> list[np.ndarray]:
    """Extract frames from video at given stride, resized to max_width."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video: {video_path} — {total} frames, {fps:.1f} fps")

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (max_width, int(h * scale)))
            frames.append(frame)
        idx += 1

    cap.release()
    logger.info(f"Extracted {len(frames)} frames (stride={stride})")
    return frames


def get_video_info(video_path: str) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_s"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info
