# MIT License - Copyright (c) 2025 SAM3 Project Contributors
"""
Minimal task-aligned utilities: anchor helpers used by detection heads.
"""
from __future__ import annotations

import torch

from .torch_utils import TORCH_1_11


def make_anchors(feats, strides, grid_cell_offset: float = 0.5):
    """Generate anchor points and stride tensors from feature maps."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):
        stride = strides[i]
        h, w = (
            feats[i].shape[2:]
            if isinstance(feats, list)
            else (int(feats[i][0]), int(feats[i][1]))
        )
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = (
            torch.meshgrid(sy, sx, indexing="ij")
            if TORCH_1_11
            else torch.meshgrid(sy, sx)
        )
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device)
        )
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def dist2bbox(distance, anchor_points, xywh: bool = True, dim: int = -1):
    """Transform ltrb distance predictions to xyxy or xywh bounding boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    return torch.cat((x1y1, x2y2), dim)


def dist2rbox(pred_dist, pred_angle, anchor_points, dim: int = -1):
    """Decode predicted rotated bounding box coordinates from anchor points."""
    lt, rb = pred_dist.split(2, dim=dim)
    cos, sin = torch.cos(pred_angle), torch.sin(pred_angle)
    xf, yf = ((rb - lt) / 2).split(1, dim=dim)
    x = xf * cos - yf * sin
    y = xf * sin + yf * cos
    xy = torch.cat([x, y], dim=dim) + anchor_points
    return torch.cat([xy, lt + rb], dim=dim)
