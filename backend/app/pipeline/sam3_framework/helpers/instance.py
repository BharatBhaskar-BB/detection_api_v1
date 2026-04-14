# MIT License - Copyright (c) 2025 SAM3 Project Contributors
"""Minimal instance utilities."""
from __future__ import annotations
from itertools import repeat
import collections.abc


def to_2tuple(x):
    """Convert a scalar or iterable to a 2-element tuple."""
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, 2))
