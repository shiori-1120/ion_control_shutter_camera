from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RoiStats:
    mean: float
    std: float
    n: int
    meta: dict[str, Any]


def analyze_roi(frames: Any, *, roi: tuple[int, int, int, int]) -> RoiStats:
    """Placeholder for ROI analysis pipeline (implementation to be moved here)."""
    return RoiStats(mean=0.0, std=0.0, n=0, meta={"roi": roi})
