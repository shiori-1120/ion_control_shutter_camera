from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SpectrumResult:
    points: list[tuple[float, int, int]]
    meta: dict[str, Any]


def summarize_sweep(results: list[Any]) -> SpectrumResult:
    """Placeholder for spectrum summary (implementation to be moved here)."""
    return SpectrumResult(points=[], meta={"n": len(results)})
