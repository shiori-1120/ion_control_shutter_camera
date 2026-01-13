from __future__ import annotations

from typing import Any

import numpy as np


def robust_gray_limits(
    img: Any, *, lo_pct: float = 1.0, hi_pct: float = 99.0
) -> tuple[float | None, float | None]:
    """Return (vmin, vmax) for grayscale imshow using robust percentiles."""

    try:
        arr = np.asarray(img)
        if arr.size == 0:
            return (None, None)
        a = np.asarray(arr, dtype=float)
        if not np.isfinite(a).any():
            return (None, None)
        vmin = float(np.nanpercentile(a, float(lo_pct)))
        vmax = float(np.nanpercentile(a, float(hi_pct)))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (None, None)
        if vmax <= vmin:
            return (None, None)
        return (vmin, vmax)
    except Exception:
        return (None, None)
