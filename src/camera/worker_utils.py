from __future__ import annotations

import os
from typing import Any


def limit_blas_threads() -> None:
    # Safety for online operation: avoid NumPy/SciPy consuming all cores and starving DAQ.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def to_uint8_image(arr: Any) -> Any:
    """Convert arbitrary array-like image to uint8 in [0,255] for dry-mode tests."""
    try:
        import numpy as _np

        x = _np.asarray(arr)
        if x.size == 0:
            return x.astype(_np.uint8)

        # If already uint8, keep.
        if x.dtype == _np.uint8:
            return x

        # Handle float images in [0,1]
        x_f = x.astype(float)
        finite = x_f[_np.isfinite(x_f)]
        if finite.size == 0:
            return _np.zeros_like(x_f, dtype=_np.uint8)

        vmin = float(finite.min())
        vmax = float(finite.max())

        if 0.0 <= vmin and vmax <= 1.0:
            y = _np.clip(x_f, 0.0, 1.0) * 255.0
            return _np.asarray(_np.rint(y), dtype=_np.uint8)

        # If already roughly in [0,255], just clip.
        if -1.0 <= vmin and vmax <= 256.0:
            y = _np.clip(x_f, 0.0, 255.0)
            return _np.asarray(_np.rint(y), dtype=_np.uint8)

        # Otherwise, normalize robustly (percentiles) then scale to [0,255].
        p1 = float(_np.percentile(finite, 1))
        p99 = float(_np.percentile(finite, 99))
        if not _np.isfinite(p1) or not _np.isfinite(p99) or abs(p99 - p1) < 1e-12:
            y = _np.clip(x_f, 0.0, 255.0)
            return _np.asarray(_np.rint(y), dtype=_np.uint8)

        y = (x_f - p1) / (p99 - p1)
        y = _np.clip(y, 0.0, 1.0) * 255.0
        return _np.asarray(_np.rint(y), dtype=_np.uint8)
    except Exception:
        return arr


def as_roi_tuple(x: Any) -> tuple[int, int, int, int] | None:
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) == 4:
        return (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
    return None
