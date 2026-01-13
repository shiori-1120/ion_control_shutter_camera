from __future__ import annotations

import os


def limit_blas_threads() -> None:
    """Best-effort: limit BLAS threads to reduce jitter in online runs."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
