from __future__ import annotations

import time
from typing import Any


def join_with_ui(app: Any, proc: Any, *, timeout: float, poll_s: float = 0.02) -> None:
    """Process.join(timeout=...) that keeps the Tk UI responsive."""
    deadline = time.time() + float(timeout)
    while True:
        try:
            if not proc.is_alive():
                return
        except Exception:
            return
        if time.time() >= deadline:
            return
        try:
            app.update()
        except Exception:
            pass
        time.sleep(poll_s)
