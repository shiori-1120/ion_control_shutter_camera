from __future__ import annotations

from typing import Any


def require_connected(app: Any) -> None:
    if not app._daq.connected:
        raise RuntimeError("Not connected")