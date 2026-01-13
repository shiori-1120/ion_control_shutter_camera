from __future__ import annotations

from multiprocessing import Queue
from typing import Any

from .camera_iface import CameraDevice, FrameResult


class CameraWorkerDevice(CameraDevice):
    """Adapter around camera worker queues."""

    def __init__(self, *, cmd_q: Queue, resp_q: Queue) -> None:
        self._cmd_q = cmd_q
        self._resp_q = resp_q

    def open(self, cfg: dict[str, Any]) -> None:
        self._cmd_q.put({"cmd": "open", "cfg": cfg})
        resp = self._resp_q.get(timeout=10)
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Camera open failed"))

    def prime(self, timeout_s: float) -> None:
        self._cmd_q.put({"cmd": "prime", "timeout_s": float(timeout_s)})
        resp = self._resp_q.get(timeout=max(2.0, float(timeout_s) + 2.0))
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Camera prime failed"))

    def capture(self, timeout_s: float) -> FrameResult:
        self._cmd_q.put({"cmd": "get_frame", "timeout_s": float(timeout_s)})
        resp = self._resp_q.get(timeout=max(2.0, float(timeout_s) + 2.0))
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise RuntimeError(resp.get("error", "Camera capture failed"))
        return FrameResult(frame=resp.get("frame"), roi=resp.get("roi"), meta={"event": resp.get("event")})

    def close(self) -> None:
        try:
            self._cmd_q.put({"cmd": "close"})
        except Exception:
            pass
