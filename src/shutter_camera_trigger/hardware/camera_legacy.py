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


class CameraQueueDevice:
    """Adapter for raw camera worker queues (send-only helpers)."""

    def __init__(self, *, cmd_q: Queue) -> None:
        self._cmd_q = cmd_q

    def send_get_state(self, timeout_s: float) -> None:
        self._cmd_q.put({"cmd": "get_state", "timeout_s": float(timeout_s)})

    def send_get_frame(self, timeout_s: float, *, prefer_sample: str | None = None) -> None:
        cmd = {"cmd": "get_frame", "timeout_s": float(timeout_s)}
        if prefer_sample:
            cmd["prefer_sample"] = str(prefer_sample)
        self._cmd_q.put(cmd)

    def set_roi(self, roi: list[int] | None) -> None:
        payload = list(roi) if roi is not None else None
        self._cmd_q.put({"cmd": "set_roi", "roi": payload})

    def set_threshold(self, tau_on: float, tau_off: float) -> None:
        self._cmd_q.put({"cmd": "set_threshold", "tau_on": float(tau_on), "tau_off": float(tau_off)})

    def set_subarray(self, subarray: list[int] | None) -> None:
        payload = list(subarray) if subarray is not None else None
        self._cmd_q.put({"cmd": "set_subarray", "subarray": payload})

    def close(self) -> None:
        try:
            self._cmd_q.put({"cmd": "close"})
        except Exception:
            pass
