from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DaqClient:
    """Thin client for request/response over DAQ worker queues.

    This isolates Queue put/get + pairing lock from GUI code.
    """

    _cmd_q: Any | None = None
    _resp_q: Any | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def connected(self) -> bool:
        return self._cmd_q is not None and self._resp_q is not None

    def attach(self, cmd_q: Any, resp_q: Any) -> None:
        self._cmd_q = cmd_q
        self._resp_q = resp_q

    def detach(self) -> None:
        self._cmd_q = None
        self._resp_q = None

    def try_close(self) -> None:
        try:
            if self._cmd_q is not None:
                self._cmd_q.put({"cmd": "close"})
        except Exception:
            pass

    def request(self, cmd: dict, *, timeout: float = 5.0) -> dict:
        # Serialize to keep request/response pairing correct.
        with self._lock:
            if self._cmd_q is None or self._resp_q is None:
                raise RuntimeError("Not connected")
            self._cmd_q.put(cmd)
            resp = self._resp_q.get(timeout=timeout)
            if not isinstance(resp, dict):
                raise RuntimeError(f"Invalid DAQ response: {resp!r}")
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "DAQ error"))
            return resp
