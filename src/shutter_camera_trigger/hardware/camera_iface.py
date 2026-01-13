from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class FrameResult:
    frame: Any
    roi: tuple[int, int, int, int] | None
    meta: dict[str, Any]


class CameraDevice(Protocol):
    def open(self, cfg: dict[str, Any]) -> None: ...

    def prime(self, timeout_s: float) -> None: ...

    def capture(self, timeout_s: float) -> FrameResult: ...

    def close(self) -> None: ...
