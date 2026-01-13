from __future__ import annotations

from typing import Protocol


class FgDevice(Protocol):
    def open(self, resource: str) -> None: ...

    def apply(self, cfg: dict[str, float | str | bool]) -> None: ...

    def close(self) -> None: ...
