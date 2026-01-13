from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class DaqSequenceCommand:
    do_sequence: list[tuple[int, float]]
    ao_insert_index: int
    ao_width_ms: float
    ao_rate_hz: float
    ao_v_high: float
    ao_v_low: float


class DaqDevice(Protocol):
    def open(self, device: str) -> None: ...

    def set_do(self, value: int) -> None: ...

    def run_sequence_once(self, spec: DaqSequenceCommand) -> None: ...

    def all_off(self) -> None: ...

    def close(self) -> None: ...
