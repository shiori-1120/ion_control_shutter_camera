from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

AO_RATE_HZ = 1.0 / 0.0002
AO_V_HIGH = 5.0
AO_V_LOW = 0.0


@dataclass
class DaqSequenceCommand:
    do_sequence: list[tuple[int, float]]
    ao_insert_index: int
    ao_width_ms: float
    ao_rate_hz: float = AO_RATE_HZ
    ao_v_high: float = AO_V_HIGH
    ao_v_low: float = AO_V_LOW

    def __post_init__(self) -> None:
        self.ao_rate_hz = AO_RATE_HZ
        self.ao_v_high = AO_V_HIGH
        self.ao_v_low = AO_V_LOW


class DaqDevice(Protocol):
    def open(self, device: str) -> None: ...

    def set_do(self, value: int) -> None: ...

    def run_sequence_once(self, spec: DaqSequenceCommand) -> None: ...

    def all_off(self) -> None: ...

    def close(self) -> None: ...
