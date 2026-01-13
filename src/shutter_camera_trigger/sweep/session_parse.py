from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def parse_freqs_from_expressions(*, start_expr: str, stop_expr: str, step_expr: str) -> list[float]:
    """Parse freq start/stop/step expressions and generate frequency list.

    The GUI historically accepts expressions like "80e6" via eval.
    This function preserves that behavior.
    """

    freq_start = float(eval(str(start_expr)))
    freq_stop = float(eval(str(stop_expr)))
    freq_step = float(eval(str(step_expr)))

    if freq_step == 0:
        raise ValueError("freq_step must be non-zero")

    freqs: list[float] = []
    f = float(freq_start)

    if freq_step > 0:
        while f <= freq_stop + 1e-12:
            freqs.append(float(f))
            f += freq_step
    else:
        while f >= freq_stop - 1e-12:
            freqs.append(float(f))
            f += freq_step

    if not freqs:
        raise ValueError("No frequencies generated")

    return freqs


@dataclass(frozen=True)
class SequenceJsonParams:
    sequence_text: str
    ao_insert_index: int
    ao_width_ms: float


def read_sequence_json_params(*, seq_path: Path) -> SequenceJsonParams:
    """Read sequence JSON and return text + AO params used by the GUI."""

    seq_data = json.loads(Path(seq_path).read_text(encoding="utf-8"))
    if not isinstance(seq_data, dict):
        raise ValueError("sequence-json must be a JSON object")

    raw = seq_data.get("sequence_text", "")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raise ValueError("sequence_text must be a string")

    insert_index = int(seq_data.get("ao_insert_index", -1))
    ao_width_ms = float(seq_data.get("ao_width_ms", 15.0))

    return SequenceJsonParams(sequence_text=raw, ao_insert_index=insert_index, ao_width_ms=ao_width_ms)
