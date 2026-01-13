from __future__ import annotations

from typing import Any

from ..gui_support.sequence_text import SequenceParseOptions, parse_do_sequence_text


def parse_sequence_text(raw: str, *, bits: int) -> list[tuple[int, float]]:
    return parse_do_sequence_text(
        raw,
        options=SequenceParseOptions(
            bits=bits,
            strict_bitstring_length=False,
            allow_symbolic_names=False,
        ),
        name_to_value=None,
        value_min=0,
        value_max=0b1111,
    )