from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceParseOptions:
    bits: int = 4
    strict_bitstring_length: bool = False
    allow_symbolic_names: bool = False


def parse_do_sequence_text(
    raw: str,
    *,
    options: SequenceParseOptions | None = None,
    name_to_value: dict[str, int] | None = None,
    value_min: int = 0,
    value_max: int = 0b1111,
) -> list[tuple[int, float]]:
    """Parse a DO sequence text.

    Line format:
      <KEY> <hold_s> [ignored...]

    KEY can be:
    - bitstring (e.g. 0101)
    - integer literal (e.g. 5, 0b0101)
    - optional symbolic name if allow_symbolic_names is True
    """

    if options is None:
        options = SequenceParseOptions()

    steps: list[tuple[int, float]] = []

    for line in (raw or "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        parts = s.split()
        if len(parts) < 2:
            raise ValueError(f"Invalid sequence line: {line!r}")

        key = parts[0]
        hold_s = float(parts[1])
        if hold_s < 0:
            raise ValueError(f"hold_s must be >= 0: {line!r}")

        value: int
        if all(ch in "01" for ch in key):
            if options.strict_bitstring_length and len(key) != int(options.bits):
                raise ValueError(f"Bitstring must be {int(options.bits)} digits: {line!r}")
            value = int(key, 2)
        elif options.allow_symbolic_names and name_to_value and key in name_to_value:
            value = int(name_to_value[key])
        else:
            value = int(key, 0)

        if not (int(value_min) <= value <= int(value_max)):
            raise ValueError(f"DO value must be {value_min}..{value_max}: {line!r}")

        steps.append((int(value), float(hold_s)))

    if not steps:
        raise ValueError("Sequence is empty")

    return steps
