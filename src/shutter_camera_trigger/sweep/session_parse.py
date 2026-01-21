from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


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
    do_sequence: list[tuple[int, float]]
    ao_insert_index: int
    ao_width_ms: float
    camera_actions: list[dict[str, Any]]
    sync_markers: list[dict[str, Any]]


def _format_sequence_text(do_sequence: Iterable[tuple[int, float]], *, bits: int = 4) -> str:
    lines = [
        "# Sequence generated from do_sequence",
        "# Format: <BITSTRING> <hold_s>",
    ]
    for value, hold_s in do_sequence:
        bitstring = format(int(value), f"0{int(bits)}b")
        lines.append(f"{bitstring} {float(hold_s):.6f}")
    return "\n".join(lines) + "\n"


def _parse_do_sequence_from_json(seq_data: dict) -> list[tuple[int, float]]:
    raw = seq_data.get("do_sequence")
    if raw is None:
        raise ValueError("sequence_json must include do_sequence")
    if not isinstance(raw, list) or not raw:
        raise ValueError("do_sequence must be a non-empty list")
    seq: list[tuple[int, float]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"do_sequence[{idx}] must be an object")
        if "value" not in item or "hold_s" not in item:
            raise ValueError(f"do_sequence[{idx}] must include value and hold_s")
        try:
            value = int(item["value"])
            hold_s = float(item["hold_s"])
        except Exception as e:
            raise ValueError(f"Invalid do_sequence[{idx}] value/hold_s") from e
        if value < 0 or value > 0b1111:
            raise ValueError(f"do_sequence[{idx}] value must be 0..15")
        if hold_s <= 0:
            raise ValueError(f"do_sequence[{idx}] hold_s must be > 0")
        seq.append((value, hold_s))
    return seq


def _parse_camera_actions(seq_data: dict) -> list[dict[str, Any]]:
    raw = seq_data.get("camera_actions") or []
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("camera_actions must be a list")
    actions: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"camera_actions[{idx}] must be an object")
        if "t_s" not in item or "kind" not in item:
            raise ValueError(f"camera_actions[{idx}] must include t_s and kind")
        try:
            t_s = float(item["t_s"])
        except Exception as e:
            raise ValueError(f"camera_actions[{idx}] invalid t_s") from e
        kind = str(item["kind"]).strip()
        if not kind:
            raise ValueError(f"camera_actions[{idx}] kind must be non-empty")
        meta = item.get("meta") or {}
        if not isinstance(meta, dict):
            raise ValueError(f"camera_actions[{idx}] meta must be an object")
        actions.append({"t_s": t_s, "kind": kind, "meta": dict(meta)})
    return actions


def _parse_sync_markers(seq_data: dict) -> list[dict[str, Any]]:
    raw = seq_data.get("sync_markers") or []
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("sync_markers must be a list")
    markers: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"sync_markers[{idx}] must be an object")
        if "t_s" not in item or "label" not in item:
            raise ValueError(f"sync_markers[{idx}] must include t_s and label")
        try:
            t_s = float(item["t_s"])
        except Exception as e:
            raise ValueError(f"sync_markers[{idx}] invalid t_s") from e
        label = str(item["label"]).strip()
        if not label:
            raise ValueError(f"sync_markers[{idx}] label must be non-empty")
        markers.append({"t_s": t_s, "label": label})
    return markers


def read_sequence_json_params(*, seq_path: Path) -> SequenceJsonParams:
    """Read sequence JSON and return text + AO params used by the GUI."""

    seq_data = json.loads(Path(seq_path).read_text(encoding="utf-8-sig"))
    if not isinstance(seq_data, dict):
        raise ValueError("sequence-json must be a JSON object")

    raw = seq_data.get("sequence_text", "")
    if raw is None:
        raw = ""
    if not isinstance(raw, str):
        raise ValueError("sequence_text must be a string")

    insert_index = int(seq_data.get("ao_insert_index", -1))
    ao_width_ms = float(seq_data.get("ao_width_ms", 15.0))

    do_sequence = _parse_do_sequence_from_json(seq_data)
    camera_actions = _parse_camera_actions(seq_data)
    sync_markers = _parse_sync_markers(seq_data)
    if insert_index < -1 or insert_index >= len(do_sequence):
        raise ValueError("ao_insert_index must be -1..len(do_sequence)-1")
    if ao_width_ms < 0:
        raise ValueError("ao_width_ms must be >= 0")

    display_text = raw
    if not display_text.strip():
        display_text = _format_sequence_text(do_sequence, bits=4)

    return SequenceJsonParams(
        sequence_text=display_text,
        do_sequence=do_sequence,
        ao_insert_index=insert_index,
        ao_width_ms=ao_width_ms,
        camera_actions=camera_actions,
        sync_markers=sync_markers,
    )
