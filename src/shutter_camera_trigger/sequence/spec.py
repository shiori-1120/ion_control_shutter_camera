from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..hardware import CameraCommand, DaqSequenceCommand


@dataclass(frozen=True)
class DoStep:
    value: int
    hold_s: float


@dataclass(frozen=True)
class CameraAction:
    t_s: float
    kind: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class SyncMarker:
    t_s: float
    label: str


@dataclass(frozen=True)
class SequenceSpec:
    do_sequence: list[DoStep]
    ao_insert_index: int
    ao_width_ms: float
    ao_rate_hz: float
    ao_v_high: float
    ao_v_low: float
    camera_actions: list[CameraAction]
    sync_markers: list[SyncMarker]


def build_sequence_spec(
    *,
    do_sequence: Iterable[tuple[int, float]],
    ao_insert_index: int,
    ao_width_ms: float,
    ao_rate_hz: float,
    ao_v_high: float,
    ao_v_low: float,
    camera_actions: Iterable[dict[str, Any]],
    sync_markers: Iterable[dict[str, Any]],
) -> SequenceSpec:
    return SequenceSpec(
        do_sequence=[DoStep(int(value), float(hold_s)) for value, hold_s in do_sequence],
        ao_insert_index=int(ao_insert_index),
        ao_width_ms=float(ao_width_ms),
        ao_rate_hz=float(ao_rate_hz),
        ao_v_high=float(ao_v_high),
        ao_v_low=float(ao_v_low),
        camera_actions=[
            CameraAction(float(action["t_s"]), str(action["kind"]), dict(action.get("meta") or {}))
            for action in camera_actions
        ],
        sync_markers=[
            SyncMarker(float(marker["t_s"]), str(marker["label"])) for marker in sync_markers
        ],
    )


def compile_sequence_spec(
    spec: SequenceSpec,
    *,
    default_camera_timeout_s: float = 5.0,
) -> tuple[DaqSequenceCommand, list[CameraCommand]]:
    seq_cmd = DaqSequenceCommand(
        do_sequence=[(step.value, step.hold_s) for step in spec.do_sequence],
        ao_insert_index=int(spec.ao_insert_index),
        ao_width_ms=float(spec.ao_width_ms),
        ao_rate_hz=float(spec.ao_rate_hz),
        ao_v_high=float(spec.ao_v_high),
        ao_v_low=float(spec.ao_v_low),
    )
    cam_cmds: list[CameraCommand] = []
    for action in spec.camera_actions:
        meta = dict(action.meta)
        meta.setdefault("t_s", float(action.t_s))
        timeout_s = float(meta.get("timeout_s") or default_camera_timeout_s)
        cam_cmds.append(CameraCommand(kind=str(action.kind), timeout_s=timeout_s, meta=meta))
    return seq_cmd, cam_cmds
