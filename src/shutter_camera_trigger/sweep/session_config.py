from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..hardware import DaqSequenceCommand


@dataclass(frozen=True)
class SweepPersistedConfig:
    """Config persisted to config.json for a sweep session."""

    freqs: list[float]
    n_target: int
    max_attempt: int
    settle_s: float
    update_interval: float
    daq_mode: str
    device: str
    sequence_json: str
    insert_index: int
    ao_width_ms: float
    camera_actions: list[dict[str, Any]]
    sync_markers: list[dict[str, Any]]
    camera_mode: str
    camera_exposure_s: float
    fg_amp_mvpp: float
    roi_bootstrap: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "freqs": list(self.freqs),
            "n_target": int(self.n_target),
            "max_attempt": int(self.max_attempt),
            "settle_s": float(self.settle_s),
            "update_interval": float(self.update_interval),
            "daq_mode": str(self.daq_mode),
            "device": str(self.device),
            "sequence_json": str(self.sequence_json),
            "insert_index": int(self.insert_index),
            "ao_width_ms": float(self.ao_width_ms),
            "camera_actions": list(self.camera_actions),
            "sync_markers": list(self.sync_markers),
            "camera_mode": str(self.camera_mode),
            "camera_exposure_s": float(self.camera_exposure_s),
            "fg_amp_mvpp": float(self.fg_amp_mvpp),
            "roi_bootstrap": dict(self.roi_bootstrap),
        }


def write_sweep_config_json(*, out_dir: Path, cfg: SweepPersistedConfig) -> Path:
    """Write sweep config.json and return its path."""

    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "config.json"
    p.write_text(json.dumps(cfg.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def build_sweep_session_dict(
    *,
    freqs: list[float],
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    ao_width_ms: float,
    seq_cmd: DaqSequenceCommand | None,
    camera_actions: list[dict[str, Any]],
    sync_markers: list[dict[str, Any]],
    n_target: int,
    max_attempt: int,
    settle_s: float,
    update_interval: float,
    daq_mode: str,
    cam_mode: str,
    device: str,
    visa_res: str,
    no_fg: bool,
    fg_amp_vpp: float,
    trig_cfg: dict[str, Any],
    cam_exposure_s: float,
    seq_path: str,
) -> dict[str, Any]:
    """Build the in-memory session dict used by the GUI sweep runtime."""

    return {
        "freqs": list(freqs),
        "do_sequence": list(do_sequence),
        "insert_index": int(insert_index),
        "ao_width_ms": float(ao_width_ms),
        "seq_cmd": seq_cmd,
        "camera_actions": list(camera_actions),
        "sync_markers": list(sync_markers),
        "n_target": int(n_target),
        "max_attempt": int(max_attempt),
        "settle_s": float(settle_s),
        "update_interval": float(update_interval),
        "daq_mode": str(daq_mode),
        "cam_mode": str(cam_mode),
        "device": str(device),
        "visa_res": str(visa_res),
        "no_fg": bool(no_fg),
        "fg_amp_vpp": float(fg_amp_vpp),
        "trig_cfg": dict(trig_cfg),
        "cam_exposure_s": float(cam_exposure_s),
        "seq_path": str(seq_path),
    }


def build_daq_sequence_command(
    *,
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    ao_width_ms: float,
    ao_rate_hz: float,
    ao_v_high: float = 5.0,
    ao_v_low: float = 0.0,
) -> DaqSequenceCommand:
    return DaqSequenceCommand(
        do_sequence=list(do_sequence),
        ao_insert_index=int(insert_index),
        ao_width_ms=float(ao_width_ms),
        ao_rate_hz=float(ao_rate_hz),
        ao_v_high=float(ao_v_high),
        ao_v_low=float(ao_v_low),
    )


def write_manifest_json(*, out_dir: Path, run_type: str, files: dict[str, Path]) -> Path:
    """Write manifest.json listing generated artifacts."""

    manifest = {
        "run_type": str(run_type),
        "files": [],
    }
    for label, path in files.items():
        try:
            p = Path(path)
        except Exception:
            continue
        if not p.exists():
            continue
        manifest["files"].append(
            {
                "name": str(label),
                "path": str(p.name),
            }
        )

    p = out_dir / "manifest.json"
    p.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return p
