from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SweepPersistedConfig:
    """Config persisted to config.json for a sweep session."""

    freqs: list[float]
    n_target: int
    max_attempt: int
    settle_s: float
    daq_mode: str
    device: str
    sequence_json: str
    insert_index: int
    ao_width_ms: float
    camera_mode: str
    camera_exposure_s: float
    fg_amp_mvpp: float
    dry_image_dir: str
    roi_bootstrap: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "freqs": list(self.freqs),
            "n_target": int(self.n_target),
            "max_attempt": int(self.max_attempt),
            "settle_s": float(self.settle_s),
            "daq_mode": str(self.daq_mode),
            "device": str(self.device),
            "sequence_json": str(self.sequence_json),
            "insert_index": int(self.insert_index),
            "ao_width_ms": float(self.ao_width_ms),
            "camera_mode": str(self.camera_mode),
            "camera_exposure_s": float(self.camera_exposure_s),
            "fg_amp_mvpp": float(self.fg_amp_mvpp),
            "dry_image_dir": str(self.dry_image_dir),
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
