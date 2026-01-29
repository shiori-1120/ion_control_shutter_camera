from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable


class SweepPhase(str, Enum):
    IDLE = "idle"
    PREPARED = "prepared"
    ROI_DONE = "roi_done"
    THRESHOLD_DONE = "threshold_done"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class SweepState:
    phase: SweepPhase = SweepPhase.IDLE
    procs: list[Any] = field(default_factory=list)
    queues: dict[str, Any] = field(default_factory=dict)
    freqs: list[float] = field(default_factory=list)
    results: list[tuple[float, int, int]] = field(default_factory=list)
    out_dir: Path | None = None
    session: dict[str, Any] | None = None
    spectrum_outputs: dict[str, Path] = field(default_factory=dict)
    threshold_samples: list[float] = field(default_factory=list)
    threshold_profiles: list[Any] = field(default_factory=list)
    threshold_roi: list[int] | None = None
    threshold_tau: float | None = None
    threshold_tau_on: float | None = None
    threshold_tau_off: float | None = None


@dataclass(frozen=True)
class SweepInput:
    freqs: list[float]
    do_sequence: list[tuple[int, float]]
    insert_index: int
    ao_width_ms: float
    camera_actions: list[dict[str, Any]]
    sync_markers: list[dict[str, Any]]
    n_target: int
    max_attempt: int
    settle_s: float
    update_interval: float
    daq_mode: str
    cam_mode: str
    cam_exposure_s: float
    device: str
    visa_res: str
    no_fg: bool
    fg_amp_vpp: float
    trig_cfg: dict[str, Any]
    seq_path: Path
    camera_verbose: bool


@dataclass(frozen=True)
class SweepDeps:
    write_sweep_config_json: Callable[..., Any]
    SweepPersistedConfig: Any
    create_sweep_workers: Callable[..., Any]
    bootstrap_workers_for_sweep: Callable[..., Any]
    build_sweep_session_dict: Callable[..., Any]
    run_roi_bootstrap_stage: Callable[..., Any]
    stop_sweep_workers: Callable[..., Any]
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]]
    ui_pump: Callable[[], None]
    AO_RATE_HZ: float
    NM_397: int
    CAMERA_TRIGGER: int
    ROI_PULSE_S: float
    ROI_IDLE_S: float
    ROI_MAX_ATTEMPT: int
    log_dir: Any | None
    run_id: str | None
    output_root: Any | None


@dataclass(frozen=True)
class SweepEvents:
    on_status: Callable[[str], None]
    on_warning: Callable[[str], None]
    on_error: Callable[[str, str], None]
    on_input_error: Callable[[str], None]
    on_plot_reset: Callable[[], None]
    on_plot_update: Callable[[int, float, int, int], None]
    on_state_change: Callable[[SweepPhase, SweepPhase], None]


@dataclass(frozen=True)
class SweepIO:
    toggle_controls: Callable[[bool], None]
    refresh_buttons: Callable[[], None]
    cleanup_stale_workers: Callable[[], None]
    apply_subarray: Callable[[dict], None]
    write_last_worker_pids_cb: Callable[[dict], None]
    format_worker_failure: Callable[..., str]
    confirm_threshold: Callable[[dict[str, Any], float, float], bool]
    update_threshold_ui: Callable[[float, float, float], None]
    get_threshold_save_frames: Callable[[], bool]
    join_with_ui: Callable[[Any, float], None]
    set_last_error_cb: Callable[[str, str, str | None], None]
