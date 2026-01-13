from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from src.shutter_camera_trigger.sweep import workflow
from src.shutter_camera_trigger.sweep.model import SweepDeps, SweepEvents, SweepIO, SweepInput, SweepState


@dataclass
class _DummyPersistedConfig:
    def to_json_dict(self) -> dict:
        return {}


def _fake_prepare_sweep_session(**kwargs) -> SimpleNamespace:
    session = {
        "freqs": [1.0],
        "do_sequence": [(0, 0.001)],
        "insert_index": -1,
        "ao_width_ms": 0.0,
        "n_target": 1,
        "max_attempt": 1,
        "settle_s": 0.0,
        "update_interval": 0.2,
        "visa_res": "",
        "no_fg": True,
        "fg_amp_vpp": 0.0,
        "cam_exposure_s": 0.001,
        "seq_cmd": None,
    }
    workers = SimpleNamespace(
        daq_proc=None,
        cam_proc=None,
        daq_cmd_q=object(),
        daq_resp_q=object(),
        cam_cmd_q=object(),
        cam_resp_q=object(),
    )
    return SimpleNamespace(ok=True, session=session, workers=workers)


def main() -> None:
    events = SweepEvents(
        on_status=lambda msg: print(f"[status] {msg}"),
        on_warning=lambda msg: print(f"[warn] {msg}"),
        on_error=lambda title, msg: print(f"[error] {title}: {msg}"),
        on_input_error=lambda msg: print(f"[input] {msg}"),
        on_plot_reset=lambda: print("[plot] reset"),
        on_plot_update=lambda *_: None,
    )
    io = SweepIO(
        toggle_controls=lambda enable: print(f"[ui] controls={enable}"),
        refresh_buttons=lambda: print("[ui] refresh_buttons"),
        cleanup_stale_workers=lambda: None,
        apply_subarray=lambda cfg: None,
        write_last_worker_pids_cb=lambda data: None,
        format_worker_failure=lambda *args, **kwargs: "worker_failed",
        confirm_threshold=lambda *args, **kwargs: True,
        join_with_ui=lambda *args, **kwargs: None,
        set_last_error_cb=lambda *args, **kwargs: None,
    )
    deps = SweepDeps(
        write_sweep_config_json=lambda *args, **kwargs: None,
        SweepPersistedConfig=_DummyPersistedConfig,
        create_sweep_workers=lambda *args, **kwargs: None,
        bootstrap_workers_for_sweep=lambda *args, **kwargs: None,
        build_sweep_session_dict=lambda *args, **kwargs: {},
        run_roi_bootstrap_stage=lambda *args, **kwargs: True,
        stop_sweep_workers=lambda *args, **kwargs: [],
        mpq_get_with_ui=lambda *args, **kwargs: {"ok": True},
        ui_pump=lambda: None,
        AO_RATE_HZ=5000.0,
        NM_397=1,
        CAMERA_TRIGGER=4,
        ROI_PULSE_S=0.002,
        ROI_IDLE_S=0.002,
        ROI_MAX_ATTEMPT=1,
        log_dir=None,
        run_id=None,
    )

    inputs = SweepInput(
        freqs=[1.0],
        do_sequence=[(0, 0.001)],
        insert_index=-1,
        ao_width_ms=0.0,
        n_target=1,
        max_attempt=1,
        settle_s=0.0,
        update_interval=0.2,
        daq_mode="dry",
        cam_mode="dry",
        cam_exposure_s=0.001,
        device="Dev1",
        visa_res="",
        no_fg=True,
        fg_amp_vpp=0.0,
        trig_cfg={"source": "EXTERNAL"},
        seq_path=Path("dummy_sequence.json"),
        camera_verbose=False,
    )

    state = SweepState()
    orig_prepare = workflow.prepare_sweep_session
    workflow.prepare_sweep_session = _fake_prepare_sweep_session
    try:
        ok = workflow.prepare_session(state=state, inputs=inputs, events=events, io=io, deps=deps)
        print(f"[result] prepare_session ok={ok}")
    finally:
        workflow.prepare_sweep_session = orig_prepare

    state.out_dir = None
    workflow.stop_sweep(state=state, events=events, io=io, deps=deps, clean_only=True, fig=None)


if __name__ == "__main__":
    main()
