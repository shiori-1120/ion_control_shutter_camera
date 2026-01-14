from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .model import SweepDeps, SweepEvents, SweepInput, SweepIO, SweepPhase, SweepState

_MSG_NEED_ROI = "Run '1) ROI check' first."
_MSG_NEED_ROI_SET = "ROI is not set. Run '1) ROI check' first."
_MSG_NEED_THRESH = "Run '1) ROI check' and '2) Threshold' first."
from .roi_threshold_flow import run_roi_check_flow, run_threshold_flow
from .session_config import write_manifest_json
from .session_ready import prepare_sweep_session
from .spectrum_flow import run_spectrum_flow
from .spectrum_ui import save_spectrum_plot


def prepare_session(
    *,
    state: SweepState,
    inputs: SweepInput,
    events: SweepEvents,
    io: SweepIO,
    deps: SweepDeps,
) -> bool:
    if state.phase in {SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE}:
        return True
    if state.phase in {SweepPhase.RUNNING, SweepPhase.STOPPING, SweepPhase.ERROR}:
        return False

    io.cleanup_stale_workers()
    if inputs.cam_mode == "real" and inputs.daq_mode != "real":
        events.on_error("Sweep", "Camera mode is real but DAQ mode is not real. Set DAQ mode to real.")
        io.set_last_error_cb(
            "Sweep",
            "Camera mode is real but DAQ mode is not real.",
            None,
        )
        return False

    io.toggle_controls(False)
    events.on_status("Preparing session...")
    io.refresh_buttons()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = Path(deps.output_root) if deps.output_root else Path("data/output")
    out_dir = base_output / "spectrum" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    state.out_dir = out_dir

    result = prepare_sweep_session(
        freqs=inputs.freqs,
        do_sequence=inputs.do_sequence,
        insert_index=inputs.insert_index,
        ao_width_ms=inputs.ao_width_ms,
        camera_actions=inputs.camera_actions,
        sync_markers=inputs.sync_markers,
        n_target=inputs.n_target,
        max_attempt=inputs.max_attempt,
        settle_s=inputs.settle_s,
        update_interval=inputs.update_interval,
        daq_mode=inputs.daq_mode,
        cam_mode=inputs.cam_mode,
        device=inputs.device,
        visa_res=inputs.visa_res,
        no_fg=inputs.no_fg,
        fg_amp_vpp=inputs.fg_amp_vpp,
        trig_cfg=inputs.trig_cfg,
        cam_exposure_s=inputs.cam_exposure_s,
        seq_path=inputs.seq_path,
        camera_verbose=inputs.camera_verbose,
        subarray_cb=io.apply_subarray,
        write_sweep_config_json=deps.write_sweep_config_json,
        SweepPersistedConfig=deps.SweepPersistedConfig,
        create_sweep_workers=deps.create_sweep_workers,
        bootstrap_workers_for_sweep=deps.bootstrap_workers_for_sweep,
        build_sweep_session_dict=deps.build_sweep_session_dict,
        run_roi_bootstrap_stage=deps.run_roi_bootstrap_stage,
        AO_RATE_HZ=deps.AO_RATE_HZ,
        NM_397=deps.NM_397,
        CAMERA_TRIGGER=deps.CAMERA_TRIGGER,
        ROI_PULSE_S=deps.ROI_PULSE_S,
        ROI_IDLE_S=deps.ROI_IDLE_S,
        ROI_MAX_ATTEMPT=deps.ROI_MAX_ATTEMPT,
        log_dir=deps.log_dir,
        run_id=deps.run_id,
        out_dir=out_dir,
        mpq_get_with_ui=deps.mpq_get_with_ui,
        ui_pump=deps.ui_pump,
        status_cb=events.on_status,
        show_error_cb=events.on_error,
        stop_sweep_cb=lambda clean_only: stop_sweep(
            state=state,
            events=events,
            deps=deps,
            io=io,
            clean_only=clean_only,
            fig=None,
        ),
        write_last_worker_pids_cb=io.write_last_worker_pids_cb,
        format_worker_failure=io.format_worker_failure,
    )
    if not result.ok or result.session is None or result.workers is None:
        io.set_last_error_cb("Sweep", "Prepare session failed", None)
        return False

    state.session = result.session
    _set_phase(state, events, SweepPhase.PREPARED)

    workers = result.workers
    state.procs = [workers.daq_proc, workers.cam_proc]
    state.queues = {
        "daq_cmd": workers.daq_cmd_q,
        "daq_resp": workers.daq_resp_q,
        "cam_cmd": workers.cam_cmd_q,
        "cam_resp": workers.cam_resp_q,
    }

    events.on_status("Ready: 1) ROI check")
    io.refresh_buttons()
    return True


def roi_check(
    *,
    state: SweepState,
    fig: Any,
    canvas: Any,
    events: SweepEvents,
    io: SweepIO,
    deps: SweepDeps,
) -> None:
    if state.out_dir is None:
        return
    if fig is None or canvas is None:
        return

    daq_cmd_q = state.queues.get("daq_cmd")
    daq_resp_q = state.queues.get("daq_resp")
    cam_cmd_q = state.queues.get("cam_cmd")
    cam_resp_q = state.queues.get("cam_resp")
    if not (daq_cmd_q and daq_resp_q and cam_cmd_q and cam_resp_q):
        return

    try:
        pulse_seq = [
            (deps.NM_397, deps.ROI_IDLE_S),
            (deps.NM_397 | deps.CAMERA_TRIGGER, deps.ROI_PULSE_S),
            (deps.NM_397, deps.ROI_IDLE_S),
        ]
        prefer_sample = None
        try:
            if state.session and state.session.get("cam_mode") == "dry":
                prefer_sample = "data/input/dry_samples/roi_test.npy"
        except Exception:
            prefer_sample = None

        cam_log_base = deps.log_dir or state.out_dir
        cam_log_path = str((Path(cam_log_base) / "camera_worker.log") if cam_log_base else "") or None
        r = run_roi_check_flow(
            daq_cmd_q=daq_cmd_q,
            daq_resp_q=daq_resp_q,
            cam_cmd_q=cam_cmd_q,
            cam_resp_q=cam_resp_q,
            pulse_seq=pulse_seq,
            ao_rate_hz=deps.AO_RATE_HZ,
            out_dir=state.out_dir,
            cam_log_path=cam_log_path,
            mpq_get_with_ui=deps.mpq_get_with_ui,
            ui_pump=deps.ui_pump,
            status_cb=events.on_status,
            fig=fig,
            canvas=canvas,
            prefer_sample_path=prefer_sample,
            session=state.session,
        )
        roi = r.roi

        if roi is None:
            events.on_status("ROI not detected. Retry: 1) ROI check")
        else:
            events.on_status("ROI locked. Next: 2) Threshold")
            _set_phase(state, events, SweepPhase.ROI_DONE)
        io.refresh_buttons()
    except Exception as e:
        events.on_error("Sweep", str(e))
        _set_phase(state, events, SweepPhase.ERROR)
        events.on_status("Error (ROI check)")
        io.refresh_buttons()
        io.set_last_error_cb("Sweep", str(e), cam_log_path)


def threshold_check(
    *,
    state: SweepState,
    fig: Any,
    canvas: Any,
    events: SweepEvents,
    io: SweepIO,
    deps: SweepDeps,
) -> None:
    if state.phase not in {SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE} or not state.session:
        events.on_error("Sweep", _MSG_NEED_ROI)
        io.set_last_error_cb("Sweep", _MSG_NEED_ROI, None)
        return

    roi = state.session.get("roi")
    if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
        events.on_error("Sweep", _MSG_NEED_ROI_SET)
        io.set_last_error_cb("Sweep", _MSG_NEED_ROI_SET, None)
        return

    if fig is None or canvas is None:
        return

    daq_cmd_q = state.queues.get("daq_cmd")
    daq_resp_q = state.queues.get("daq_resp")
    cam_cmd_q = state.queues.get("cam_cmd")
    cam_resp_q = state.queues.get("cam_resp")
    if not (daq_cmd_q and daq_resp_q and cam_cmd_q and cam_resp_q):
        return

    try:
        do_sequence = state.session["do_sequence"]
        n = int(state.session.get("n_target") or 50)
        max_attempt = int(state.session.get("max_attempt") or max(100, n))
        try:
            cam_exposure_s = float(state.session.get("cam_exposure_s") or 0.001)
        except Exception:
            cam_exposure_s = 0.001

        r, applied = run_threshold_flow(
            daq_cmd_q=daq_cmd_q,
            daq_resp_q=daq_resp_q,
            cam_cmd_q=cam_cmd_q,
            cam_resp_q=cam_resp_q,
            do_sequence=do_sequence,
            roi=[int(v) for v in roi],
            n_target=int(n),
            max_attempt=int(max_attempt),
            cam_exposure_s=float(cam_exposure_s),
            ao_rate_hz=deps.AO_RATE_HZ,
            mpq_get_with_ui=deps.mpq_get_with_ui,
            ui_pump=deps.ui_pump,
            status_cb=events.on_status,
            fig=fig,
            canvas=canvas,
            out_dir=state.out_dir,
            confirm_apply_cb=io.confirm_threshold,
        )
        if applied:
            acc = float(r.agreement)
            events.on_status(f"Threshold applied ({acc*100:.1f}%). Next: 3) Start spectrum")
            _set_phase(state, events, SweepPhase.THRESHOLD_DONE)
            io.refresh_buttons()
        else:
            events.on_status("Threshold plotted (not applied). Apply to continue.")

    except Exception as e:
        events.on_error("Sweep", str(e))
        _set_phase(state, events, SweepPhase.ERROR)
        events.on_status("Error (threshold)")
        io.refresh_buttons()
        io.set_last_error_cb("Sweep", str(e), None)


def start_sweep(
    *,
    state: SweepState,
    fig: Any,
    canvas: Any,
    fg_connected: bool,
    fg_handle: Any | None,
    fallback_fg_amp_vpp: float,
    events: SweepEvents,
    io: SweepIO,
    deps: SweepDeps,
) -> None:
    if state.phase is not SweepPhase.THRESHOLD_DONE or not state.session:
        events.on_error("Sweep", _MSG_NEED_THRESH)
        io.set_last_error_cb("Sweep", _MSG_NEED_THRESH, None)
        return

    if state.phase is SweepPhase.STOPPING:
        return
    if state.phase is SweepPhase.RUNNING:
        return

    freqs: list[float] = list(state.session["freqs"])
    do_sequence = state.session["do_sequence"]
    insert_index = int(state.session["insert_index"])
    ao_width_ms = float(state.session["ao_width_ms"])
    seq_cmd = state.session.get("seq_cmd")
    camera_commands = state.session.get("camera_commands") or []

    n_target = int(state.session["n_target"])
    max_attempt = int(state.session["max_attempt"])
    settle_s = float(state.session["settle_s"])
    update_interval = float(state.session["update_interval"])

    daq_cmd_q = state.queues["daq_cmd"]
    daq_resp_q = state.queues["daq_resp"]
    cam_cmd_q = state.queues["cam_cmd"]
    cam_resp_q = state.queues["cam_resp"]

    visa_res = str(state.session.get("visa_res") or "")
    no_fg = bool(state.session.get("no_fg"))
    fg_amp_vpp = float(state.session.get("fg_amp_vpp") or fallback_fg_amp_vpp)

    base_output = Path(deps.output_root) if deps.output_root else Path("data/output")
    out_dir = state.out_dir or base_output / "spectrum" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    state.out_dir = out_dir
    state.freqs = freqs
    state.results = []
    state.spectrum_outputs = {}

    try:
        if fig is not None and canvas is not None:
            events.on_plot_reset()
    except Exception:
        pass

    _set_phase(state, events, SweepPhase.RUNNING)
    events.on_status("Running: spectrum sweep")
    io.refresh_buttons()
    try:
        r = run_spectrum_flow(
            freqs=freqs,
            do_sequence=do_sequence,
            insert_index=int(insert_index),
            ao_width_ms=float(ao_width_ms),
            seq_cmd=seq_cmd,
            camera_commands=camera_commands,
            n_target=int(n_target),
            max_attempt=int(max_attempt),
            settle_s=float(settle_s),
            update_interval_s=float(update_interval),
            daq_cmd_q=daq_cmd_q,
            daq_resp_q=daq_resp_q,
            cam_cmd_q=cam_cmd_q,
            cam_resp_q=cam_resp_q,
            ao_rate_hz=deps.AO_RATE_HZ,
            mpq_get_with_ui=deps.mpq_get_with_ui,
            should_stop=lambda: (state.phase is not SweepPhase.RUNNING),
            ui_pump=deps.ui_pump,
            status_cb=events.on_status,
            update_point_cb=events.on_plot_update,
            out_dir=out_dir,
            fg_connected=fg_connected,
            fg_handle=fg_handle,
            fg_amp_vpp=fg_amp_vpp,
            visa_res=visa_res,
            no_fg=no_fg,
            warn_cb=events.on_warning,
        )
        state.results = list(r.results)
        state.spectrum_outputs = {
            "shots": Path(r.shots_csv),
            "spectrum": Path(r.spectrum_csv),
        }

    except Exception as e:
        events.on_error("Sweep", str(e))
        _set_phase(state, events, SweepPhase.ERROR)
        events.on_status("Error (sweep)")
        io.refresh_buttons()
        io.set_last_error_cb("Sweep", str(e), None)

    finally:
        stop_sweep(state=state, events=events, io=io, deps=deps, clean_only=True, fig=fig)


def stop_sweep(
    *,
    state: SweepState,
    events: SweepEvents,
    io: SweepIO,
    deps: SweepDeps,
    clean_only: bool = False,
    fig: Any | None = None,
) -> None:
    _set_phase(state, events, SweepPhase.STOPPING)
    events.on_status("Stopping sweep...")
    io.refresh_buttons()
    state.procs = deps.stop_sweep_workers(
        queues=state.queues,
        procs=state.procs,
        nm_397=deps.NM_397,
        join_with_ui=io.join_with_ui,
        write_last_worker_pids_cb=io.write_last_worker_pids_cb,
    )

    state.session = None

    io.toggle_controls(True)
    events.on_status("Idle" if clean_only else "Stopped")
    io.refresh_buttons()
    _set_phase(state, events, SweepPhase.IDLE)

    if state.out_dir and fig is not None:
        try:
            save_spectrum_plot(fig, state.out_dir, dpi=120)
        except Exception:
            pass
    if state.out_dir:
        try:
            files: dict[str, Path] = {}
            config_path = state.out_dir / "config.json"
            plot_path = state.out_dir / "spectrum.png"
            if config_path.exists():
                files["config"] = config_path
            if plot_path.exists():
                files["plot"] = plot_path
            for label, path in state.spectrum_outputs.items():
                files[label] = path
            if files:
                write_manifest_json(out_dir=state.out_dir, run_type="spectrum", files=files)
        except Exception:
            pass
    state.spectrum_outputs = {}


def _set_phase(state: SweepState, events: SweepEvents, phase: SweepPhase) -> None:
    prev = state.phase
    if prev == phase:
        return
    state.phase = phase
    try:
        events.on_state_change(prev, phase)
    except Exception:
        pass
