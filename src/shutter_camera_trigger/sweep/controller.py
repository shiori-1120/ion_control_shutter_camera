from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .roi_threshold_flow import run_roi_check_flow, run_threshold_flow
from .session_ready import prepare_sweep_session
from .spectrum_flow import run_spectrum_flow
from .spectrum_ui import save_spectrum_plot


@dataclass
class SweepState:
    running: bool = False
    prepared: bool = False
    threshold_done: bool = False
    procs: list[Any] = field(default_factory=list)
    queues: dict[str, Any] = field(default_factory=dict)
    freqs: list[float] = field(default_factory=list)
    results: list[tuple[float, int, int]] = field(default_factory=list)
    out_dir: Path | None = None
    session: dict[str, Any] | None = None


@dataclass(frozen=True)
class SweepInput:
    freqs: list[float]
    do_sequence: list[tuple[int, float]]
    insert_index: int
    ao_width_ms: float
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
    dry_image_dir: str
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
    AO_RATE_HZ: float
    NM_397: int
    CAMERA_TRIGGER: int
    ROI_PULSE_S: float
    ROI_IDLE_S: float
    ROI_MAX_ATTEMPT: int
    log_dir: Any | None
    run_id: str | None


@dataclass(frozen=True)
class SweepUi:
    status_cb: Callable[[str], None]
    messagebox: Any
    ui_pump: Callable[[], None]
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]]
    toggle_controls: Callable[[bool], None]
    refresh_buttons: Callable[[], None]
    cleanup_stale_workers: Callable[[], None]
    apply_subarray_cb: Callable[[dict], None]
    write_last_worker_pids_cb: Callable[[dict], None]
    format_worker_failure: Callable[..., str]
    confirm_threshold_cb: Callable[[dict[str, Any], float, float], bool]
    warn_cb: Callable[[str], None]
    reset_plot_cb: Callable[[], None]
    update_plot_cb: Callable[[int, float, int, int], None]
    join_with_ui: Callable[[Any, float], None]


class SweepController:
    def __init__(self, *, ui: SweepUi, deps: SweepDeps) -> None:
        self._ui = ui
        self._deps = deps

    def prepare_session(self, state: SweepState, inputs: SweepInput) -> bool:
        if state.prepared:
            return True
        if state.running:
            return False

        self._ui.cleanup_stale_workers()
        if inputs.cam_mode == "real" and inputs.daq_mode != "real":
            self._ui.messagebox.showerror(
                "Sweep", "Camera mode is real but DAQ mode is not real. Set DAQ mode to real."
            )
            return False

        state.running = True
        self._ui.toggle_controls(False)
        self._ui.status_cb("Starting session...")
        self._ui.refresh_buttons()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/output/spectrum") / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        state.out_dir = out_dir

        result = prepare_sweep_session(
            freqs=inputs.freqs,
            do_sequence=inputs.do_sequence,
            insert_index=inputs.insert_index,
            ao_width_ms=inputs.ao_width_ms,
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
            dry_image_dir=inputs.dry_image_dir,
            camera_verbose=inputs.camera_verbose,
            subarray_cb=self._ui.apply_subarray_cb,
            write_sweep_config_json=self._deps.write_sweep_config_json,
            SweepPersistedConfig=self._deps.SweepPersistedConfig,
            create_sweep_workers=self._deps.create_sweep_workers,
            bootstrap_workers_for_sweep=self._deps.bootstrap_workers_for_sweep,
            build_sweep_session_dict=self._deps.build_sweep_session_dict,
            run_roi_bootstrap_stage=self._deps.run_roi_bootstrap_stage,
            AO_RATE_HZ=self._deps.AO_RATE_HZ,
            NM_397=self._deps.NM_397,
            CAMERA_TRIGGER=self._deps.CAMERA_TRIGGER,
            ROI_PULSE_S=self._deps.ROI_PULSE_S,
            ROI_IDLE_S=self._deps.ROI_IDLE_S,
            ROI_MAX_ATTEMPT=self._deps.ROI_MAX_ATTEMPT,
            log_dir=self._deps.log_dir,
            run_id=self._deps.run_id,
            out_dir=out_dir,
            mpq_get_with_ui=self._ui.mpq_get_with_ui,
            ui_pump=self._ui.ui_pump,
            status_cb=self._ui.status_cb,
            messagebox=self._ui.messagebox,
            stop_sweep_cb=lambda clean_only: self.stop_sweep(state, clean_only=clean_only, fig=None),
            write_last_worker_pids_cb=self._ui.write_last_worker_pids_cb,
            format_worker_failure=self._ui.format_worker_failure,
        )
        if not result.ok or result.session is None or result.workers is None:
            return False

        state.session = result.session
        state.prepared = True
        state.threshold_done = False

        workers = result.workers
        state.procs = [workers.daq_proc, workers.cam_proc]
        state.queues = {
            "daq_cmd": workers.daq_cmd_q,
            "daq_resp": workers.daq_resp_q,
            "cam_cmd": workers.cam_cmd_q,
            "cam_resp": workers.cam_resp_q,
        }

        self._ui.status_cb("Session ready. Step 1: ROI check.")
        self._ui.refresh_buttons()
        return True

    def roi_check(self, state: SweepState, *, fig: Any, canvas: Any) -> None:
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
                (self._deps.NM_397, self._deps.ROI_IDLE_S),
                (self._deps.NM_397 | self._deps.CAMERA_TRIGGER, self._deps.ROI_PULSE_S),
                (self._deps.NM_397, self._deps.ROI_IDLE_S),
            ]
            prefer_sample = None
            try:
                if state.session and state.session.get("cam_mode") == "dry":
                    prefer_sample = "data/input/dry_samples/roi_test.npy"
            except Exception:
                prefer_sample = None

            cam_log_base = self._deps.log_dir or state.out_dir
            cam_log_path = str((Path(cam_log_base) / "camera_worker.log") if cam_log_base else "") or None
            r = run_roi_check_flow(
                daq_cmd_q=daq_cmd_q,
                daq_resp_q=daq_resp_q,
                cam_cmd_q=cam_cmd_q,
                cam_resp_q=cam_resp_q,
                pulse_seq=pulse_seq,
                ao_rate_hz=self._deps.AO_RATE_HZ,
                out_dir=state.out_dir,
                cam_log_path=cam_log_path,
                mpq_get_with_ui=self._ui.mpq_get_with_ui,
                ui_pump=self._ui.ui_pump,
                status_cb=self._ui.status_cb,
                fig=fig,
                canvas=canvas,
                prefer_sample_path=prefer_sample,
                session=state.session,
            )
            roi = r.roi

            if roi is None:
                self._ui.status_cb("ROI: failed to detect ROI. Retry Step 1.")
            else:
                self._ui.status_cb("ROI: locked. Step 2: Threshold.")
            self._ui.refresh_buttons()
        except Exception as e:
            self._ui.messagebox.showerror("Sweep", str(e))

    def threshold_check(self, state: SweepState, *, fig: Any, canvas: Any) -> None:
        if not state.prepared or not state.running or not state.session:
            self._ui.messagebox.showerror("Sweep", "Run '1) ROI check' first.")
            return

        roi = state.session.get("roi")
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            self._ui.messagebox.showerror("Sweep", "ROI is not set. Run '1) ROI check' first.")
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
                ao_rate_hz=self._deps.AO_RATE_HZ,
                mpq_get_with_ui=self._ui.mpq_get_with_ui,
                ui_pump=self._ui.ui_pump,
                status_cb=self._ui.status_cb,
                fig=fig,
                canvas=canvas,
                out_dir=state.out_dir,
                confirm_apply_cb=self._ui.confirm_threshold_cb,
            )
            if applied:
                acc = float(r.agreement)
                state.threshold_done = True
                self._ui.status_cb(f"Threshold applied. agreement={acc*100:.1f}%. Step 3: Start spectrum.")
                self._ui.refresh_buttons()
            else:
                self._ui.status_cb("Threshold plotted (not applied).")

        except Exception as e:
            self._ui.messagebox.showerror("Sweep", str(e))

    def start_sweep(
        self,
        state: SweepState,
        *,
        fig: Any,
        canvas: Any,
        fg_connected: bool,
        fg_handle: Any | None,
        fallback_fg_amp_vpp: float,
    ) -> None:
        if not state.prepared or not state.threshold_done or not state.session:
            self._ui.messagebox.showerror("Sweep", "Run '1) ROI check' and '2) Threshold' first.")
            return

        if not state.running:
            return

        freqs: list[float] = list(state.session["freqs"])
        do_sequence = state.session["do_sequence"]
        insert_index = int(state.session["insert_index"])
        ao_width_ms = float(state.session["ao_width_ms"])

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

        out_dir = state.out_dir or Path("data/output/spectrum") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        state.out_dir = out_dir
        state.freqs = freqs
        state.results = []

        try:
            if fig is not None and canvas is not None:
                self._ui.reset_plot_cb()
        except Exception:
            pass

        try:
            r = run_spectrum_flow(
                freqs=freqs,
                do_sequence=do_sequence,
                insert_index=int(insert_index),
                ao_width_ms=float(ao_width_ms),
                n_target=int(n_target),
                max_attempt=int(max_attempt),
                settle_s=float(settle_s),
                update_interval_s=float(update_interval),
                daq_cmd_q=daq_cmd_q,
                daq_resp_q=daq_resp_q,
                cam_cmd_q=cam_cmd_q,
                cam_resp_q=cam_resp_q,
                ao_rate_hz=self._deps.AO_RATE_HZ,
                mpq_get_with_ui=self._ui.mpq_get_with_ui,
                should_stop=lambda: (not bool(state.running)),
                ui_pump=self._ui.ui_pump,
                status_cb=self._ui.status_cb,
                update_point_cb=self._ui.update_plot_cb,
                out_dir=out_dir,
                fg_connected=fg_connected,
                fg_handle=fg_handle,
                fg_amp_vpp=fg_amp_vpp,
                visa_res=visa_res,
                no_fg=no_fg,
                warn_cb=self._ui.warn_cb,
            )
            state.results = list(r.results)

        except Exception as e:
            self._ui.messagebox.showerror("Sweep", str(e))

        finally:
            self.stop_sweep(state, clean_only=True, fig=fig)

    def stop_sweep(self, state: SweepState, *, clean_only: bool = False, fig: Any | None = None) -> None:
        state.running = False
        state.procs = self._deps.stop_sweep_workers(
            queues=state.queues,
            procs=state.procs,
            nm_397=self._deps.NM_397,
            join_with_ui=self._ui.join_with_ui,
            write_last_worker_pids_cb=self._ui.write_last_worker_pids_cb,
        )

        state.prepared = False
        state.threshold_done = False
        state.session = None

        self._ui.toggle_controls(True)
        self._ui.status_cb("Idle" if clean_only else "Stopped")
        self._ui.refresh_buttons()

        if state.out_dir and fig is not None:
            try:
                save_spectrum_plot(fig, state.out_dir, dpi=120)
            except Exception:
                pass
