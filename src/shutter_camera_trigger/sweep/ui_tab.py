from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from ..gui_support.perf import limit_blas_threads
from ..gui_support.validators import apply_subarray_to_cam_cfg
from ..gui_support.worker_cleanup import cleanup_stale_workers, write_last_worker_pids
from ..gui_support.worker_messages import format_worker_failure
from ..gui_support.diagnostics import append_state_history, set_last_error
from .controller import SweepController
from .model import SweepDeps, SweepEvents, SweepIO, SweepState
from .roi_threshold_flow import format_threshold_prompt
from .session_config import SweepPersistedConfig, build_sweep_session_dict, write_sweep_config_json
from .session_start import bootstrap_workers_for_sweep
from .session_workers import create_sweep_workers
from .stages import run_roi_bootstrap_stage
from .stop_flow import stop_sweep_workers
from .ui_helpers import (
    join_with_ui,
    mpq_get_with_ui,
    refresh_sweep_buttons,
    reset_spectrum_plot_ui,
    toggle_sweep_controls,
    ui_pump,
    update_spectrum_plot_ui,
)


def build_sweep_tab(
    app: Any,
    *,
    default_daq_device: str,
    ao_rate_hz: float,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    roi_max_attempt: int,
    roi_check_cb: Callable[[], None],
    threshold_check_cb: Callable[[], None],
    threshold_override_replot_cb: Callable[[], None],
    threshold_override_apply_cb: Callable[[], None],
    start_sweep_cb: Callable[[], None],
    stop_sweep_cb: Callable[[], None],
) -> None:
    limit_blas_threads()

    row = ttk.Frame(app.sweep_tab)
    row.pack(fill=tk.X, pady=(0, 8))

    freq = ttk.LabelFrame(row, text="Frequencies")
    freq.grid(row=0, column=0, sticky=tk.W + tk.E, padx=(0, 12))

    ttk.Label(freq, text="Start").grid(row=0, column=0, sticky=tk.W)
    app.sw_freq_start = tk.StringVar(value="199e6")
    ttk.Entry(freq, textvariable=app.sw_freq_start, width=12).grid(row=0, column=1, padx=4)
    ttk.Label(freq, text="Hz").grid(row=0, column=2, sticky=tk.W)

    ttk.Label(freq, text="Stop").grid(row=0, column=3, sticky=tk.W)
    app.sw_freq_stop = tk.StringVar(value="201e6")
    ttk.Entry(freq, textvariable=app.sw_freq_stop, width=12).grid(row=0, column=4, padx=4)
    ttk.Label(freq, text="Hz").grid(row=0, column=5, sticky=tk.W)

    ttk.Label(freq, text="Step").grid(row=0, column=6, sticky=tk.W)
    app.sw_freq_step = tk.StringVar(value="0.5e6")
    ttk.Entry(freq, textvariable=app.sw_freq_step, width=12).grid(row=0, column=7, padx=4)
    ttk.Label(freq, text="Hz").grid(row=0, column=8, sticky=tk.W)

    targets = ttk.LabelFrame(row, text="Targets")
    targets.grid(row=1, column=0, sticky=tk.W + tk.E, padx=(0, 12), pady=(8, 0))

    show_debug = bool(getattr(app, "show_debug_fields", True))

    ttk.Label(targets, text="n_target").grid(row=0, column=0, sticky=tk.W)
    if getattr(app, "sw_n_target", None) is None:
        app.sw_n_target = tk.StringVar(value="50")
    ttk.Entry(targets, textvariable=app.sw_n_target, width=8).grid(row=0, column=1, padx=4)

    if show_debug:
        ttk.Label(targets, text="max_attempt").grid(row=0, column=2, sticky=tk.W)
        if getattr(app, "sw_max_attempt", None) is None:
            app.sw_max_attempt = tk.StringVar(value="100")
        ttk.Entry(targets, textvariable=app.sw_max_attempt, width=8).grid(row=0, column=3, padx=4)

    ttk.Label(targets, text="settle").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    if getattr(app, "sw_settle_s", None) is None:
        app.sw_settle_s = tk.StringVar(value="0.02")
    ttk.Entry(targets, textvariable=app.sw_settle_s, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))
    ttk.Label(targets, text="s").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))

    if show_debug:
        ttk.Label(targets, text="update interval").grid(row=1, column=3, sticky=tk.W, pady=(6, 0))
        if getattr(app, "sw_update_interval", None) is None:
            app.sw_update_interval = tk.StringVar(value="1.0")
        ttk.Entry(targets, textvariable=app.sw_update_interval, width=8).grid(row=1, column=4, padx=4, pady=(6, 0))
        ttk.Label(targets, text="s").grid(row=1, column=5, sticky=tk.W, pady=(6, 0))

    seq = ttk.LabelFrame(row, text="Sequence (from Setup)")
    seq.grid(row=2, column=0, sticky=tk.W + tk.E, padx=(0, 12), pady=(8, 0))
    ttk.Label(seq, text="JSON").grid(row=0, column=0, sticky=tk.W)
    ttk.Label(seq, textvariable=app.sw_seq_path).grid(row=0, column=1, sticky=tk.W, padx=4)

    snapshot = ttk.LabelFrame(row, text="Setup snapshot")
    snapshot.grid(row=3, column=0, sticky=tk.W + tk.E, padx=(0, 12), pady=(8, 0))

    ttk.Label(snapshot, text="DAQ mode").grid(row=0, column=0, sticky=tk.W)
    app.sw_daq_mode = app.device_mode_var
    ttk.Label(snapshot, textvariable=app.sw_daq_mode).grid(row=0, column=1, sticky=tk.W, padx=4)

    ttk.Label(snapshot, text="Camera mode").grid(row=0, column=2, sticky=tk.W)
    app.sw_cam_mode = app.camera_mode_top_var
    ttk.Label(snapshot, textvariable=app.sw_cam_mode).grid(row=0, column=3, sticky=tk.W, padx=4)

    ttk.Label(snapshot, text="DAQ device").grid(row=0, column=4, sticky=tk.W)
    app.sw_device = app.device_var
    ttk.Label(snapshot, textvariable=app.sw_device).grid(row=0, column=5, sticky=tk.W, padx=4)

    ttk.Label(snapshot, text="FG VISA").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_visa = app.fg_resource_var
    ttk.Label(snapshot, textvariable=app.sw_visa).grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(6, 0))
    ttk.Checkbutton(snapshot, text="No FG", variable=app.sw_no_fg, state="disabled").grid(
        row=1, column=4, columnspan=2, sticky=tk.W, pady=(6, 0)
    )

    ttk.Label(snapshot, text="FG amp (mVpp)").grid(row=1, column=6, sticky=tk.W, pady=(6, 0))
    app.sw_fg_amp_mvpp = app.fg_amp_mvpp_var
    ttk.Label(snapshot, textvariable=app.sw_fg_amp_mvpp).grid(row=1, column=7, sticky=tk.W, padx=4, pady=(6, 0))

    btn_row = ttk.Frame(app.sweep_tab)
    btn_row.pack(fill=tk.X, pady=(8, 8))
    app.sw_roi_btn = ttk.Button(btn_row, text="1) ROI check", command=roi_check_cb)
    app.sw_roi_btn.pack(side=tk.LEFT, padx=4)

    app.sw_thr_btn = ttk.Button(btn_row, text="2) Threshold", command=threshold_check_cb, state=tk.DISABLED)
    app.sw_thr_btn.pack(side=tk.LEFT, padx=4)

    app.sw_start_btn = ttk.Button(btn_row, text="3) Start spectrum", command=start_sweep_cb, state=tk.DISABLED)
    app.sw_start_btn.pack(side=tk.LEFT, padx=4)
    app.sw_stop_btn = ttk.Button(btn_row, text="Stop", command=stop_sweep_cb, state=tk.DISABLED)
    app.sw_stop_btn.pack(side=tk.LEFT, padx=4)
    app.sw_status = tk.StringVar(value="Idle")
    ttk.Label(btn_row, textvariable=app.sw_status).pack(side=tk.LEFT, padx=12)

    thr_opts = ttk.LabelFrame(app.sweep_tab, text="Threshold options")
    thr_opts.pack(fill=tk.X, pady=(0, 8))
    app.sw_thr_save_frames_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        thr_opts,
        text="Save threshold frames (.npy)",
        variable=app.sw_thr_save_frames_var,
    ).pack(anchor=tk.W, padx=6, pady=4)

    thr_override = ttk.LabelFrame(app.sweep_tab, text="Threshold override")
    thr_override.pack(fill=tk.X, pady=(0, 8))
    ttk.Label(thr_override, text="tau (roi_mean)").grid(row=0, column=0, sticky=tk.W)
    app.sw_thr_tau_var = tk.StringVar(value="")
    ttk.Entry(thr_override, textvariable=app.sw_thr_tau_var, width=12).grid(row=0, column=1, padx=4)
    app.sw_thr_replot_btn = ttk.Button(
        thr_override,
        text="Replot",
        command=threshold_override_replot_cb,
        state=tk.DISABLED,
    )
    app.sw_thr_replot_btn.grid(row=0, column=2, padx=4)
    app.sw_thr_apply_btn = ttk.Button(
        thr_override,
        text="Apply",
        command=threshold_override_apply_cb,
        state=tk.DISABLED,
    )
    app.sw_thr_apply_btn.grid(row=0, column=3, padx=4)

    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        app.sw_fig = Figure(figsize=(7.5, 3.2), dpi=100)
        app.sw_ax = app.sw_fig.add_subplot(111)
        app.sw_canvas = FigureCanvasTkAgg(app.sw_fig, master=app.sweep_tab)
        app.sw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    except Exception:
        app.sw_fig = None
        app.sw_ax = None
        app.sw_canvas = None
        ttk.Label(app.sweep_tab, text="matplotlib not available; real-time plot disabled").pack()

    app._sweep_state = SweepState()
    app._sweep_show_input_error_cb = lambda msg: messagebox.showerror("Sweep", msg)
    app._sweep_events = SweepEvents(
        on_status=app.sw_status.set,
        on_warning=lambda msg: messagebox.showwarning("FG", msg),
        on_error=lambda title, msg: messagebox.showerror(title, msg),
        on_input_error=app._sweep_show_input_error_cb,
        on_plot_reset=lambda: reset_spectrum_plot_ui(app),
        on_plot_update=lambda step_idx, freq, processed, n_bright: update_spectrum_plot_ui(
            app,
            step_idx,
            freq,
            processed,
            n_bright,
        ),
        on_state_change=lambda prev, next_state: append_state_history(
            app,
            prev=getattr(prev, "value", str(prev)),
            next_state=getattr(next_state, "value", str(next_state)),
        ),
    )
    app._sweep_ctrl = SweepController(
        events=app._sweep_events,
        io=SweepIO(
            toggle_controls=lambda enable: toggle_sweep_controls(app, enable),
            refresh_buttons=lambda: refresh_sweep_buttons(app),
            cleanup_stale_workers=lambda: cleanup_stale_workers(app.worker_pids_path),
            apply_subarray=lambda cfg: apply_subarray_to_cam_cfg(app, cfg),
            write_last_worker_pids_cb=lambda data: write_last_worker_pids(app.worker_pids_path, data),
            format_worker_failure=format_worker_failure,
            confirm_threshold=lambda th, acc, tau: messagebox.askyesno(
                "Threshold",
                format_threshold_prompt(th, acc, tau),
                parent=app,
            ),
            update_threshold_ui=lambda tau, tau_on, tau_off: app.sw_thr_tau_var.set(f"{float(tau):.3g}"),
            get_threshold_save_frames=lambda: bool(app.sw_thr_save_frames_var.get()),
            join_with_ui=lambda proc, timeout: join_with_ui(app, proc, timeout=timeout),
            set_last_error_cb=lambda label, message, log_path: set_last_error(
                app,
                label=label,
                message=message,
                log_path=log_path,
            ),
        ),
        deps=SweepDeps(
            write_sweep_config_json=write_sweep_config_json,
            SweepPersistedConfig=SweepPersistedConfig,
            create_sweep_workers=create_sweep_workers,
            bootstrap_workers_for_sweep=bootstrap_workers_for_sweep,
            build_sweep_session_dict=build_sweep_session_dict,
            run_roi_bootstrap_stage=run_roi_bootstrap_stage,
            stop_sweep_workers=stop_sweep_workers,
            mpq_get_with_ui=lambda q, timeout, label: mpq_get_with_ui(app, q, timeout=timeout, label=label),
            ui_pump=lambda: ui_pump(app),
            AO_RATE_HZ=ao_rate_hz,
            NM_397=nm_397,
            CAMERA_TRIGGER=camera_trigger,
            ROI_PULSE_S=roi_pulse_s,
            ROI_IDLE_S=roi_idle_s,
            ROI_MAX_ATTEMPT=roi_max_attempt,
            log_dir=getattr(getattr(app, "_log_ctx", None), "log_dir", None),
            run_id=getattr(getattr(app, "_log_ctx", None), "run_id", None),
            output_root=getattr(app, "output_root", None),
        ),
    )
