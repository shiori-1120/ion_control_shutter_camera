from __future__ import annotations

from typing import Any, Callable
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

from ..gui_support.perf import limit_blas_threads
from ..gui_support.validators import apply_subarray_to_cam_cfg
from ..gui_support.worker_cleanup import cleanup_stale_workers, write_last_worker_pids
from ..gui_support.worker_messages import format_worker_failure
from .controller import SweepController, SweepDeps, SweepState, SweepUi
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
    pick_seq_json_cb: Callable[[], None],
    roi_check_cb: Callable[[], None],
    threshold_check_cb: Callable[[], None],
    start_sweep_cb: Callable[[], None],
    stop_sweep_cb: Callable[[], None],
) -> None:
    limit_blas_threads()

    row = ttk.Frame(app.sweep_tab)
    row.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(row, text="Freq start (Hz)").grid(row=0, column=0, sticky=tk.W)
    app.sw_freq_start = tk.StringVar(value="199e6")
    ttk.Entry(row, textvariable=app.sw_freq_start, width=12).grid(row=0, column=1, padx=4)

    ttk.Label(row, text="Freq stop (Hz)").grid(row=0, column=2, sticky=tk.W)
    app.sw_freq_stop = tk.StringVar(value="201e6")
    ttk.Entry(row, textvariable=app.sw_freq_stop, width=12).grid(row=0, column=3, padx=4)

    ttk.Label(row, text="Freq step (Hz)").grid(row=0, column=4, sticky=tk.W)
    app.sw_freq_step = tk.StringVar(value="0.5e6")
    ttk.Entry(row, textvariable=app.sw_freq_step, width=12).grid(row=0, column=5, padx=4)

    ttk.Label(row, text="n_target").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_n_target = tk.StringVar(value="50")
    ttk.Entry(row, textvariable=app.sw_n_target, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))

    ttk.Label(row, text="max_attempt").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))
    app.sw_max_attempt = tk.StringVar(value="100")
    ttk.Entry(row, textvariable=app.sw_max_attempt, width=8).grid(row=1, column=3, padx=4, pady=(6, 0))

    ttk.Label(row, text="settle_s").grid(row=1, column=4, sticky=tk.W, pady=(6, 0))
    app.sw_settle_s = tk.StringVar(value="0.02")
    ttk.Entry(row, textvariable=app.sw_settle_s, width=8).grid(row=1, column=5, padx=4, pady=(6, 0))

    ttk.Label(row, text="Sequence JSON").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_seq_path = tk.StringVar(value="src/shutter_camera_trigger/sequence_examples/minimal_sequence.json")
    ttk.Entry(row, textvariable=app.sw_seq_path, width=48).grid(
        row=2, column=1, columnspan=4, sticky=tk.W, padx=4, pady=(6, 0)
    )
    ttk.Button(row, text="...", width=3, command=pick_seq_json_cb).grid(row=2, column=5, pady=(6, 0))

    ttk.Label(row, text="DAQ mode").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_daq_mode = tk.StringVar(value="dry")
    ttk.Combobox(row, textvariable=app.sw_daq_mode, values=["dry", "real"], width=6, state="readonly").grid(
        row=3, column=1, padx=4, pady=(6, 0)
    )

    ttk.Label(row, text="Camera mode").grid(row=3, column=2, sticky=tk.W, pady=(6, 0))
    app.sw_cam_mode = app.camera_mode_top_var
    ttk.Combobox(row, textvariable=app.sw_cam_mode, values=["dry", "real"], width=6, state="readonly").grid(
        row=3, column=3, padx=4, pady=(6, 0)
    )

    ttk.Label(row, text="DAQ device").grid(row=3, column=4, sticky=tk.W, pady=(6, 0))
    app.sw_device = tk.StringVar(value=default_daq_device)
    ttk.Entry(row, textvariable=app.sw_device, width=10).grid(row=3, column=5, padx=4, pady=(6, 0))

    ttk.Label(row, text="FG VISA").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_visa = app.fg_resource_var
    ttk.Entry(row, textvariable=app.sw_visa, width=32).grid(
        row=4, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(6, 0)
    )
    app.sw_no_fg = tk.BooleanVar(value=True)
    ttk.Checkbutton(row, text="No FG", variable=app.sw_no_fg).grid(
        row=4, column=4, columnspan=2, sticky=tk.W, pady=(6, 0)
    )

    ttk.Label(row, text="FG amp (mVpp)").grid(row=5, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_fg_amp_mvpp = app.fg_amp_mvpp_var
    ttk.Entry(row, textvariable=app.sw_fg_amp_mvpp, width=10).grid(row=5, column=1, padx=4, pady=(6, 0))

    ttk.Label(row, text="Update interval (s)").grid(row=6, column=0, sticky=tk.W, pady=(6, 0))
    app.sw_update_interval = tk.StringVar(value="1.0")
    ttk.Entry(row, textvariable=app.sw_update_interval, width=8).grid(row=6, column=1, padx=4, pady=(6, 0))

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
    app._sweep_ctrl = SweepController(
        ui=SweepUi(
            status_cb=app.sw_status.set,
            messagebox=messagebox,
            ui_pump=lambda: ui_pump(app),
            mpq_get_with_ui=lambda q, timeout, label: mpq_get_with_ui(app, q, timeout=timeout, label=label),
            toggle_controls=lambda enable: toggle_sweep_controls(app, enable),
            refresh_buttons=lambda: refresh_sweep_buttons(app),
            cleanup_stale_workers=lambda: cleanup_stale_workers(app.worker_pids_path),
            apply_subarray_cb=lambda cfg: apply_subarray_to_cam_cfg(app, cfg),
            write_last_worker_pids_cb=lambda data: write_last_worker_pids(app.worker_pids_path, data),
            format_worker_failure=format_worker_failure,
            confirm_threshold_cb=lambda th, acc, tau: messagebox.askyesno(
                "Threshold",
                format_threshold_prompt(th, acc, tau),
                parent=app,
            ),
            warn_cb=lambda msg: messagebox.showwarning("FG", msg),
            reset_plot_cb=lambda: reset_spectrum_plot_ui(app),
            update_plot_cb=lambda step_idx, freq, processed, n_bright: update_spectrum_plot_ui(
                app,
                step_idx,
                freq,
                processed,
                n_bright,
            ),
            join_with_ui=lambda proc, timeout: join_with_ui(app, proc, timeout=timeout),
        ),
        deps=SweepDeps(
            write_sweep_config_json=write_sweep_config_json,
            SweepPersistedConfig=SweepPersistedConfig,
            create_sweep_workers=create_sweep_workers,
            bootstrap_workers_for_sweep=bootstrap_workers_for_sweep,
            build_sweep_session_dict=build_sweep_session_dict,
            run_roi_bootstrap_stage=run_roi_bootstrap_stage,
            stop_sweep_workers=stop_sweep_workers,
            AO_RATE_HZ=ao_rate_hz,
            NM_397=nm_397,
            CAMERA_TRIGGER=camera_trigger,
            ROI_PULSE_S=roi_pulse_s,
            ROI_IDLE_S=roi_idle_s,
            ROI_MAX_ATTEMPT=roi_max_attempt,
            log_dir=getattr(getattr(app, "_log_ctx", None), "log_dir", None),
            run_id=getattr(getattr(app, "_log_ctx", None), "run_id", None),
        ),
    )
