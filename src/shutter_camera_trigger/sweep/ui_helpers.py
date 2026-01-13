from __future__ import annotations

import queue
import time
from typing import Any

from .model import SweepPhase
from .spectrum_ui import reset_spectrum_plot, update_spectrum_plot
from ..gui_support.process_cleanup import join_with_ui as join_process_with_ui


def ui_pump(app: Any) -> None:
    """Process Tk events to avoid UI freeze during long operations."""
    try:
        app.update()
    except Exception:
        pass


def mpq_get_with_ui(
    app: Any,
    q: Any,
    timeout: float,
    *,
    label: str = "response",
    poll_s: float = 0.02,
) -> Any:
    """Queue.get(timeout=...) that keeps the Tk UI responsive."""
    deadline = time.time() + float(timeout)
    while True:
        if app._sweep_state.phase in {SweepPhase.IDLE, SweepPhase.STOPPING, SweepPhase.ERROR}:
            raise RuntimeError("Stopped")
        try:
            return q.get_nowait()
        except queue.Empty:
            if time.time() >= deadline:
                raise RuntimeError(f"Timeout waiting for {label} ({timeout:.1f}s)")
            ui_pump(app)
            time.sleep(poll_s)


def reset_spectrum_plot_ui(app: Any) -> None:
    if app.sw_fig is None or app.sw_canvas is None:
        return
    app.sw_ax = reset_spectrum_plot(app.sw_fig, app.sw_canvas)


def update_spectrum_plot_ui(app: Any, step_idx: int, freq: float, processed: int, n_bright: int) -> None:
    if app.sw_ax is None or app.sw_canvas is None:
        return
    update_spectrum_plot(
        app.sw_ax,
        app.sw_canvas,
        app._sweep_state.results,
        int(step_idx),
        float(freq),
        int(processed),
        int(n_bright),
    )


def refresh_sweep_buttons(app: Any) -> None:
    """Update Sweep tab button enabled/disabled states based on stage."""
    try:
        state = app._sweep_state
        phase = state.phase
        if phase is SweepPhase.IDLE:
            app.sw_stop_btn.configure(state="disabled")
            app.sw_roi_btn.configure(state="normal")
            app.sw_thr_btn.configure(state="disabled")
            app.sw_start_btn.configure(state="disabled")
        elif phase in {SweepPhase.PREPARED, SweepPhase.ROI_DONE, SweepPhase.THRESHOLD_DONE}:
            app.sw_stop_btn.configure(state="normal")
            app.sw_roi_btn.configure(state="normal")
            app.sw_thr_btn.configure(state="normal")
            app.sw_start_btn.configure(state=("normal" if phase is SweepPhase.THRESHOLD_DONE else "disabled"))
        elif phase is SweepPhase.RUNNING:
            app.sw_stop_btn.configure(state="normal")
            app.sw_roi_btn.configure(state="disabled")
            app.sw_thr_btn.configure(state="disabled")
            app.sw_start_btn.configure(state="disabled")
        else:
            app.sw_stop_btn.configure(state="disabled")
            app.sw_roi_btn.configure(state="disabled")
            app.sw_thr_btn.configure(state="disabled")
            app.sw_start_btn.configure(state="disabled")
    except Exception:
        pass


def toggle_sweep_controls(app: Any, enable: bool) -> None:
    if enable:
        try:
            app.sw_roi_btn.configure(state="normal")
            app.sw_thr_btn.configure(state="disabled")
            app.sw_start_btn.configure(state="disabled")
            app.sw_stop_btn.configure(state="disabled")
        except Exception:
            pass
    else:
        try:
            app.sw_roi_btn.configure(state="disabled")
            app.sw_thr_btn.configure(state="disabled")
            app.sw_start_btn.configure(state="disabled")
            app.sw_stop_btn.configure(state="normal")
        except Exception:
            pass

    child_state = "!disabled" if enable else "disabled"
    for child in app.sweep_tab.winfo_children():
        if child is app.sw_stop_btn and not enable:
            continue
        try:
            child.state([child_state])
        except Exception:
            try:
                child.configure(state=("normal" if enable else "disabled"))
            except Exception:
                pass


def join_with_ui(app: Any, proc: Any, *, timeout: float, poll_s: float = 0.02) -> None:
    join_process_with_ui(app, proc, timeout=timeout, poll_s=poll_s)
