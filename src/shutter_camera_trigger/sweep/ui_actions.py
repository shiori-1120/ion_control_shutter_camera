from __future__ import annotations

from typing import Any

from ..gui_support.validators import parse_fg_amp_vpp_safe
from .input import collect_sweep_input
from .model import SweepPhase


def prepare_session(app: Any, *, default_daq_device: str) -> bool:
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_prepare_start")
    except Exception:
        pass
    inputs = collect_sweep_input(
        app,
        default_daq_device=default_daq_device,
        show_input_error_cb=getattr(getattr(app, "_sweep_events", None), "on_input_error", lambda msg: None),
    )
    if inputs is None:
        try:
            if getattr(app, "_logger", None):
                app._logger.error("sweep_prepare_failed")
        except Exception:
            pass
        return False
    ok = app._sweep_ctrl.prepare_session(app._sweep_state, inputs)
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_prepare_done ok=%s", ok)
            if ok:
                app._logger.info(
                    "sweep_sequence_meta camera_actions=%d sync_markers=%d",
                    len(inputs.camera_actions),
                    len(inputs.sync_markers),
                )
    except Exception:
        pass
    return ok


def roi_check(app: Any, *, default_daq_device: str) -> None:
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_roi_check_start")
    except Exception:
        pass
    if not prepare_session(app, default_daq_device=default_daq_device):
        return
    if app.sw_fig is None or app.sw_canvas is None:
        return
    app._sweep_ctrl.roi_check(app._sweep_state, fig=app.sw_fig, canvas=app.sw_canvas)
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_roi_check_done")
    except Exception:
        pass


def threshold_check(app: Any) -> None:
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_threshold_start")
    except Exception:
        pass
    if app.sw_fig is None or app.sw_canvas is None:
        return
    app._sweep_ctrl.threshold_check(app._sweep_state, fig=app.sw_fig, canvas=app.sw_canvas)
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_threshold_done")
    except Exception:
        pass


def start_sweep(
    app: Any,
    *,
    default_daq_device: str,
    fg_amp_max_mvpp: float,
    default_fg_amp_vpp: float,
) -> None:
    state = app._sweep_state
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_start")
    except Exception:
        pass
    if state.phase is SweepPhase.IDLE:
        if not prepare_session(app, default_daq_device=default_daq_device):
            return

    app._sweep_ctrl.start_sweep(
        state,
        fig=app.sw_fig,
        canvas=app.sw_canvas,
        fg_connected=app._fg_connected,
        fg_handle=app._fg_handle,
        fallback_fg_amp_vpp=parse_fg_amp_vpp_safe(
            app,
            max_mvpp=fg_amp_max_mvpp,
            default_vpp=default_fg_amp_vpp,
        ),
    )
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_start_done")
    except Exception:
        pass


def stop_sweep(app: Any, *, clean_only: bool = False) -> None:
    try:
        if getattr(app, "_logger", None):
            app._logger.info("sweep_stop clean_only=%s", clean_only)
    except Exception:
        pass
    app._sweep_ctrl.stop_sweep(app._sweep_state, clean_only=clean_only, fig=app.sw_fig)
