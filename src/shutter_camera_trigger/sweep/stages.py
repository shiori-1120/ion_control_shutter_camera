from __future__ import annotations

from typing import Any, Callable

from .roi_bootstrap import run_roi_bootstrap


def run_roi_bootstrap_stage(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    max_attempt: int,
    status_cb: Callable[[str], None] | None = None,
    ui_pump: Callable[[], None] | None = None,
) -> bool:
    """Run the ROI bootstrap stage (status update + optional UI pump + bootstrap loop)."""

    if status_cb is not None:
        try:
            status_cb("ROI bootstrap...")
        except Exception:
            pass

    if ui_pump is not None:
        try:
            ui_pump()
        except Exception:
            pass

    return run_roi_bootstrap(
        daq_cmd_q=daq_cmd_q,
        daq_resp_q=daq_resp_q,
        cam_cmd_q=cam_cmd_q,
        cam_resp_q=cam_resp_q,
        nm_397=nm_397,
        camera_trigger=camera_trigger,
        roi_pulse_s=roi_pulse_s,
        roi_idle_s=roi_idle_s,
        max_attempt=max_attempt,
        status_cb=status_cb,
    )
