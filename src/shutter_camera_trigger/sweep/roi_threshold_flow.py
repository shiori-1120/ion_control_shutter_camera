from __future__ import annotations

from typing import Any, Callable

from ..hardware import CameraQueueDevice

from .stages import RoiCheckResult, ThresholdStageResult, run_roi_check_stage, run_threshold_stage


def run_roi_check_flow(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    pulse_seq: list[tuple[int, float]],
    ao_rate_hz: float,
    out_dir: Any,
    cam_log_path: str | None,
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]],
    ui_pump: Callable[[], None] | None,
    status_cb: Callable[[str], None] | None,
    fig: Any,
    canvas: Any,
    session: dict[str, Any] | None,
    prefer_sample_path: str | None = None,
) -> RoiCheckResult:
    r: RoiCheckResult = run_roi_check_stage(
        daq_cmd_q=daq_cmd_q,
        daq_resp_q=daq_resp_q,
        cam_cmd_q=cam_cmd_q,
        cam_resp_q=cam_resp_q,
        pulse_seq=pulse_seq,
        ao_rate_hz=ao_rate_hz,
        out_dir=out_dir,
        cam_log_path=cam_log_path,
        mpq_get_with_ui=mpq_get_with_ui,
        ui_pump=ui_pump,
        status_cb=status_cb,
        fig=fig,
        canvas=canvas,
        prefer_sample_path=prefer_sample_path,
    )

    roi = r.roi
    if session is not None:
        session["roi"] = roi

    # Propagate ROI to camera worker so get_state uses the same ROI scalar as Step 2.
    try:
        CameraQueueDevice(cmd_q=cam_cmd_q).set_roi(list(roi) if roi is not None else None)
        _ = mpq_get_with_ui(cam_resp_q, timeout=5, label="Camera set_roi")
    except Exception:
        pass

    return r


def format_threshold_prompt(threshold: dict[str, Any], agreement: float, tau: float) -> str:
    return (
        "Apply threshold?\n"
        f"mode={threshold.get('mode')}\n"
        f"agreement={agreement * 100:.1f}% (hysteresis OFF)\n\n"
        " metric=roi_mean\n"
        f" tau={tau:.3g}"
    )


def run_threshold_flow(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    do_sequence: list[tuple[int, float]],
    roi: list[int],
    n_target: int,
    max_attempt: int,
    cam_exposure_s: float,
    ao_rate_hz: float,
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]],
    ui_pump: Callable[[], None] | None,
    status_cb: Callable[[str], None] | None,
    fig: Any,
    canvas: Any,
    out_dir: Any,
    confirm_apply_cb: Callable[[dict[str, Any], float, float], bool],
) -> tuple[ThresholdStageResult, bool]:
    r: ThresholdStageResult = run_threshold_stage(
        daq_cmd_q=daq_cmd_q,
        daq_resp_q=daq_resp_q,
        cam_cmd_q=cam_cmd_q,
        cam_resp_q=cam_resp_q,
        do_sequence=do_sequence,
        roi=[int(v) for v in roi],
        n_target=int(n_target),
        max_attempt=int(max_attempt),
        cam_exposure_s=float(cam_exposure_s),
        ao_rate_hz=ao_rate_hz,
        mpq_get_with_ui=mpq_get_with_ui,
        ui_pump=ui_pump,
        status_cb=status_cb,
        fig=fig,
        canvas=canvas,
        out_dir=out_dir,
    )

    tau = float(r.tau)
    tau_on = float(r.tau_on)
    tau_off = float(r.tau_off)
    agreement = float(r.agreement)
    threshold = dict(r.threshold or {})

    apply_ok = bool(confirm_apply_cb(threshold, agreement, tau))
    if apply_ok:
        CameraQueueDevice(cmd_q=cam_cmd_q).set_threshold(float(tau_on), float(tau_off))
        ack = mpq_get_with_ui(cam_resp_q, timeout=5, label="Camera set_threshold")
        if not ack.get("ok"):
            raise RuntimeError(f"set_threshold failed: {ack}")

    return r, apply_ok
