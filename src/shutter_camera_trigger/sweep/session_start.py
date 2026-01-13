from __future__ import annotations

from typing import Any, Callable

from .priming import prime_until_camera_ready


def bootstrap_workers_for_sweep(
    *,
    daq_resp_q: Any,
    cam_proc: Any,
    cam_resp_q: Any,
    mpq_get_with_ui: Callable[[Any, float, str], dict],
    format_worker_failure: Callable[..., str],
    cam_log_path: str | None,
    cam_mode: str,
    trig_src: str,
    prime_cmd: dict,
    daq_send: Callable[[dict], None],
    daq_recv: Callable[[float, str], dict],
    ui_pump: Callable[[], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
    daq_ready_timeout_s: float = 5.0,
    cam_ready_timeout_s: float = 30.0,
    prime_deadline_s: float = 30.0,
) -> dict:
    """Wait DAQ ready, start camera process, optionally prime, then wait camera ready.

    Returns camera ready dict on success.
    Raises RuntimeError on worker failures.

    This function is GUI-framework-agnostic: it relies on injected callbacks.
    """

    # wait DAQ ready first
    daq_ready = mpq_get_with_ui(daq_resp_q, float(daq_ready_timeout_s), "DAQ ready")
    if not isinstance(daq_ready, dict) or not daq_ready.get("ok"):
        raise RuntimeError(f"DAQ worker failed: {daq_ready}")

    # start camera after DAQ is ready (preserve existing ordering)
    try:
        cam_proc.start()
    except Exception as e:
        raise RuntimeError(f"Failed to start camera worker: {e}")

    # prime external-trigger camera during bootstrap
    cam_ready: dict[str, Any] | None = None
    trig_src_u = str(trig_src or "EXTERNAL").strip().upper() or "EXTERNAL"
    cam_mode_l = str(cam_mode or "dry").strip().lower() or "dry"

    if cam_mode_l == "real" and trig_src_u in ("EXTERNAL", "EXT", "2", ""):
        if status_cb is not None:
            try:
                status_cb("Camera priming...")
            except Exception:
                pass

        cam_ready = prime_until_camera_ready(
            cam_resp_q=cam_resp_q,
            daq_send=daq_send,
            daq_recv=daq_recv,
            prime_cmd=prime_cmd,
            deadline_s=float(prime_deadline_s),
            ui_pump=ui_pump,
            status_cb=status_cb,
            sleep_s=0.01,
        )

    # wait camera ready
    if cam_ready is None:
        cam_ready = mpq_get_with_ui(cam_resp_q, float(cam_ready_timeout_s), "Camera ready")

    if not isinstance(cam_ready, dict) or not cam_ready.get("ok"):
        label = "Camera worker init failed"
        msg = format_worker_failure(
            cam_ready,
            label=label,
            log_path=(str(cam_log_path) if cam_log_path else None),
        )
        raise RuntimeError(msg)

    return cam_ready
