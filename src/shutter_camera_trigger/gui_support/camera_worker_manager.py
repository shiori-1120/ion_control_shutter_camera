from __future__ import annotations

import time
from typing import Any, Callable

from ..gui_support.validators import (
    apply_subarray_to_cam_cfg,
    parse_camera_trigger_cfg,
    parse_exposure_s_safe,
)
from ..workers.camera_worker_process import start_camera_worker_process, stop_worker_process


def _normalize_cam_cfg(cfg: dict[str, Any]) -> tuple:
    return (
        str(cfg.get("mode") or ""),
        float(cfg.get("exposure_s") or 0.0),
        float(cfg.get("frame_timeout_s") or 0.0),
        int(cfg.get("bootstrap_n") or 0),
        bool(cfg.get("diagnostics_mode") or False),
        tuple(cfg.get("subarray") or []),
        tuple(sorted((cfg.get("trigger") or {}).items())),
    )


def build_cam_cfg(app: Any, *, log_path: str | None = None, run_id: str | None = None) -> dict[str, Any]:
    trig_cfg = dict(parse_camera_trigger_cfg(app))
    trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
    base_timeout_s = 5.0 if trig_src in ("EXTERNAL", "EXT", "2", "") else 1.0
    exposure_s = float(parse_exposure_s_safe(app))
    cam_cfg: dict[str, Any] = {
        "mode": str(app.camera_mode_top_var.get() or "dry").strip().lower(),
        "exposure_s": exposure_s,
        "frame_timeout_s": max(base_timeout_s, exposure_s * 4.0 + 0.5),
        "bootstrap_n": 10,
        "trigger": trig_cfg,
        "verbose": bool(getattr(app, "camera_verbose_var", None) and app.camera_verbose_var.get()),
    }
    if log_path:
        cam_cfg["log_path"] = str(log_path)
    if run_id:
        cam_cfg["run_id"] = str(run_id)
    apply_subarray_to_cam_cfg(app, cam_cfg)
    return cam_cfg


def ensure_camera_worker(
    app: Any,
    *,
    cam_cfg: dict[str, Any],
    ready_timeout_s: float = 30.0,
    prime_cb: Callable[[], None] | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    proc = getattr(app, "_cam_worker_proc", None)
    cmd_q = getattr(app, "_cam_worker_cmd_q", None)
    resp_q = getattr(app, "_cam_worker_resp_q", None)
    cfg_key = _normalize_cam_cfg(cam_cfg)
    prev_key = getattr(app, "_cam_worker_cfg_key", None)
    ready_flag = bool(getattr(app, "_cam_worker_ready", False))
    running = bool(proc is not None and getattr(proc, "is_alive", lambda: False)())

    if running and cmd_q is not None and resp_q is not None and prev_key == cfg_key and ready_flag:
        return proc, cmd_q, {"ok": True, "event": "ready", "mode": cam_cfg.get("mode")}

    if running and prev_key != cfg_key:
        stop_worker_process(proc=proc, cmd_q=cmd_q)
        setattr(app, "_cam_worker_proc", None)
        setattr(app, "_cam_worker_cmd_q", None)
        setattr(app, "_cam_worker_resp_q", None)
        setattr(app, "_cam_worker_ready", False)

    if not running or prev_key != cfg_key or cmd_q is None or resp_q is None:
        proc, cmd_q, resp_q = start_camera_worker_process(cfg=cam_cfg, start=True)
        setattr(app, "_cam_worker_proc", proc)
        setattr(app, "_cam_worker_cmd_q", cmd_q)
        setattr(app, "_cam_worker_resp_q", resp_q)
        setattr(app, "_cam_worker_cfg_key", cfg_key)
        setattr(app, "_cam_worker_ready", False)

    deadline = time.time() + float(ready_timeout_s)
    last_error: Any | None = None
    while time.time() < deadline:
        try:
            msg = resp_q.get_nowait()
            if isinstance(msg, dict):
                if msg.get("ok"):
                    setattr(app, "_cam_worker_ready", True)
                    return proc, cmd_q, msg
                last_error = msg
        except Exception:
            pass
        if prime_cb is not None:
            try:
                prime_cb()
            except Exception as e:
                last_error = {"ok": False, "error": str(e)}
        time.sleep(0.05)

    raise RuntimeError(f"Timeout waiting for camera ready: {last_error}")


def stop_camera_worker(app: Any) -> None:
    proc = getattr(app, "_cam_worker_proc", None)
    cmd_q = getattr(app, "_cam_worker_cmd_q", None)
    if proc is not None:
        stop_worker_process(proc=proc, cmd_q=cmd_q)
    setattr(app, "_cam_worker_proc", None)
    setattr(app, "_cam_worker_cmd_q", None)
    setattr(app, "_cam_worker_resp_q", None)
    setattr(app, "_cam_worker_cfg_key", None)
    setattr(app, "_cam_worker_ready", False)
