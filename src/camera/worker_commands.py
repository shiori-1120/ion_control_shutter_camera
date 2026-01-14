from __future__ import annotations

import traceback
from typing import Any

from .worker_utils import as_roi_tuple


def handle_roi_threshold_cmd(
    name: str,
    cmd: dict[str, Any],
    *,
    roi_t: tuple[int, int, int, int] | None,
    bg_roi_t: tuple[int, int, int, int] | None,
    tau_on: float | None,
    tau_off: float | None,
    prev_state: bool | None,
    log,
) -> tuple[bool, dict[str, Any] | None, tuple[Any, Any, Any, Any, Any]]:
    if name == "set_roi":
        try:
            roi_new = as_roi_tuple(cmd.get("roi"))
            if roi_new is None:
                raise ValueError("set_roi requires roi=[xw,yw,xs,ys]")
            roi_t = roi_new
            prev_state = None
            resp = {"ok": True, "event": "roi", "roi": list(roi_t)}
            log(f"set_roi {roi_t}")
        except Exception as e:
            log(f"set_roi error {type(e).__name__}: {e}")
            resp = {"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)}
        return True, resp, (roi_t, bg_roi_t, tau_on, tau_off, prev_state)

    if name == "set_bg_roi":
        try:
            bg_new = as_roi_tuple(cmd.get("bg_roi"))
            bg_roi_t = bg_new
            prev_state = None
            resp = {"ok": True, "event": "bg_roi", "bg_roi": (list(bg_roi_t) if bg_roi_t else None)}
            log(f"set_bg_roi {bg_roi_t}")
        except Exception as e:
            log(f"set_bg_roi error {type(e).__name__}: {e}")
            resp = {"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)}
        return True, resp, (roi_t, bg_roi_t, tau_on, tau_off, prev_state)

    if name == "set_threshold":
        try:
            tau_on_new = cmd.get("tau_on")
            tau_off_new = cmd.get("tau_off")
            tau_new = cmd.get("tau")

            if tau_new is not None:
                tau = float(tau_new)
                tau_on_new = float(tau)
                tau_off_new = float(tau)

            if tau_on_new is None or tau_off_new is None:
                raise ValueError("set_threshold requires tau or (tau_on and tau_off)")

            tau_on = float(tau_on_new)
            tau_off = float(tau_off_new)
            prev_state = None

            resp = {"ok": True, "event": "threshold", "tau_on": float(tau_on), "tau_off": float(tau_off)}
            log(f"set_threshold tau_on={tau_on} tau_off={tau_off}")
        except Exception as e:
            log(f"set_threshold error {type(e).__name__}: {e}")
            resp = {"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)}
        return True, resp, (roi_t, bg_roi_t, tau_on, tau_off, prev_state)

    return False, None, (roi_t, bg_roi_t, tau_on, tau_off, prev_state)
