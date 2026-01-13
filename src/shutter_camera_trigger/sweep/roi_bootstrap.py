from __future__ import annotations

import time
from typing import Any, Callable


def run_roi_bootstrap(
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
) -> bool:
    """Send simple TTL pulses (camera trigger only) until camera replies.

    Returns True on success, False on exhaustion.

    Notes:
    - Keeps 397 ON during bootstrap so ions remain cooled/visible.
    - Uses only Queue put/get; no GUI dependencies.
    """

    roi_sequence = [
        (nm_397, roi_idle_s),
        (nm_397 | camera_trigger, roi_pulse_s),
        (nm_397, roi_idle_s),
    ]

    success = 0
    last_err: str | None = None

    for _attempt in range(int(max_attempt)):
        try:
            daq_cmd_q.put(
                {
                    "cmd": "run_sequence_once",
                    "do_sequence": roi_sequence,
                    "insert_index": -1,
                    "ao_width_ms": 0.0,
                    "ao_rate_hz": 5000.0,
                    "ao_v_high": 5.0,
                    "ao_v_low": 0.0,
                }
            )
            daq_resp = daq_resp_q.get(timeout=5)
            if not isinstance(daq_resp, dict) or not daq_resp.get("ok"):
                last_err = f"DAQ: {daq_resp}"
                continue

            cam_cmd_q.put({"cmd": "get_state", "timeout_s": 1.0})
            cam_resp = cam_resp_q.get(timeout=5)
            if not isinstance(cam_resp, dict) or not cam_resp.get("ok"):
                last_err = f"Camera: {cam_resp}"
                continue

            success += 1
            if success >= 1:
                return True
        except Exception as e:
            last_err = str(e)

        time.sleep(max(0.0, float(roi_idle_s)))

    if last_err and status_cb is not None:
        try:
            status_cb(f"ROI bootstrap failed: {last_err}")
        except Exception:
            pass

    return False
