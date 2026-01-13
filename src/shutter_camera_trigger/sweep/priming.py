from __future__ import annotations

import time
from typing import Any, Callable


def prime_until_camera_ready(
    *,
    cam_resp_q: Any,
    daq_send: Callable[[dict], None],
    daq_recv: Callable[[float, str], dict],
    prime_cmd: dict,
    deadline_s: float,
    ui_pump: Callable[[], None] | None = None,
    status_cb: Callable[[str], None] | None = None,
    sleep_s: float = 0.01,
) -> dict[str, Any] | None:
    """Prime an external-trigger camera until it reports ready, or timeout.

    This is used during camera bootstrap: the camera worker may wait for external
    triggers before sending its initial "ready" message.

    Returns:
        - camera ready dict if received
        - None if deadline is reached without ready
    """

    deadline_t = time.time() + float(deadline_s)

    while time.time() < deadline_t:
        try:
            cam_ready = cam_resp_q.get_nowait()
            if isinstance(cam_ready, dict):
                return cam_ready
        except Exception:
            pass

        try:
            daq_send(dict(prime_cmd))
            _ = daq_recv(5.0, "DAQ prime response")
        except Exception:
            time.sleep(0.05)

        if ui_pump is not None:
            try:
                ui_pump()
            except Exception:
                pass

        time.sleep(float(sleep_s))

    if status_cb is not None:
        try:
            status_cb("Camera priming timed out")
        except Exception:
            pass

    return None
