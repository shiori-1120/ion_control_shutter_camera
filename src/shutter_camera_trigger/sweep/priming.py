from __future__ import annotations

import time
from typing import Any, Callable

from ..hardware import DaqQueueDevice, DaqSequenceCommand


def prime_until_camera_ready(
    *,
    cam_resp_q: Any,
    daq_send: Callable[[dict], None],
    daq_recv: Callable[[float, str], dict],
    prime_cmd: DaqSequenceCommand | dict,
    daq_device: DaqQueueDevice | None = None,
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
            if isinstance(prime_cmd, DaqSequenceCommand):
                seq_cmd = prime_cmd
            else:
                seq_cmd = DaqSequenceCommand(
                    do_sequence=list(prime_cmd.get("do_sequence") or []),
                    ao_insert_index=int(prime_cmd.get("insert_index", -1)),
                    ao_width_ms=float(prime_cmd.get("ao_width_ms", 0.0)),
                    ao_rate_hz=float(prime_cmd.get("ao_rate_hz", 5000.0)),
                    ao_v_high=float(prime_cmd.get("ao_v_high", 5.0)),
                    ao_v_low=float(prime_cmd.get("ao_v_low", 0.0)),
                )
            if daq_device is not None:
                daq_device.run_sequence_once(seq_cmd)
            else:
                daq_send(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": list(seq_cmd.do_sequence),
                        "insert_index": int(seq_cmd.ao_insert_index),
                        "ao_width_ms": float(seq_cmd.ao_width_ms),
                        "ao_rate_hz": float(seq_cmd.ao_rate_hz),
                        "ao_v_high": float(seq_cmd.ao_v_high),
                        "ao_v_low": float(seq_cmd.ao_v_low),
                    }
                )
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
