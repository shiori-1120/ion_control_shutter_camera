"""Dry DAQ worker (no NI-DAQmx required).

This worker matches the same multiprocessing.Queue protocol as the real DAQ
worker used by the spectrum runner, but performs no hardware I/O.

Purpose:
- allow end-to-end bring-up of the runner/IPC/logging without a connected DAQ
- keep the real worker implementation free to import nidaqmx/numpy

Protocol:
- cmd_q: dict commands
  - {"cmd":"set_do","value":int}
  - {"cmd":"run_sequence_once","do_sequence":[(value,hold_s),...],"insert_index":int,
     "ao_width_ms":float,"ao_rate_hz":float,"ao_v_high":float,"ao_v_low":float}
  - {"cmd":"close"}
- resp_q: dict responses

Notes:
- We optionally sleep for the requested hold times so timing looks realistic.
"""

from __future__ import annotations

import queue
import time
import traceback
from multiprocessing.queues import Queue
from typing import Any


def daq_worker_dry_main(cmd_q: Queue, resp_q: Queue, cfg: dict[str, Any]) -> None:
    def send(msg: dict[str, Any]) -> None:
        try:
            resp_q.put(msg)
        except Exception:
            pass

    device = str(cfg.get("device") or "(dry)")
    send({"ok": True, "event": "ready", "mode": "dry", "device": device})

    try:
        while True:
            try:
                cmd = cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not isinstance(cmd, dict):
                continue

            name = cmd.get("cmd")
            if name in ("quit", "close"):
                send({"ok": True, "event": "closing"})
                return

            try:
                if name == "set_do":
                    # no-op
                    send({"ok": True, "event": "set_do"})

                elif name == "run_sequence_once":
                    do_sequence = cmd.get("do_sequence")
                    if not isinstance(do_sequence, list) or not do_sequence:
                        raise ValueError("do_sequence must be a non-empty list")

                    total_s = 0.0
                    for item in do_sequence:
                        if not (isinstance(item, (list, tuple)) and len(item) == 2):
                            raise ValueError("do_sequence items must be (value, hold_s)")
                        total_s += max(0.0, float(item[1]))

                    insert_index = int(cmd.get("insert_index", -1))
                    ao_width_ms = float(cmd.get("ao_width_ms", 0.0))
                    if insert_index >= 0 and ao_width_ms > 0:
                        total_s += ao_width_ms / 1000.0

                    # Avoid long sleeps if someone passes huge numbers accidentally.
                    time.sleep(min(total_s, 0.5))
                    send({"ok": True, "event": "run_sequence_once", "slept_s": float(min(total_s, 0.5))})

                else:
                    send({"ok": False, "event": "error", "error": f"unknown cmd: {name}"})

            except Exception as e:
                send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})

    except Exception as e:
        send({"ok": False, "event": "fatal", "error": str(e), "traceback": traceback.format_exc(limit=12)})
