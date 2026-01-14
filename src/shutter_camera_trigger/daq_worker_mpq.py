"""DAQ worker process (multiprocessing.Queue based).

Why this exists:
- The repo already has a socket-based daq_worker for GUI use.
- For the new same-PC multi-process runner, a Queue-based worker is simpler and
  avoids managing listener ports/auth.

Protocol:
- cmd_q: dict commands
  - {"cmd":"set_do","value":int}
  - {"cmd":"run_sequence_once","do_sequence":[(value,hold_s),...],"insert_index":int,
     "ao_width_ms":float,"ao_rate_hz":float,"ao_v_high":float,"ao_v_low":float}
  - {"cmd":"close"}
- resp_q: dict responses

This process is intentionally lightweight.
"""

from __future__ import annotations

import queue
import traceback
import time
import os
from multiprocessing.queues import Queue
from typing import Any

from .daq_core import DaqSession, DryDaqSession, run_do_sequence_once


def daq_worker_mpq_main(cmd_q: Queue, resp_q: Queue, cfg: dict[str, Any]) -> None:
    session: DaqSession | DryDaqSession | None = None
    log_path = cfg.get("log_path")
    run_id = str(cfg.get("run_id") or "")
    _log_file: Any | None = None

    def log(msg: str) -> None:
        nonlocal _log_file
        if not log_path:
            return
        try:
            if _log_file is None:
                from pathlib import Path

                p = Path(str(log_path))
                p.parent.mkdir(parents=True, exist_ok=True)
                _log_file = open(p, "a", encoding="utf-8")
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"[{ts}]"
            if run_id:
                prefix = f"{prefix} {run_id}"
            _log_file.write(f"{prefix} {msg}\n")
            _log_file.flush()
        except Exception:
            pass

    def send(msg: dict[str, Any]) -> None:
        try:
            resp_q.put(msg)
        except Exception:
            pass
    trace_daq = str(os.environ.get("ION_CONTROL_DAQ_TRACE", "")).strip() == "1"

    try:
        device = str(cfg.get("device") or "Dev1")
        mode = str(cfg.get("mode") or "real").lower()
        log(f"worker start | pid={getattr(__import__('os'), 'getpid')()} | mode={mode} | device={device}")
        if mode == "dry":
            session = DryDaqSession(device=f"{device} (dry)")
        else:
            session = DaqSession(device=device)
        send({"ok": True, "event": "ready", "device": device, "mode": mode})

        while True:
            try:
                cmd = cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not isinstance(cmd, dict):
                continue

            name = cmd.get("cmd")
            if name in ("quit", "close"):
                log("closing")
                send({"ok": True, "event": "closing"})
                break

            if session is None:
                send({"ok": False, "event": "error", "error": "not connected"})
                continue

            try:
                if name == "set_do":
                    session.set_do(int(cmd.get("value", 0)))
                    log(f"set_do value={int(cmd.get('value', 0))}")
                    send({"ok": True, "event": "set_do"})

                elif name == "run_sequence_once":
                    do_sequence = cmd.get("do_sequence")
                    if not isinstance(do_sequence, list) or not do_sequence:
                        raise ValueError("do_sequence must be a non-empty list")

                    parsed: list[tuple[int, float]] = []
                    for item in do_sequence:
                        if not (isinstance(item, (list, tuple)) and len(item) == 2):
                            raise ValueError("do_sequence items must be (value, hold_s)")
                        parsed.append((int(item[0]), float(item[1])))

                    if trace_daq:
                        log(f"run_sequence_once do_sequence={parsed}")

                    run_do_sequence_once(
                        session,
                        parsed,
                        insert_index=int(cmd.get("insert_index", -1)),
                        ao_rate_hz=float(cmd.get("ao_rate_hz", 5000.0)),
                        ao_width_ms=float(cmd.get("ao_width_ms", 1.0)),
                        ao_v_high=float(cmd.get("ao_v_high", 5.0)),
                        ao_v_low=float(cmd.get("ao_v_low", 0.0)),
                    )
                    log(
                        f"run_sequence_once len={len(parsed)} "
                        f"insert_index={int(cmd.get('insert_index', -1))} "
                        f"ao_width_ms={float(cmd.get('ao_width_ms', 1.0)):.3f}"
                    )
                    send({"ok": True, "event": "run_sequence_once"})

                else:
                    send({"ok": False, "event": "error", "error": f"unknown cmd: {name}"})

            except Exception as e:
                log(f"error {type(e).__name__}: {e}")
                send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})

    except Exception as e:
        log(f"fatal {type(e).__name__}: {e}")
        send({"ok": False, "event": "fatal", "error": str(e), "traceback": traceback.format_exc(limit=12)})
    finally:
        try:
            if session is not None:
                session.close()
        except Exception:
            pass
