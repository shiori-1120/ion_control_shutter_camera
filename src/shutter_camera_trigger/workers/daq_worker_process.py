from __future__ import annotations

from multiprocessing import Process, Queue
from typing import Callable

from ..daq_worker_dry import daq_worker_dry_main
from ..daq_worker_mpq import daq_worker_mpq_main


def start_daq_worker_process(
    *,
    device: str,
    mode: str,
    start: bool = True,
    wait_ready: bool = True,
    log_path: str | None = None,
    run_id: str | None = None,
) -> tuple[Process, Queue, Queue]:
    """Start DAQ worker process and return (process, cmd_q, resp_q).

    The worker sends a single "ready" response dict on resp_q.

    Args:
        start: If False, returns a not-yet-started Process.
        wait_ready: If True, waits for and validates the ready message.
    """

    cmd_q: Queue = Queue()
    resp_q: Queue = Queue()

    worker: Callable
    worker = daq_worker_dry_main if mode == "dry" else daq_worker_mpq_main

    proc = Process(
        target=worker,
        args=(
            cmd_q,
            resp_q,
            {"device": device, "mode": mode, "log_path": log_path, "run_id": run_id},
        ),
        daemon=True,
    )
    if start:
        proc.start()

    if wait_ready:
        ready = resp_q.get(timeout=8)
        if not isinstance(ready, dict) or not ready.get("ok"):
            raise RuntimeError(f"DAQ worker failed: {ready}")

    return proc, cmd_q, resp_q
