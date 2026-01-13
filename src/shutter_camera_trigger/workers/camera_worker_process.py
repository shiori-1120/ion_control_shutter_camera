from __future__ import annotations

from multiprocessing import Process, Queue
from typing import Any, Callable


def start_camera_worker_process(
    *,
    cfg: dict[str, Any],
    start: bool = True,
    cmd_q: Queue | None = None,
    resp_q: Queue | None = None,
) -> tuple[Process, Queue, Queue]:
    """Start camera worker process and return (process, cmd_q, resp_q).

    Note: This function intentionally does not wait for the initial "ready" message.
    Some call sites need to prime external-trigger cameras while waiting for ready.
    """

    if cmd_q is None:
        cmd_q = Queue()
    if resp_q is None:
        resp_q = Queue()

    # Import lazily to keep module import lightweight.
    from src.camera.ion_state_worker import ion_state_worker_main

    worker: Callable = ion_state_worker_main
    proc = Process(target=worker, args=(cmd_q, resp_q, cfg), daemon=True)
    if start:
        proc.start()

    return proc, cmd_q, resp_q


def stop_worker_process(
    *,
    proc: Process | None,
    cmd_q: Any | None,
    join_timeout_s: float = 3.0,
    terminate_timeout_s: float = 1.0,
) -> None:
    """Best-effort graceful stop for a worker process.

    Sends {"cmd":"close"} then joins; terminates if still alive.
    """

    try:
        if cmd_q is not None:
            cmd_q.put({"cmd": "close"})
    except Exception:
        pass

    try:
        if proc is None:
            return
        proc.join(timeout=float(join_timeout_s))
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=float(terminate_timeout_s))
    except Exception:
        pass
