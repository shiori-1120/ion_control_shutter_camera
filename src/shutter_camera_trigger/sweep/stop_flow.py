from __future__ import annotations

from typing import Any, Callable


def stop_sweep_workers(
    *,
    queues: dict[str, Any],
    procs: list[Any],
    nm_397: int,
    join_with_ui: Callable[[Any, float], None],
    write_last_worker_pids_cb: Callable[[dict[str, Any]], None],
) -> list[Any]:
    # Best-effort: keep 397 ON when leaving sweep.
    try:
        if queues.get("daq_cmd"):
            queues["daq_cmd"].put({"cmd": "set_do", "value": int(nm_397)})
    except Exception:
        pass
    # Tell workers to close.
    try:
        if queues.get("daq_cmd"):
            queues["daq_cmd"].put({"cmd": "close"})
        if queues.get("cam_cmd"):
            queues["cam_cmd"].put({"cmd": "close"})
    except Exception:
        pass

    # Prefer graceful shutdown so camera resources are properly released.
    # Order is [daq_p, cam_p]. Give camera longer.
    for i, p in enumerate(procs):
        try:
            timeout = 2.0 if i == 0 else 6.0
            join_with_ui(p, timeout=timeout)
        except Exception:
            pass

    for p in procs:
        try:
            if p.is_alive():
                p.terminate()
                join_with_ui(p, timeout=1.0)
        except Exception:
            pass

    # Clear pid record after stopping sweep.
    try:
        write_last_worker_pids_cb({})
    except Exception:
        pass

    return []
