from __future__ import annotations

from typing import Any

from ..gui_support.process_cleanup import join_with_ui
from ..workers.daq_worker_process import start_daq_worker_process


def start_daq_worker(app: Any, *, device: str, mode: str) -> None:
    stop_daq_worker(app)
    log_ctx = getattr(app, "_log_ctx", None)
    log_path = None
    run_id = None
    if log_ctx is not None:
        try:
            log_path = str(log_ctx.log_dir / "daq_worker.log")
            run_id = log_ctx.run_id
        except Exception:
            log_path = None
            run_id = None
    proc, cmd_q, resp_q = start_daq_worker_process(device=device, mode=mode, log_path=log_path, run_id=run_id)
    app._daq_proc = proc
    app._daq.attach(cmd_q, resp_q)


def stop_daq_worker(app: Any) -> None:
    app._daq.try_close()
    try:
        if app._daq_proc is not None and app._daq_proc.is_alive():
            join_with_ui(app, app._daq_proc, timeout=2.0)
            if app._daq_proc.is_alive():
                app._daq_proc.terminate()
                join_with_ui(app, app._daq_proc, timeout=1.0)
    except Exception:
        pass

    app._daq_proc = None
    app._daq.detach()
