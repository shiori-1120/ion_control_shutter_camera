from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from tkinter import messagebox

from ..daq.guards import require_connected

from ..gui_support.diagnostics import resolve_log_path, set_last_error
from ..hardware import DaqClientDevice
from ..sequence.spec import build_sequence_spec, compile_sequence_spec
from ..sweep.session_parse import read_sequence_json_params


def _load_sequence_params(seq_path: Path):
    return read_sequence_json_params(seq_path=seq_path)


def start_sequence(
    app: Any,
    *,
    seq_path: Path,
    ao_rate_hz: float,
    nm_397: int,
) -> None:
    try:
        if not seq_path or not Path(seq_path).exists():
            raise FileNotFoundError(f"Sequence JSON not found: {seq_path}")
        require_connected(app)
        params = _load_sequence_params(Path(seq_path))
        seq_spec = build_sequence_spec(
            do_sequence=params.do_sequence,
            ao_insert_index=int(params.ao_insert_index),
            ao_width_ms=float(params.ao_width_ms),
            ao_rate_hz=float(ao_rate_hz),
            ao_v_high=5.0,
            ao_v_low=0.0,
            camera_actions=params.camera_actions,
            sync_markers=params.sync_markers,
        )
        seq_cmd, _ = compile_sequence_spec(seq_spec)
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        set_last_error(
            app,
            label="Sequence",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )
        return

    app._seq_running = True
    app._seq_thread = threading.Thread(
        target=sequence_loop,
        args=(app, seq_cmd, int(nm_397)),
        daemon=True,
    )
    app._seq_thread.start()

    app.status_var.set(f"Connected: {app._daq_device} ({app._daq_mode}) | Sequence running")
    app.start_btn.configure(state=tk.DISABLED)
    app.stop_btn.configure(state=tk.NORMAL)


def sequence_stopped_ui(app: Any, *, nm_397: int) -> None:
    app.start_btn.configure(state=tk.NORMAL)
    app.stop_btn.configure(state=tk.DISABLED)
    if app._daq.connected:
        try:
            DaqClientDevice(app._daq).set_do(int(nm_397))
        except Exception:
            pass
        app.status_var.set(f"Connected: {app._daq_device} ({app._daq_mode})")


def stop_sequence(app: Any, *, nm_397: int) -> None:
    app._seq_running = False

    if app._seq_thread is None:
        sequence_stopped_ui(app, nm_397=nm_397)
        return

    try:
        alive = app._seq_thread.is_alive()
    except Exception:
        alive = False

    if not alive:
        sequence_stopped_ui(app, nm_397=nm_397)
        return

    try:
        app.start_btn.configure(state=tk.DISABLED)
        app.stop_btn.configure(state=tk.DISABLED)
    except Exception:
        pass

    if not app._seq_stop_polling:
        app._seq_stop_polling = True
        app.after(100, lambda: poll_sequence_stop(app, nm_397=nm_397))


def poll_sequence_stop(app: Any, *, nm_397: int) -> None:
    try:
        t = app._seq_thread
        alive = bool(t and t.is_alive())
    except Exception:
        alive = False

    if alive:
        app.after(100, lambda: poll_sequence_stop(app, nm_397=nm_397))
        return

    app._seq_stop_polling = False
    sequence_stopped_ui(app, nm_397=nm_397)


def sequence_loop(
    app: Any,
    seq_cmd: Any,
    nm_397: int,
) -> None:
    try:
        est_s = 0.0
        try:
            est_s = float(sum(float(hold_s) for _, hold_s in seq_cmd.do_sequence))
        except Exception:
            est_s = 0.0
        req_timeout = max(5.0, est_s + 2.0)

        while app._seq_running:
            DaqClientDevice(app._daq).run_sequence_once(
                seq_cmd
            )
    except Exception as e:
        err = str(e)
        app.after(0, lambda msg=err: messagebox.showerror("Sequence", msg))
        app.after(
            0,
            lambda msg=err: set_last_error(
                app,
                label="Sequence",
                message=msg,
                log_path=resolve_log_path(app, filename="app.log"),
            ),
        )
    finally:
        app._seq_running = False
        app.after(0, lambda: sequence_stopped_ui(app, nm_397=nm_397))
