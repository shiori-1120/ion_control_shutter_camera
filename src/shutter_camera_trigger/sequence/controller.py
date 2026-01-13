from __future__ import annotations

import threading
import tkinter as tk
from typing import Any
from tkinter import messagebox

from ..daq.guards import require_connected

from ..gui_support.sequence_text import SequenceParseOptions, parse_do_sequence_text
from ..gui_support.diagnostics import resolve_log_path, set_last_error
from ..hardware import DaqClientDevice, DaqSequenceCommand


def parse_sequence_text(
    app: Any,
    *,
    seq_bits: int,
    all_off: int,
    nm_397: int,
    nm_397_sig: int,
    nm_729: int,
    nm_854: int,
) -> list[tuple[int, float]]:
    raw = app.seq_text.get("1.0", tk.END)
    name_to_value = {
        "ALL_OFF": int(all_off),
        "NM_397": int(nm_397),
        "NM_397_SIG": int(nm_397_sig),
        "NM_729": int(nm_729),
        "NM_854": int(nm_854),
        "NM_729_854": int(nm_729 | nm_854),
    }
    return parse_do_sequence_text(
        raw,
        options=SequenceParseOptions(
            bits=int(seq_bits),
            strict_bitstring_length=True,
            allow_symbolic_names=True,
        ),
        name_to_value=name_to_value,
        value_min=0,
        value_max=0b1111,
    )


def start_sequence(
    app: Any,
    *,
    seq_bits: int,
    all_off: int,
    nm_397: int,
    nm_397_sig: int,
    nm_729: int,
    nm_854: int,
    ao_rate_hz: float,
) -> None:
    try:
        require_connected(app)
        insert_index = int(app.insert_index_var.get())
        width_ms = float(app.width_var.get())
        do_sequence = parse_sequence_text(
            app,
            seq_bits=seq_bits,
            all_off=all_off,
            nm_397=nm_397,
            nm_397_sig=nm_397_sig,
            nm_729=nm_729,
            nm_854=nm_854,
        )
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
        args=(app, do_sequence, insert_index, width_ms, float(ao_rate_hz), int(nm_397)),
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
    do_sequence: list[tuple[int, float]],
    insert_index: int,
    width_ms: float,
    ao_rate_hz: float,
    nm_397: int,
) -> None:
    try:
        est_s = 0.0
        try:
            est_s = float(sum(float(hold_s) for _, hold_s in do_sequence))
        except Exception:
            est_s = 0.0
        req_timeout = max(5.0, est_s + 2.0)

        while app._seq_running:
            DaqClientDevice(app._daq).run_sequence_once(
                DaqSequenceCommand(
                    do_sequence=do_sequence,
                    ao_insert_index=int(insert_index),
                    ao_width_ms=float(width_ms),
                    ao_rate_hz=float(ao_rate_hz),
                    ao_v_high=5.0,
                    ao_v_low=0.0,
                )
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
