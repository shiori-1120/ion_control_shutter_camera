from __future__ import annotations

from typing import Any

from ..hardware import DaqClientDevice

from ..daq.guards import require_connected
from ..gui_support.diagnostics import resolve_log_path, set_last_error
from tkinter import messagebox


def all_off(app: Any, *, all_off: int, nm_397: int) -> None:
    try:
        require_connected(app)
        do_all_off = False
        try:
            do_all_off = bool(
                messagebox.askyesno(
                    "All Off",
                    "397 nm is normally kept ON outside sequences.\n\nTurn all outputs OFF?",
                    parent=app,
                )
            )
        except Exception:
            do_all_off = False
    DaqClientDevice(app._daq).set_do(int(all_off if do_all_off else nm_397))
    except Exception as e:
        messagebox.showerror("DO error", str(e))
        set_last_error(
            app,
            label="Manual DO",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )


def apply_manual(
    app: Any,
    *,
    nm_397: int,
    nm_397_sig: int,
    nm_729: int,
    nm_854: int,
) -> None:
    try:
        require_connected(app)
        value = 0
        if app.v_397.get():
            value |= nm_397
        if app.v_397s.get():
            value |= nm_397_sig
        if app.v_729.get():
            value |= nm_729
        if app.v_854.get():
            value |= nm_854
    DaqClientDevice(app._daq).set_do(int(value))
    except Exception as e:
        messagebox.showerror("Manual apply error", str(e))
        set_last_error(
            app,
            label="Manual apply",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )
