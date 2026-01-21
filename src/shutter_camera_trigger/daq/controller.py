from __future__ import annotations

from typing import Any

from .workers import start_daq_worker, stop_daq_worker
from ..gui_support.diagnostics import resolve_log_path, set_last_error
from ..hardware import DaqClientDevice
from ..gui_support.output_state import set_output_state


def connect_daq(app: Any, *, default_daq_device: str, nm_397: int) -> None:
    try:
        device = app.device_var.get().strip() or default_daq_device
        mode = app.device_mode_var.get().strip().lower() or "real"
        start_daq_worker(app, device=device, mode=mode)
        app._daq_device = device
        app._daq_mode = mode

        try:
            value = int(nm_397)
            DaqClientDevice(app._daq).set_do(value)
            set_output_state(app, value)
        except Exception:
            pass

        app.status_var.set(f"Connected: {device} ({mode})")
        app.connect_btn.configure(state="disabled")
        app.disconnect_btn.configure(state="normal")
        try:
            if getattr(app, "_logger", None):
                app._logger.info("daq_connect_ok device=%s mode=%s", device, mode)
        except Exception:
            pass
    except Exception as e:
        app._daq_device = None
        from tkinter import messagebox

        messagebox.showerror("Connect failed", str(e))
        set_last_error(
            app,
            label="DAQ connect",
            message=str(e),
            log_path=resolve_log_path(app, filename="app.log"),
        )
        try:
            if getattr(app, "_logger", None):
                app._logger.error("daq_connect_failed error=%s", e)
        except Exception:
            pass


def disconnect_daq(app: Any, *, all_off: int) -> None:
    try:
        app._stop_sequence()
    except Exception:
        pass

    try:
        if app._daq.connected:
            try:
                value = int(all_off)
                DaqClientDevice(app._daq).set_do(value)
                set_output_state(app, value)
            except Exception:
                pass
        stop_daq_worker(app)
    except Exception:
        pass

    app._daq_device = None
    app.status_var.set("Disconnected")
    app.connect_btn.configure(state="normal")
    app.disconnect_btn.configure(state="disabled")
    try:
        if getattr(app, "_logger", None):
            app._logger.info("daq_disconnect")
    except Exception:
        pass
