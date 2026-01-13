from __future__ import annotations

from typing import Any

from .workers import start_daq_worker, stop_daq_worker


def connect_daq(app: Any, *, default_daq_device: str, nm_397: int) -> None:
    try:
        device = app.device_var.get().strip() or default_daq_device
        mode = app.device_mode_var.get().strip().lower() or "real"
        start_daq_worker(app, device=device, mode=mode)
        app._daq_device = device
        app._daq_mode = mode

        try:
            app._daq.request({"cmd": "set_do", "value": int(nm_397)}, timeout=2.0)
        except Exception:
            pass

        app.status_var.set(f"Connected: {device} ({mode})")
        app.connect_btn.configure(state="disabled")
        app.disconnect_btn.configure(state="normal")
    except Exception as e:
        app._daq_device = None
        from tkinter import messagebox

        messagebox.showerror("Connect failed", str(e))


def disconnect_daq(app: Any, *, all_off: int) -> None:
    try:
        app._stop_sequence()
    except Exception:
        pass

    try:
        if app._daq.connected:
            try:
                app._daq.request({"cmd": "set_do", "value": int(all_off)}, timeout=2.0)
            except Exception:
                pass
        stop_daq_worker(app)
    except Exception:
        pass

    app._daq_device = None
    app.status_var.set("Disconnected")
    app.connect_btn.configure(state="normal")
    app.disconnect_btn.configure(state="disabled")
