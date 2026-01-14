from __future__ import annotations

from typing import Any

from ..gui_support.logging_setup import get_file_logger
from ..gui_support.diagnostics import resolve_log_path, set_last_error


def connect_fg(app: Any, *, get_amp_vpp) -> None:
    log_ctx = getattr(app, "_log_ctx", None)
    fg_logger = None
    if log_ctx is not None:
        try:
            fg_logger = get_file_logger(
                name="shutter.fg",
                log_dir=log_ctx.log_dir,
                run_id=log_ctx.run_id,
                filename="fg.log",
            )
        except Exception:
            fg_logger = None

    if app._fg_connected:
        disconnect_fg(app)

    resource = app.fg_resource_var.get().strip()
    if not resource:
        from tkinter import messagebox

        messagebox.showerror("FG", "VISA resource is empty")
        set_last_error(
            app,
            label="FG",
            message="VISA resource is empty",
            log_path=resolve_log_path(app, filename="app.log"),
        )
        return

    try:
        from ..hardware import RigolFgDevice

        rig = RigolFgDevice(channel=1, timeout_ms=5000)
        rig.open(resource)
        try:
            rig.apply({"amp_vpp": get_amp_vpp()})
        except Exception:
            pass
        try:
            idn = rig.idn()
            if fg_logger:
                fg_logger.info("idn %s", idn)
            if getattr(app, "_logger", None):
                app._logger.info("fg_idn %s", idn)
        except Exception:
            pass

        app._fg_handle = rig
        app._fg_resource = resource
        app._fg_connected = True
        app.fg_connect_btn.configure(state="disabled")
        app.fg_disconnect_btn.configure(state="normal")
        app.status_var.set(f"FG connected: {resource}")
        try:
            if fg_logger:
                fg_logger.info("connect ok resource=%s", resource)
            if getattr(app, "_logger", None):
                app._logger.info("fg_connect_ok resource=%s", resource)
        except Exception:
            pass
    except Exception as e:
        app._fg_handle = None
        app._fg_resource = None
        app._fg_connected = False
        from tkinter import messagebox

        try:
            if fg_logger:
                fg_logger.error("connect failed error=%s", e)
            if getattr(app, "_logger", None):
                app._logger.error("fg_connect_failed error=%s", e)
        except Exception:
            pass
        messagebox.showerror("FG", str(e))
        set_last_error(
            app,
            label="FG",
            message=str(e),
            log_path=resolve_log_path(app, filename="fg.log"),
        )


def disconnect_fg(app: Any) -> None:
    log_ctx = getattr(app, "_log_ctx", None)
    fg_logger = None
    if log_ctx is not None:
        try:
            fg_logger = get_file_logger(
                name="shutter.fg",
                log_dir=log_ctx.log_dir,
                run_id=log_ctx.run_id,
                filename="fg.log",
            )
        except Exception:
            fg_logger = None

    try:
        if app._fg_handle is not None:
            try:
                app._fg_handle.close()
            except Exception:
                pass
    except Exception:
        pass
    app._fg_handle = None
    app._fg_resource = None
    app._fg_connected = False
    try:
        app.fg_connect_btn.configure(state="normal")
        app.fg_disconnect_btn.configure(state="disabled")
    except Exception:
        pass
    try:
        app.status_var.set("FG disconnected")
    except Exception:
        pass
    try:
        if fg_logger:
            fg_logger.info("disconnect")
        if getattr(app, "_logger", None):
            app._logger.info("fg_disconnect")
    except Exception:
        pass
