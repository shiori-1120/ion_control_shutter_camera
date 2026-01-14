from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Any


def append_state_history(app: Any, *, prev: str, next_state: str) -> None:
    """Record a sweep state transition in the Diagnostics tab."""

    try:
        history = getattr(app, "_state_history", None)
        if history is None:
            history = []
            app._state_history = history
        entry = {
            "t": datetime.now().strftime("%H:%M:%S"),
            "prev": str(prev),
            "next": str(next_state),
        }
        history.append(entry)
        if len(history) > 40:
            del history[:-40]
    except Exception:
        return

    try:
        lb = getattr(app, "diag_state_list", None)
        if lb is not None:
            lb.delete(0, "end")
            for item in getattr(app, "_state_history", []):
                lb.insert("end", f"{item.get('t','')} {item.get('prev','')} -> {item.get('next','')}")
            lb.see("end")
    except Exception:
        pass

    try:
        if getattr(app, "_logger", None):
            app._logger.info("diagnostics_state prev=%s next=%s", prev, next_state)
    except Exception:
        pass


def set_last_error(app: Any, *, label: str, message: str, log_path: str | None = None) -> None:
    """Update the Diagnostics tab with the latest error information."""

    error_text = f"Error: {label} | {message}"
    log_text = f"Log: {log_path}" if log_path else ""
    try:
        app._last_error_label = str(label)
        app._last_error_message = str(message)
        app._last_error_log = str(log_path) if log_path else ""
    except Exception:
        pass

    try:
        history = getattr(app, "_error_history", None)
        if history is None:
            history = []
            app._error_history = history
        entry = {
            "t": datetime.now().strftime("%H:%M:%S"),
            "label": str(label),
            "message": str(message),
            "log_path": str(log_path) if log_path else "",
        }
        history.append(entry)
        if len(history) > 20:
            del history[:-20]
    except Exception:
        pass

    try:
        if getattr(app, "diag_error_var", None) is not None:
            app.diag_error_var.set(error_text)
    except Exception:
        pass
    try:
        if getattr(app, "diag_log_var", None) is not None:
            app.diag_log_var.set(log_text)
    except Exception:
        pass

    try:
        lb = getattr(app, "diag_history_list", None)
        if lb is not None:
            lb.delete(0, "end")
            for item in getattr(app, "_error_history", []):
                lb.insert("end", f"{item.get('t','')} {item.get('label','')}: {item.get('message','')}")
            lb.selection_clear(0, "end")
            lb.selection_set("end")
            lb.see("end")
    except Exception:
        pass

    try:
        if getattr(app, "_logger", None):
            app._logger.error("diagnostics_error label=%s message=%s log_path=%s", label, message, log_path or "")
            if sys.exc_info()[0]:
                app._logger.error("Traceback:\n%s", traceback.format_exc())
    except Exception:
        pass


def resolve_log_path(app: Any, *, filename: str) -> str | None:
    """Resolve a log file path from the app log context."""

    ctx = getattr(app, "_log_ctx", None)
    if ctx is None:
        return None
    try:
        log_dir = getattr(ctx, "log_dir", None)
        if log_dir is None:
            return None
        return str(Path(log_dir) / filename)
    except Exception:
        return None
