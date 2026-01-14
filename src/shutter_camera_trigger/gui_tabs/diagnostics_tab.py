from __future__ import annotations

from typing import Any, Callable
import tkinter as tk
from tkinter import ttk


def build_diagnostics_tab(
    app: Any,
    *,
    camera_check_cb: Callable[[], None] | None = None,
    camera_snap_cb: Callable[[], None] | None = None,
) -> None:
    target = getattr(app, "diag_info_tab", None) or app.diag_tab
    if target is None:
        return

    row = ttk.Frame(target)
    row.pack(fill=tk.BOTH, expand=True)

    left = ttk.Frame(row)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

    if camera_check_cb is not None or camera_snap_cb is not None:
        tool_row = ttk.LabelFrame(left, text="Diagnostics tools")
        tool_row.pack(fill=tk.X, pady=(0, 10))
        if camera_check_cb is not None:
            ttk.Button(tool_row, text="Camera check", command=camera_check_cb).pack(side=tk.LEFT, padx=(6, 8), pady=6)
        if camera_snap_cb is not None:
            ttk.Button(tool_row, text="Camera snap", command=camera_snap_cb).pack(side=tk.LEFT, padx=(0, 8), pady=6)

    app.diag_error_var = tk.StringVar(value="Error: (none)")
    app.diag_log_var = tk.StringVar(value="Log: ")

    ttk.Label(left, text="Last error", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 6))
    ttk.Label(left, textvariable=app.diag_error_var, wraplength=520, justify=tk.LEFT).pack(
        anchor=tk.W, fill=tk.X, pady=(0, 6)
    )
    ttk.Label(left, textvariable=app.diag_log_var, wraplength=520, justify=tk.LEFT).pack(anchor=tk.W, fill=tk.X)

    btn_row = ttk.Frame(left)
    btn_row.pack(anchor=tk.W, pady=(12, 0))

    ttk.Button(btn_row, text="Copy log path", command=lambda: _copy_log_path(app)).pack(side=tk.LEFT, padx=(0, 8))
    ttk.Button(btn_row, text="Clear", command=lambda: _clear_error(app)).pack(side=tk.LEFT)

    ttk.Label(left, text="History", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 4))
    history_row = ttk.Frame(left)
    history_row.pack(fill=tk.BOTH, expand=True)

    app.diag_history_list = tk.Listbox(history_row, height=6)
    app.diag_history_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    history_scroll = ttk.Scrollbar(history_row, orient=tk.VERTICAL, command=app.diag_history_list.yview)
    history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    app.diag_history_list.configure(yscrollcommand=history_scroll.set)

    app.diag_history_list.bind("<<ListboxSelect>>", lambda _e: _select_history(app))

    ttk.Label(left, text="Sweep state history", font=("", 10, "bold")).pack(anchor=tk.W, pady=(12, 4))
    state_row = ttk.Frame(left)
    state_row.pack(fill=tk.BOTH, expand=True)

    app.diag_state_list = tk.Listbox(state_row, height=6)
    app.diag_state_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    state_scroll = ttk.Scrollbar(state_row, orient=tk.VERTICAL, command=app.diag_state_list.yview)
    state_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    app.diag_state_list.configure(yscrollcommand=state_scroll.set)

    right = ttk.LabelFrame(row, text="Quick summary")
    right.pack(side=tk.LEFT, fill=tk.Y)
    ttk.Label(right, text="Last error").pack(anchor=tk.W, padx=8, pady=(6, 0))
    ttk.Label(right, textvariable=app.diag_error_var, wraplength=260, justify=tk.LEFT).pack(
        anchor=tk.W, padx=8, pady=(0, 6)
    )
    ttk.Label(right, text="Log").pack(anchor=tk.W, padx=8, pady=(0, 0))
    ttk.Label(right, textvariable=app.diag_log_var, wraplength=260, justify=tk.LEFT).pack(anchor=tk.W, padx=8, pady=(0, 8))
    ttk.Label(right, text="Paths").pack(anchor=tk.W, padx=8, pady=(4, 0))
    app.diag_logs_root_var = tk.StringVar(value=_resolve_logs_path(app))
    app.diag_output_root_var = tk.StringVar(value=_resolve_output_path(app))
    ttk.Label(right, textvariable=app.diag_logs_root_var, wraplength=260, justify=tk.LEFT).pack(
        anchor=tk.W, padx=8, pady=(0, 2)
    )
    ttk.Label(right, textvariable=app.diag_output_root_var, wraplength=260, justify=tk.LEFT).pack(
        anchor=tk.W, padx=8, pady=(0, 8)
    )


def _copy_log_path(app: Any) -> None:
    log_path = str(getattr(app, "_last_error_log", "") or "")
    if not log_path:
        return
    try:
        app.clipboard_clear()
        app.clipboard_append(log_path)
    except Exception:
        pass


def _clear_error(app: Any) -> None:
    try:
        app._last_error_label = ""
        app._last_error_message = ""
        app._last_error_log = ""
    except Exception:
        pass
    try:
        app.diag_error_var.set("Error: (none)")
        app.diag_log_var.set("Log: ")
    except Exception:
        pass
    try:
        if getattr(app, "diag_history_list", None) is not None:
            app.diag_history_list.selection_clear(0, "end")
    except Exception:
        pass


def _select_history(app: Any) -> None:
    try:
        lb = getattr(app, "diag_history_list", None)
        history = getattr(app, "_error_history", [])
        if lb is None:
            return
        sel = lb.curselection()
        if not sel:
            return
        idx = int(sel[0])
        if idx < 0 or idx >= len(history):
            return
        item = history[idx]
        app.diag_error_var.set(f"Error: {item.get('label','')} | {item.get('message','')}")
        log_path = item.get("log_path") or ""
        app.diag_log_var.set(f"Log: {log_path}" if log_path else "Log: ")
        app._last_error_log = log_path
    except Exception:
        pass


def _resolve_logs_path(app: Any) -> str:
    log_dir = getattr(getattr(app, "_log_ctx", None), "log_dir", None)
    return f"Logs: {log_dir}" if log_dir else "Logs: (unknown)"


def _resolve_output_path(app: Any) -> str:
    output_root = getattr(app, "output_root", None)
    return f"Output: {output_root}" if output_root else "Output: (unknown)"
