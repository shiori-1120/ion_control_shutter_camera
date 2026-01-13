from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
from typing import Any, Optional

import tkinter as tk
from tkinter import ttk


@dataclass
class LogPanelState:
    text: tk.Text
    queue: queue.Queue
    buffer: list[str]
    max_lines: int
    follow_tail: bool
    level_var: tk.StringVar
    filter_var: tk.StringVar


def build_log_panel(app: Any) -> None:
    """Attach a log panel that drains the app log queue into a Text widget."""

    ctx = getattr(app, "_log_ctx", None)
    if ctx is None or ctx.gui_queue is None:
        return

    panel = ttk.Frame(app)
    panel.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=10, pady=(0, 10))

    controls = ttk.Frame(panel)
    controls.pack(side=tk.TOP, fill=tk.X)

    level_var = tk.StringVar(value="INFO")
    ttk.Label(controls, text="Log level").pack(side=tk.LEFT)
    ttk.Combobox(controls, textvariable=level_var, values=["INFO", "DEBUG"], width=8, state="readonly").pack(
        side=tk.LEFT, padx=6
    )

    filter_var = tk.StringVar(value="all")
    ttk.Label(controls, text="Filter").pack(side=tk.LEFT, padx=(12, 0))
    ttk.Combobox(
        controls,
        textvariable=filter_var,
        values=["all", "camera"],
        width=10,
        state="readonly",
    ).pack(side=tk.LEFT, padx=6)

    text = tk.Text(panel, height=8, wrap=tk.NONE)
    text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    yscroll = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=text.yview)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    text.configure(yscrollcommand=yscroll.set)

    state = LogPanelState(
        text=text,
        queue=ctx.gui_queue,
        buffer=[],
        max_lines=2000,
        follow_tail=True,
        level_var=level_var,
        filter_var=filter_var,
    )
    app._log_panel_state = state
    text.bind("<MouseWheel>", lambda _e: _set_follow_tail(state, False))
    text.bind("<Button-4>", lambda _e: _set_follow_tail(state, False))
    text.bind("<Button-5>", lambda _e: _set_follow_tail(state, False))
    text.bind("<Key>", lambda _e: _set_follow_tail(state, False))

    _drain_log_queue(app, state)


def _set_follow_tail(state: LogPanelState, enabled: bool) -> None:
    state.follow_tail = enabled


def _drain_log_queue(app: Any, state: LogPanelState) -> None:
    try:
        while True:
            record = state.queue.get_nowait()
            line = _format_record(record)
            if line is None:
                continue
            if not _passes_filters(state, record, line):
                continue
            state.buffer.append(line)
    except queue.Empty:
        pass

    if state.buffer:
        _flush_buffer(state)

    try:
        app.after(200, lambda: _drain_log_queue(app, state))
    except Exception:
        pass


def _format_record(record: logging.LogRecord) -> Optional[str]:
    try:
        run_id = getattr(record, "run_id", "")
        ts = getattr(record, "created", None)
        if ts is None:
            ts = ""
        else:
            ts = f"{ts:.3f}"
        return f"{ts} {run_id} {record.levelname} {record.name} {record.getMessage()}\n"
    except Exception:
        return None


def _passes_filters(state: LogPanelState, record: logging.LogRecord, line: str) -> bool:
    level = state.level_var.get()
    if level == "DEBUG" and record.levelno < logging.DEBUG:
        return False
    if level == "INFO" and record.levelno < logging.INFO:
        return False

    filt = state.filter_var.get()
    if filt == "camera":
        return "camera" in record.name.lower() or "camera" in line.lower()
    return True


def _flush_buffer(state: LogPanelState) -> None:
    state.text.configure(state=tk.NORMAL)
    state.text.insert("end", "".join(state.buffer))
    state.buffer.clear()

    _truncate_lines(state)
    if state.follow_tail:
        state.text.see("end")
    state.text.configure(state=tk.DISABLED)


def _truncate_lines(state: LogPanelState) -> None:
    try:
        current = int(state.text.index("end-1c").split(".")[0])
    except Exception:
        current = 0
    if current <= state.max_lines:
        return
    try:
        state.text.delete("1.0", f"{current - state.max_lines}.0")
    except Exception:
        pass
