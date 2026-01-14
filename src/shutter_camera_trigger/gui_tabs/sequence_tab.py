from __future__ import annotations

from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import messagebox, ttk

from ..sequence.controller import start_sequence, stop_sequence


def _load_sequence_params(path: Path):
    from ..sweep.session_parse import read_sequence_json_params

    return read_sequence_json_params(seq_path=path)


def _format_sequence_meta(params: Any | None) -> str:
    if params is None:
        return "Camera actions: N/A | Sync markers: N/A"
    return f"Camera actions: {len(params.camera_actions)} | Sync markers: {len(params.sync_markers)}"


def _set_sequence_meta(app: Any, params: Any | None) -> None:
    if getattr(app, "seq_meta_var", None) is None:
        return
    try:
        app.seq_meta_var.set(_format_sequence_meta(params))
    except Exception:
        pass


def _resolve_sequence_path(app: Any, default_seq_path: Path) -> Path:
    try:
        candidate = getattr(app, "sw_seq_path", None)
        if candidate is not None:
            raw = str(candidate.get() or "").strip()
            if raw:
                return Path(raw)
    except Exception:
        pass
    return default_seq_path


def _refresh_sequence_text(app: Any, *, default_seq_path: Path) -> None:
    if getattr(app, "seq_text", None) is None:
        return
    path = _resolve_sequence_path(app, default_seq_path)
    try:
        params = _load_sequence_params(path)
        text = str(params.sequence_text or "")
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        text = f"# Error loading sequence JSON\n# {e}\n"
        params = None
    try:
        app.seq_text.configure(state=tk.NORMAL)
        app.seq_text.delete("1.0", tk.END)
        app.seq_text.insert("1.0", text)
        app.seq_text.configure(state=tk.DISABLED)
    except Exception:
        pass
    _set_sequence_meta(app, params)
    if params is not None:
        try:
            app.insert_index_var.set(str(int(params.ao_insert_index)))
        except Exception:
            pass
        try:
            if getattr(app, "width_var", None) is not None:
                app.width_var.set(str(float(params.ao_width_ms)))
        except Exception:
            pass


def build_sequence_tab(
    app: Any,
    *,
    bitstring_help: str,
    default_seq_path: Path,
    seq_bits: int,
    all_off: int,
    nm_397: int,
    nm_397_sig: int,
    nm_729: int,
    nm_854: int,
    ao_rate_hz: float,
) -> None:
    if app.seq_tab is None:
        return

    row = ttk.Frame(app.seq_tab)
    row.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(row, text="AO insert index").grid(row=0, column=0, sticky=tk.W)
    app.insert_index_var = tk.StringVar(value="1")
    ttk.Entry(row, textvariable=app.insert_index_var, width=6, state="readonly").grid(row=0, column=1, padx=4)

    ttk.Label(row, text=bitstring_help).grid(row=0, column=2, sticky=tk.W, padx=(8, 0))
    app.seq_meta_var = tk.StringVar(value="Camera actions: 0 | Sync markers: 0")
    ttk.Label(row, textvariable=app.seq_meta_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))

    btn_row = ttk.Frame(app.seq_tab)
    btn_row.pack(fill=tk.X, pady=(6, 6))

    app.start_btn = ttk.Button(
        btn_row,
        text="Start",
        command=lambda: start_sequence(
            app,
            seq_path=_resolve_sequence_path(app, default_seq_path),
            ao_rate_hz=ao_rate_hz,
            nm_397=nm_397,
        ),
    )
    app.start_btn.pack(side=tk.LEFT, padx=4)
    app.stop_btn = ttk.Button(
        btn_row,
        text="Stop",
        command=lambda: stop_sequence(app, nm_397=nm_397),
        state=tk.DISABLED,
    )
    app.stop_btn.pack(side=tk.LEFT, padx=4)
    ttk.Button(
        btn_row,
        text="Reload JSON",
        command=lambda: _refresh_sequence_text(app, default_seq_path=default_seq_path),
    ).pack(side=tk.LEFT, padx=4)

    text_row = ttk.Frame(app.seq_tab)
    text_row.pack(fill=tk.BOTH, expand=True)

    ttk.Label(app.seq_tab, text="Sequence JSON (read-only view)").pack(anchor=tk.W, pady=(6, 4))

    app.seq_text = tk.Text(text_row, height=14, wrap=tk.NONE)
    try:
        path = _resolve_sequence_path(app, default_seq_path)
        params = _load_sequence_params(path)
        initial_text = str(params.sequence_text or "")
        _set_sequence_meta(app, params)
        try:
            app.insert_index_var.set(str(int(params.ao_insert_index)))
        except Exception:
            pass
        try:
            if getattr(app, "width_var", None) is not None:
                app.width_var.set(str(float(params.ao_width_ms)))
        except Exception:
            pass
    except Exception as e:
        initial_text = f"# Error loading sequence JSON\n# {e}\n"
        _set_sequence_meta(app, None)
    app.seq_text.insert("1.0", initial_text)
    app.seq_text.configure(state=tk.DISABLED)
    app.seq_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    yscroll = ttk.Scrollbar(text_row, orient=tk.VERTICAL, command=app.seq_text.yview)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    app.seq_text.configure(yscrollcommand=yscroll.set)

    try:
        text_row.grid_columnconfigure(0, weight=1)
    except Exception:
        pass
