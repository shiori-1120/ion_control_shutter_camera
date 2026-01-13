from __future__ import annotations

from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import ttk

from ..sequence.controller import start_sequence, stop_sequence


def _load_sequence_text(default_seq_path: Path) -> str:
    try:
        from ..sweep.session_parse import read_sequence_json_params

        params = read_sequence_json_params(seq_path=default_seq_path)
        return str(params.sequence_text or "")
    except Exception:
        return ""


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
    ttk.Entry(row, textvariable=app.insert_index_var, width=6).grid(row=0, column=1, padx=4)

    ttk.Label(row, text=bitstring_help).grid(row=0, column=2, sticky=tk.W, padx=(8, 0))

    btn_row = ttk.Frame(app.seq_tab)
    btn_row.pack(fill=tk.X, pady=(6, 6))

    app.start_btn = ttk.Button(
        btn_row,
        text="Start",
        command=lambda: start_sequence(
            app,
            seq_bits=seq_bits,
            all_off=all_off,
            nm_397=nm_397,
            nm_397_sig=nm_397_sig,
            nm_729=nm_729,
            nm_854=nm_854,
            ao_rate_hz=ao_rate_hz,
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

    text_row = ttk.Frame(app.seq_tab)
    text_row.pack(fill=tk.BOTH, expand=True)

    app.seq_text = tk.Text(text_row, height=14, wrap=tk.NONE)
    app.seq_text.insert("1.0", _load_sequence_text(default_seq_path))
    app.seq_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    yscroll = ttk.Scrollbar(text_row, orient=tk.VERTICAL, command=app.seq_text.yview)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    app.seq_text.configure(yscrollcommand=yscroll.set)

    try:
        text_row.grid_columnconfigure(0, weight=1)
    except Exception:
        pass
