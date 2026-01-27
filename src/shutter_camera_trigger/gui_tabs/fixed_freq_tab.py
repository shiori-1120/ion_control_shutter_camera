from __future__ import annotations

from typing import Any, Callable
import tkinter as tk
from tkinter import ttk


def build_fixed_freq_tab(
    app: Any,
    *,
    start_fixed_cb: Callable[[], None],
    stop_cb: Callable[[], None],
) -> None:
    if getattr(app, "fixed_tab", None) is None:
        return

    row = ttk.Frame(app.fixed_tab)
    row.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(row, text="Fixed frequency (Hz)").pack(side=tk.LEFT, padx=4)
    app.fixed_freq_var = tk.StringVar(value="")
    ttk.Entry(row, textvariable=app.fixed_freq_var, width=16).pack(side=tk.LEFT, padx=4)

    ttk.Label(row, text="n_target").pack(side=tk.LEFT, padx=(12, 4))
    app.fixed_n_target_var = tk.StringVar(value=str(getattr(app, "sw_n_target", tk.StringVar(value="50")).get()))
    ttk.Entry(row, textvariable=app.fixed_n_target_var, width=6).pack(side=tk.LEFT, padx=4)

    ttk.Label(row, text="max_attempt").pack(side=tk.LEFT, padx=(12, 4))
    app.fixed_max_attempt_var = tk.StringVar(value=str(getattr(app, "sw_max_attempt", tk.StringVar(value="100")).get()))
    ttk.Entry(row, textvariable=app.fixed_max_attempt_var, width=6).pack(side=tk.LEFT, padx=4)

    btn_row = ttk.Frame(app.fixed_tab)
    btn_row.pack(fill=tk.X, pady=(6, 6))
    ttk.Button(btn_row, text="Run fixed freq", command=start_fixed_cb).pack(side=tk.LEFT, padx=4)
    ttk.Button(btn_row, text="Stop", command=stop_cb).pack(side=tk.LEFT, padx=4)

    app.fixed_freq_result_var = tk.StringVar(value="No results")

    ttk.Label(app.fixed_tab, textvariable=app.fixed_freq_result_var, wraplength=760, justify=tk.LEFT).pack(
        anchor=tk.W, pady=(6, 0)
    )
