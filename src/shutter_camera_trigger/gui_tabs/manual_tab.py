from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import ttk

from .manual_actions import all_off, apply_manual


def build_manual_tab(
    app: Any,
    *,
    all_off_value: int,
    nm_397: int,
    nm_397_sig: int,
    nm_729: int,
    nm_854: int,
) -> None:
    if app.manual_tab is None:
        return

    app.v_397 = tk.BooleanVar(value=False)
    app.v_397s = tk.BooleanVar(value=False)
    app.v_729 = tk.BooleanVar(value=False)
    app.v_854 = tk.BooleanVar(value=False)

    ttk.Checkbutton(app.manual_tab, text="397 (line0)", variable=app.v_397).grid(row=1, column=0, sticky=tk.W)
    ttk.Checkbutton(app.manual_tab, text="397 SIG (line1)", variable=app.v_397s).grid(row=2, column=0, sticky=tk.W)
    ttk.Checkbutton(app.manual_tab, text="Camera trigger (line2)", variable=app.v_729).grid(row=3, column=0, sticky=tk.W)
    ttk.Checkbutton(app.manual_tab, text="854 (line3)", variable=app.v_854).grid(row=4, column=0, sticky=tk.W)

    ttk.Button(
        app.manual_tab,
        text="Apply",
        command=lambda: apply_manual(
            app,
            nm_397=nm_397,
            nm_397_sig=nm_397_sig,
            nm_729=nm_729,
            nm_854=nm_854,
        ),
    ).grid(row=1, column=1, padx=10)
    ttk.Button(
        app.manual_tab,
        text="All Off",
        command=lambda: all_off(app, all_off=all_off_value, nm_397=nm_397),
    ).grid(row=2, column=1, padx=10)

    app.manual_tab.grid_columnconfigure(2, weight=1)
