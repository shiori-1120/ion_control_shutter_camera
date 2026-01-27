from __future__ import annotations

from typing import Any
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"Invalid number: {text!r}") from exc


def build_fixed_freq_tab(app: Any) -> None:
    if getattr(app, "fixed_tab", None) is None:
        return

    row = ttk.Frame(app.fixed_tab)
    row.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(row, text="Fixed frequency (Hz)").pack(side=tk.LEFT, padx=4)
    app.fixed_freq_var = tk.StringVar(value="")
    ttk.Entry(row, textvariable=app.fixed_freq_var, width=16).pack(side=tk.LEFT, padx=4)
    app.fixed_freq_result_var = tk.StringVar(value="No results")

    def _calc() -> None:
        raw = str(app.fixed_freq_var.get() or "").strip()
        if not raw:
            messagebox.showerror("Fixed frequency", "Frequency is empty")
            return
        try:
            target = _parse_float(raw)
        except Exception as e:
            messagebox.showerror("Fixed frequency", str(e))
            return

        results = list(getattr(getattr(app, "_sweep_state", None), "results", []) or [])
        if not results:
            messagebox.showerror("Fixed frequency", "No sweep results yet")
            return

        best = None
        best_delta = None
        for freq, processed, n_bright in results:
            try:
                delta = abs(float(freq) - float(target))
            except Exception:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = (float(freq), int(processed), int(n_bright))

        if best is None:
            messagebox.showerror("Fixed frequency", "No valid sweep points")
            return

        freq, processed, n_bright = best
        p_excited = (float(n_bright) / float(processed)) if processed > 0 else 0.0
        p_dark = 1.0 - p_excited if processed > 0 else 0.0
        app.fixed_freq_result_var.set(
            f"target={target:.6g} Hz | nearest={freq:.6g} Hz | "
            f"p_excite={p_excited:.4f} | p_dark={p_dark:.4f} | "
            f"n={processed} n_bright={n_bright} delta={best_delta:.3g}"
        )

    ttk.Button(row, text="Compute", command=_calc).pack(side=tk.LEFT, padx=6)

    ttk.Label(app.fixed_tab, textvariable=app.fixed_freq_result_var, wraplength=760, justify=tk.LEFT).pack(
        anchor=tk.W, pady=(6, 0)
    )
