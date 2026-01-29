from __future__ import annotations

from pathlib import Path
from typing import Any
import json
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


def _format_t_s(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except Exception:
        return str(value)


def _format_camera_actions(params: Any | None, *, limit: int = 3) -> str:
    if params is None:
        return "Camera actions: (unknown)"
    actions = list(params.camera_actions or [])
    if not actions:
        return "Camera actions: (none)"
    parts = []
    for item in actions[:limit]:
        kind = str(item.get("kind", "")).strip() or "?"
        t_s = _format_t_s(item.get("t_s", ""))
        tag = ""
        meta = item.get("meta") or {}
        if isinstance(meta, dict):
            tag_val = meta.get("tag")
            if tag_val:
                tag = f" tag={tag_val}"
        parts.append(f"{kind}@{t_s}{tag}")
    suffix = " ..." if len(actions) > limit else ""
    return f"Camera actions: {', '.join(parts)}{suffix}"


def _format_sync_markers(params: Any | None, *, limit: int = 3) -> str:
    if params is None:
        return "Sync markers: (unknown)"
    markers = list(params.sync_markers or [])
    if not markers:
        return "Sync markers: (none)"
    parts = []
    for item in markers[:limit]:
        label = str(item.get("label", "")).strip() or "?"
        t_s = _format_t_s(item.get("t_s", ""))
        parts.append(f"{label}@{t_s}")
    suffix = " ..." if len(markers) > limit else ""
    return f"Sync markers: {', '.join(parts)}{suffix}"


def _set_sequence_meta(app: Any, params: Any | None) -> None:
    if getattr(app, "seq_meta_var", None) is None:
        return
    try:
        app.seq_meta_var.set(_format_sequence_meta(params))
    except Exception:
        pass
    if getattr(app, "seq_actions_var", None) is not None:
        try:
            app.seq_actions_var.set(_format_camera_actions(params))
        except Exception:
            pass
    if getattr(app, "seq_markers_var", None) is not None:
        try:
            app.seq_markers_var.set(_format_sync_markers(params))
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
        try:
            if getattr(app, "seq_plot_win", None) is not None:
                _render_sequence_plot(app, params=params)
        except Exception:
            pass


def _resolve_insert_index(app: Any, *, fallback: int, max_index: int) -> int:
    try:
        raw = str(getattr(app, "insert_index_var", None).get() or "").strip()
    except Exception:
        return int(fallback)
    if not raw:
        return int(fallback)
    try:
        value = int(float(raw))
    except Exception:
        return int(fallback)
    if value < -1 or value > int(max_index):
        return int(fallback)
    return int(value)


def _resolve_ao_width_ms(app: Any, *, fallback: float) -> float:
    try:
        raw = str(getattr(app, "width_var", None).get() or "").strip()
    except Exception:
        return float(fallback)
    if not raw:
        return float(fallback)
    try:
        value = float(raw)
    except Exception:
        return float(fallback)
    if value < 0:
        return float(fallback)
    return float(value)


def _render_sequence_plot(app: Any, *, params: Any) -> None:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except Exception as e:
        messagebox.showerror("Sequence", f"matplotlib not available: {e}")
        return

    win = getattr(app, "seq_plot_win", None)
    if win is None or not bool(getattr(win, "winfo_exists", lambda: False)()):
        win = tk.Toplevel(app)
        win.title("Sequence visualization")
        app.seq_plot_win = win
        app.seq_plot_fig = Figure(figsize=(7.6, 3.4), dpi=100)
        app.seq_plot_ax = app.seq_plot_fig.add_subplot(111)
        app.seq_plot_canvas = FigureCanvasTkAgg(app.seq_plot_fig, master=win)
        app.seq_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    fig = app.seq_plot_fig
    ax = app.seq_plot_ax
    fig.clear()
    ax = fig.add_subplot(111)
    app.seq_plot_ax = ax

    do_sequence = list(params.do_sequence or [])
    ao_insert_index = _resolve_insert_index(
        app,
        fallback=int(params.ao_insert_index),
        max_index=max(-1, len(do_sequence) - 1),
    )
    ao_width_ms = _resolve_ao_width_ms(app, fallback=float(params.ao_width_ms))
    ao_width_s = float(ao_width_ms) / 1000.0

    bit_names = ["397", "397 SIG", "CAM", "854"]
    bit_intervals: dict[int, list[tuple[float, float]]] = {i: [] for i in range(4)}
    t = 0.0
    for value, hold_s in do_sequence:
        hold = float(hold_s)
        for bit in range(4):
            if int(value) & (1 << bit):
                bit_intervals[bit].append((t, hold))
        t += hold

    ao_intervals: list[tuple[float, float]] = []
    if ao_insert_index >= 0 and ao_width_s > 0:
        try:
            t_ao = sum(float(hold_s) for _, hold_s in do_sequence[: ao_insert_index + 1])
            ao_intervals = [(float(t_ao), float(ao_width_s))]
        except Exception:
            ao_intervals = []

    rows = [("854", 3), ("CAM", 2), ("397 SIG", 1), ("397", 0)]
    y_positions = []
    y_labels = []
    y = 0.0
    height = 0.8

    for label, bit in rows:
        intervals = bit_intervals.get(bit, [])
        if intervals:
            ax.broken_barh(intervals, (y, height), facecolors="tab:blue")
        y_positions.append(y + height / 2.0)
        y_labels.append(label)
        y += 1.2

    ao_row_y = y
    if ao_intervals:
        ax.broken_barh(ao_intervals, (ao_row_y, height), facecolors="tab:red")
    y_positions.append(ao_row_y + height / 2.0)
    y_labels.append("AO")

    total_s = float(t)
    max_t = max(total_s, (ao_intervals[0][0] + ao_intervals[0][1]) if ao_intervals else total_s)
    ax.set_xlim(0.0, max(0.001, max_t))
    ax.set_ylim(-0.2, ao_row_y + height + 0.4)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Sequence timeline (total={total_s:.6f}s, ao_index={ao_insert_index}, ao_width_ms={ao_width_ms:.3g})"
    )
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    app.seq_plot_canvas.draw()


def _show_sequence_plot(app: Any, *, default_seq_path: Path) -> None:
    path = _resolve_sequence_path(app, default_seq_path)
    try:
        params = _load_sequence_params(path)
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        return
    _render_sequence_plot(app, params=params)


def _save_sequence_text(app: Any, *, default_seq_path: Path) -> None:
    if getattr(app, "seq_text", None) is None:
        return
    path = _resolve_sequence_path(app, default_seq_path)
    if not path:
        messagebox.showerror("Sequence", "Sequence path is empty.")
        return
    try:
        raw_text = app.seq_text.get("1.0", tk.END)
    except Exception:
        messagebox.showerror("Sequence", "Failed to read sequence text.")
        return
    try:
        from ..gui_support.sequence_text import SequenceParseOptions, parse_do_sequence_text

        bits = int(getattr(app, "seq_bits", 4))
        do_sequence = parse_do_sequence_text(
            raw_text,
            options=SequenceParseOptions(bits=bits, strict_bitstring_length=False),
        )
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        return

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except Exception as e:
        messagebox.showerror("Sequence", f"Failed to read JSON: {e}")
        return

    try:
        data["sequence_text"] = str(raw_text)
        data["do_sequence"] = [{"value": int(v), "hold_s": float(s)} for v, s in do_sequence]
        try:
            raw_idx = str(getattr(app, "insert_index_var", None).get() or "").strip()
        except Exception:
            raw_idx = ""
        if raw_idx:
            ao_idx = int(float(raw_idx))
            if ao_idx < -1 or ao_idx >= len(do_sequence):
                raise ValueError(f"AO insert index must be -1..{len(do_sequence) - 1}")
            data["ao_insert_index"] = int(ao_idx)
        try:
            raw_width = str(getattr(app, "width_var", None).get() or "").strip()
        except Exception:
            raw_width = ""
        if raw_width:
            data["ao_width_ms"] = float(raw_width)
    except Exception as e:
        messagebox.showerror("Sequence", str(e))
        return

    try:
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        messagebox.showerror("Sequence", f"Failed to write JSON: {e}")
        return

    _refresh_sequence_text(app, default_seq_path=default_seq_path)


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
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
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
    app.seq_meta_var = tk.StringVar(value="Camera actions: 0 | Sync markers: 0")
    ttk.Label(row, textvariable=app.seq_meta_var).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(4, 0))
    app.seq_actions_var = tk.StringVar(value="Camera actions: (none)")
    ttk.Label(row, textvariable=app.seq_actions_var, wraplength=720, justify=tk.LEFT).grid(
        row=2, column=0, columnspan=3, sticky=tk.W, pady=(2, 0)
    )
    app.seq_markers_var = tk.StringVar(value="Sync markers: (none)")
    ttk.Label(row, textvariable=app.seq_markers_var, wraplength=720, justify=tk.LEFT).grid(
        row=3, column=0, columnspan=3, sticky=tk.W, pady=(2, 0)
    )

    capture_row = ttk.Frame(app.seq_tab)
    capture_row.pack(fill=tk.X, pady=(6, 0))
    app.seq_capture_enable_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        capture_row,
        text="Capture image each sequence",
        variable=app.seq_capture_enable_var,
    ).pack(side=tk.LEFT, padx=4)
    app.seq_log_enable_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(
        capture_row,
        text="Sequence log",
        variable=app.seq_log_enable_var,
    ).pack(side=tk.LEFT, padx=(12, 4))
    ttk.Label(capture_row, text="Display first N").pack(side=tk.LEFT, padx=(12, 4))
    app.seq_capture_show_n_var = tk.StringVar(value="3")
    ttk.Entry(capture_row, textvariable=app.seq_capture_show_n_var, width=6).pack(side=tk.LEFT, padx=4)

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
            camera_trigger=camera_trigger,
            roi_pulse_s=roi_pulse_s,
            roi_idle_s=roi_idle_s,
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
    ttk.Button(
        btn_row,
        text="Save JSON",
        command=lambda: _save_sequence_text(app, default_seq_path=default_seq_path),
    ).pack(side=tk.LEFT, padx=4)
    ttk.Button(
        btn_row,
        text="Visualize",
        command=lambda: _show_sequence_plot(app, default_seq_path=default_seq_path),
    ).pack(side=tk.LEFT, padx=4)

    text_row = ttk.Frame(app.seq_tab)
    text_row.pack(fill=tk.BOTH, expand=True)

    ttk.Label(app.seq_tab, text="Sequence JSON").pack(anchor=tk.W, pady=(6, 4))

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
    app.seq_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    yscroll = ttk.Scrollbar(text_row, orient=tk.VERTICAL, command=app.seq_text.yview)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)
    app.seq_text.configure(yscrollcommand=yscroll.set)

    app.seq_bits = int(seq_bits)

    try:
        text_row.grid_columnconfigure(0, weight=1)
    except Exception:
        pass
