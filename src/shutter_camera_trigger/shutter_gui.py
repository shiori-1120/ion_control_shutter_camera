"""Simple GUI for shutter/laser control (DO) + optional AO pulse.

Two modes:
- Sequence: DO→AO→DO pattern loop (AO used only in this mode)
- Manual: toggle each laser line and apply immediately (DO only)

Requirements:
- Windows + NI-DAQmx driver installed
- Python nidaqmx installed

Run:
    # Run from repository root:
    python -m src.shutter_camera_trigger.shutter_gui

    # Or via venv:
    C:/Users/shiori/Desktop/ion_control_shutter_camera/myenv/Scripts/python.exe -m src.shutter_camera_trigger.shutter_gui
"""

from __future__ import annotations

import csv
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import ttk

from multiprocessing import Process, Queue

from .daq_worker_dry import daq_worker_dry_main
from .daq_worker_mpq import daq_worker_mpq_main

# -------------------------
# DO bit mapping (port0/line4:7)
# bit0=line4, bit1=line5, bit2=line6, bit3=line7
# -------------------------
ALL_OFF = 0b0000

NM_397 = 0b0001  # line4
NM_397_SIG = 0b0010  # line5
# bit2 (line6) is used as Camera Trigger (DO)
CAMERA_TRIGGER = 0b0100  # line6
NM_729 = CAMERA_TRIGGER  # backward-compatible alias (was 729 shutter)
NM_854 = 0b1000  # line7

# Sequence bitstring format (recommended in GUI):
#   4 digits: b3 b2 b1 b0
#   meaning:  854 CAM_TRIG 397_SIG 397
# Example:
#   1001 => 854=ON, CAM_TRIG=OFF, 397_SIG=OFF, 397=ON
SEQUENCE_BITS = 4
BITSTRING_HELP = "b3 b2 b1 b0 = 854 CAM_TRIG 397_SIG 397 (rightmost is 397)"

DEFAULT_SEQUENCE_TEXT = (
    "# examples:\n"
    "#   0000 0.001\n"
    "#   0001 0.001   # 397 only\n"
    "#   0010 0.001   # 397_SIG only\n"
    "#   0100 0.001   # Camera trigger only\n"
    "#   1000 0.001   # 854 only\n"
    "#   1100 0.001   # Camera trigger & 854\n"
    "0000 0.001\n"
    "0001 0.001\n"
    "0010 0.001\n"
    "1100 0.001\n"
    "0000 0.001\n"
)

MINIMAL_SEQUENCE_TEXT = (
    "# Minimal DO-only sequence for bring-up\n"
    "# Format: <BITSTRING> <hold_s>\n"
    "0000 0.001\n"
    "0001 0.001\n"
    "0000 0.001\n"
)

# AO waveform: we add 1 LOW sample on both edges for clarity/safety.
AO_RATE_HZ = 5000.0
AO_EDGE_LOW_SAMPLES = 1

DEFAULT_FG_WAVE = "SIN"
DEFAULT_FG_AMP_VPP = 1.0
DEFAULT_FG_OFFSET_V = 0.0
DEFAULT_FG_START_HZ = 1_000.0
DEFAULT_FG_STOP_HZ = 10_000.0
DEFAULT_FG_TIME_S = 1.0
DEFAULT_FG_RESOURCE = "USB0::0x1AB1::0x0641::DG9A00000000::INSTR"  # Rigol DG9xx USB placeholder

# ROI bootstrap (pre-sequence camera TTL check)
ROI_PULSE_S = 0.002
ROI_IDLE_S = 0.002
ROI_MAX_ATTEMPT = 5

# UI
DEFAULT_UI_FONT_SIZE = 12


# -------------------------
# Helper: limit BLAS threads (for online run jitter削減)
# -------------------------
def _limit_blas_threads() -> None:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self._apply_default_fonts(size=DEFAULT_UI_FONT_SIZE)
        self.title("Shutter/Camera Trigger")
        # Keep it reasonably sized so the embedded plot is readable.
        self.geometry("900x650")

        self._daq_proc: Process | None = None
        self._daq_cmd_q: Queue | None = None
        self._daq_resp_q: Queue | None = None
        self._daq_connected = False
        self._daq_device: str | None = None
        self._daq_mode: str = "real"
        self._seq_thread: threading.Thread | None = None
        self._seq_running = False

        self._fg_handle = None
        self._fg_resource: str | None = None
        self._fg_connected = False

        self._camera_connected = False

        self._plot_container: ttk.Frame | None = None
        self._plot_placeholder: ttk.Label | None = None
        self._plot_fig = None
        self._plot_canvas = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _open_usage_doc(self) -> None:
        try:
            # Docs live at repo root, two levels up from this file (src/shutter_camera_trigger).
            doc_path = Path(__file__).resolve().parents[2] / "docs" / "shutter_gui_usage.md"
            if not doc_path.exists():
                messagebox.showinfo("Help", "Usage document not found: docs/shutter_gui_usage.md")
                return
            import webbrowser

            webbrowser.open(doc_path.as_uri())
        except Exception as e:
            messagebox.showerror("Help", str(e))

    def _apply_default_fonts(self, *, size: int) -> None:
        """Increase default Tk/ttk font sizes for readability (Windows tends to be small)."""
        if size <= 0:
            return

        # Tk named fonts
        for name in ("TkDefaultFont", "TkTextFont", "TkHeadingFont", "TkMenuFont", "TkTooltipFont", "TkFixedFont"):
            try:
                f = tkfont.nametofont(name)
                f.configure(size=int(size))
            except Exception:
                pass

        # Ensure ttk widgets follow the Tk default font
        try:
            style = ttk.Style(self)
            style.configure(".", font=tkfont.nametofont("TkDefaultFont"))
        except Exception:
            pass

    def _build_ui(self) -> None:
        # Menu bar (Help -> open usage doc)
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Open usage doc", command=self._open_usage_doc)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # Shared vars
        self.fg_resource_var = tk.StringVar(value="")
        self.camera_mode_top_var = tk.StringVar(value="dry")
        self.dry_image_dir_var = tk.StringVar(value="")

        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Device").grid(row=0, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="Dev3")
        ttk.Entry(top, textvariable=self.device_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(top, text="DAQ mode").grid(row=0, column=2, sticky=tk.W)
        self.device_mode_var = tk.StringVar(value="real")
        ttk.Combobox(top, textvariable=self.device_mode_var, values=["real", "dry"], width=6, state="readonly").grid(
            row=0, column=3, sticky=tk.W, padx=5
        )

        ttk.Label(top, text="AO width (ms)").grid(row=0, column=4, sticky=tk.W)
        self.width_var = tk.StringVar(value="1.0")
        ttk.Entry(top, textvariable=self.width_var, width=10).grid(row=0, column=5, sticky=tk.W, padx=5)

        self.connect_btn = ttk.Button(top, text="Connect", command=self._connect)
        self.connect_btn.grid(row=0, column=6, padx=5)
        self.disconnect_btn = ttk.Button(top, text="Disconnect", command=self._disconnect, state=tk.DISABLED)
        self.disconnect_btn.grid(row=0, column=7)

        ttk.Label(top, text="FG VISA").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.fg_resource_var, width=32).grid(row=1, column=1, columnspan=3, sticky=tk.W, padx=5, pady=(6, 0))
        self.fg_connect_btn = ttk.Button(top, text="FG Connect", command=self._connect_fg)
        self.fg_connect_btn.grid(row=1, column=4, padx=5, pady=(6, 0))
        self.fg_disconnect_btn = ttk.Button(top, text="FG Disconnect", command=self._disconnect_fg, state=tk.DISABLED)
        self.fg_disconnect_btn.grid(row=1, column=5, padx=5, pady=(6, 0))

        ttk.Label(top, text="Camera mode").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_mode_top_var, values=["dry", "real"], width=6, state="readonly").grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=(6, 0)
        )
        self.cam_check_btn = ttk.Button(top, text="Camera check", command=self._check_camera_connection)
        self.cam_check_btn.grid(row=2, column=2, padx=5, pady=(6, 0))

        ttk.Label(top, text="Dry images (dry cam)").grid(row=2, column=3, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.dry_image_dir_var, width=30).grid(row=2, column=4, columnspan=2, sticky=tk.W, padx=5, pady=(6, 0))
        ttk.Button(top, text="...", width=3, command=self._browse_dry_images).grid(row=2, column=6, pady=(6, 0))

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var).grid(row=3, column=0, columnspan=8, sticky=tk.W, pady=(8, 0))

        top.grid_columnconfigure(8, weight=1)

        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.seq_tab = ttk.Frame(nb, padding=10)
        self.manual_tab = ttk.Frame(nb, padding=10)
        self.sweep_tab = ttk.Frame(nb, padding=10)
        nb.add(self.seq_tab, text="Sequence")
        nb.add(self.manual_tab, text="Manual")
        nb.add(self.sweep_tab, text="Sweep")

        self._build_sequence_tab()
        self._build_manual_tab()
        self._build_sweep_tab()

    def _build_sequence_tab(self) -> None:
        info = (
            "DO sequence loops forever.\n"
            "AO pulse is inserted after the selected step index (0-based).\n"
            "Manual mode is DO-only (AO not used)."
        )
        ttk.Label(self.seq_tab, text=info, justify=tk.LEFT).pack(anchor=tk.W)

        ttk.Label(self.seq_tab, text=f"Bitstring mapping: {BITSTRING_HELP}").pack(anchor=tk.W, pady=(6, 0))

        row = ttk.Frame(self.seq_tab)
        row.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(row, text="AO insert index").pack(side=tk.LEFT)
        self.insert_index_var = tk.StringVar(value="1")
        ttk.Entry(row, textvariable=self.insert_index_var, width=6).pack(side=tk.LEFT, padx=5)

        self.start_btn = ttk.Button(row, text="Start", command=self._start_sequence)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(row, text="Stop", command=self._stop_sequence, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        self.plot_btn = ttk.Button(row, text="Plot", command=self._plot_sequence)
        self.plot_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = ttk.Button(row, text="Save", command=self._save_sequence)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        self.load_btn = ttk.Button(row, text="Load", command=self._load_sequence)
        self.load_btn.pack(side=tk.LEFT)

        ttk.Label(
            self.seq_tab,
            text="Sequence (one step per line): <BITSTRING|NAME|INT> <hold_s>",
        ).pack(
            anchor=tk.W, pady=(10, 0)
        )
        self.seq_text = tk.Text(self.seq_tab, height=8, width=60)
        self.seq_text.pack(fill=tk.X, pady=(4, 0))

        self.seq_text.insert("1.0", DEFAULT_SEQUENCE_TEXT)

        # Ensure the project-local sequence library exists (for Save/Load).
        try:
            self._ensure_sequence_library()
        except Exception:
            # Non-fatal: GUI can still run even if library creation fails.
            pass

        # Embedded plot area (same window)
        self._plot_container = ttk.Frame(self.seq_tab)
        self._plot_container.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self._plot_placeholder = ttk.Label(
            self._plot_container,
            text="Plot will appear here (press Plot).",
            justify=tk.LEFT,
        )
        self._plot_placeholder.pack(anchor=tk.W)

    def _ensure_plot_canvas(self) -> None:
        if self._plot_container is None:
            return
        if self._plot_canvas is not None and self._plot_fig is not None:
            return
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except Exception as e:
            messagebox.showerror(
                "Plot",
                f"matplotlib is required for plotting.\n\n{e}",
            )
            return

        if self._plot_placeholder is not None:
            self._plot_placeholder.destroy()
            self._plot_placeholder = None

        fig = Figure(figsize=(9, 3.6), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=self._plot_container)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._plot_fig = fig
        self._plot_canvas = canvas

    def _save_sequence(self) -> None:
        try:
            self._ensure_sequence_library()
            payload = self._get_sequence_payload_from_ui()
        except Exception as e:
            messagebox.showerror("Save", str(e))
            return

        name = simpledialog.askstring(
            "Save sequence",
            "Save as name (stored in project sequence library):",
            parent=self,
        )
        if not name:
            return
        name = name.strip()
        if not name:
            return

        try:
            lib = self._read_sequence_library()
            sequences = lib.setdefault("sequences", {})
            if not isinstance(sequences, dict):
                sequences = {}
                lib["sequences"] = sequences

            if name in sequences:
                if not messagebox.askyesno("Overwrite?", f"'{name}' already exists. Overwrite?", parent=self):
                    return

            sequences[name] = payload
            self._write_sequence_library(lib)
            messagebox.showinfo("Save", f"Saved to library as '{name}'.", parent=self)
        except Exception as e:
            messagebox.showerror("Save", str(e), parent=self)

    def _load_sequence(self) -> None:
        try:
            self._ensure_sequence_library()
            lib = self._read_sequence_library()
            sequences = lib.get("sequences")
            if not isinstance(sequences, dict) or not sequences:
                messagebox.showinfo("Load", "Sequence library is empty.", parent=self)
                return

            names = sorted(str(k) for k in sequences.keys())
            name = self._select_from_list(
                title="Load sequence",
                prompt="Select a sequence name:",
                options=names,
            )
            if name is None:
                return

            if name not in sequences:
                messagebox.showerror("Load", f"No such sequence: '{name}'", parent=self)
                return

            payload = sequences[name]
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid payload for '{name}'")
            self._apply_sequence_payload_to_ui(payload)
        except Exception as e:
            messagebox.showerror("Load", str(e), parent=self)

    def _select_from_list(self, *, title: str, prompt: str, options: list[str]) -> str | None:
        if not options:
            return None

        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.transient(self)
        dlg.grab_set()

        outer = ttk.Frame(dlg, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        ttk.Label(outer, text=prompt).pack(anchor=tk.W)

        list_frame = ttk.Frame(outer)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        yscroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        lb = tk.Listbox(list_frame, height=min(12, max(4, len(options))), exportselection=False)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lb.configure(yscrollcommand=yscroll.set)
        yscroll.configure(command=lb.yview)

        for opt in options:
            lb.insert(tk.END, opt)
        lb.selection_set(0)
        lb.activate(0)
        lb.focus_set()

        result: dict[str, str | None] = {"value": None}

        def on_ok() -> None:
            try:
                sel = lb.curselection()
                if not sel:
                    return
                result["value"] = str(lb.get(sel[0]))
            finally:
                dlg.destroy()

        def on_cancel() -> None:
            result["value"] = None
            dlg.destroy()

        def on_double_click(_event: Any) -> None:
            on_ok()

        lb.bind("<Double-Button-1>", on_double_click)

        btns = ttk.Frame(outer)
        btns.pack(fill=tk.X, pady=(8, 0))

        ttk.Button(btns, text="OK", command=on_ok).pack(side=tk.RIGHT, padx=(6, 0))
        ttk.Button(btns, text="Cancel", command=on_cancel).pack(side=tk.RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", on_cancel)

        # Reasonable dialog size
        dlg.geometry("420x300")

        self.wait_window(dlg)
        return result["value"]

    def _sequence_library_path(self) -> Path:
        # Project-local fixed path: src/shutter_camera_trigger/sequence_examples/sequence_library.json
        return Path(__file__).resolve().parent / "sequence_examples" / "sequence_library.json"

    def _default_sequence_library(self) -> dict[str, Any]:
        return {
            "version": 1,
            "sequences": {
                "default": {
                    "ao_width_ms": 1.0,
                    "ao_insert_index": 1,
                    "sequence_text": DEFAULT_SEQUENCE_TEXT,
                },
                "minimal": {
                    "ao_width_ms": 1.0,
                    "ao_insert_index": -1,
                    "sequence_text": MINIMAL_SEQUENCE_TEXT,
                },
            },
        }

    def _ensure_sequence_library(self) -> None:
        path = self._sequence_library_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        self._write_sequence_library(self._default_sequence_library())

    def _read_sequence_library(self) -> dict[str, Any]:
        path = self._sequence_library_path()
        if not path.exists():
            lib = self._default_sequence_library()
            self._write_sequence_library(lib)
            return lib
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

        if not isinstance(data, dict):
            data = {}

        # Normalize
        if "sequences" not in data or not isinstance(data.get("sequences"), dict):
            data["sequences"] = {}
        if "version" not in data:
            data["version"] = 1

        # Seed mandatory examples if missing
        seqs: dict[str, Any] = data["sequences"]
        if "default" not in seqs:
            seqs["default"] = {
                "ao_width_ms": 1.0,
                "ao_insert_index": 1,
                "sequence_text": DEFAULT_SEQUENCE_TEXT,
            }
        if "minimal" not in seqs:
            seqs["minimal"] = {
                "ao_width_ms": 1.0,
                "ao_insert_index": -1,
                "sequence_text": MINIMAL_SEQUENCE_TEXT,
            }

        return data

    def _write_sequence_library(self, data: dict[str, Any]) -> None:
        path = self._sequence_library_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _get_sequence_payload_from_ui(self) -> dict[str, Any]:
        return {
            "ao_width_ms": float(self.width_var.get()),
            "ao_insert_index": int(self.insert_index_var.get()),
            "sequence_text": str(self.seq_text.get("1.0", tk.END)),
        }

    def _apply_sequence_payload_to_ui(self, payload: dict[str, Any]) -> None:
        if "ao_width_ms" in payload:
            self.width_var.set(str(payload["ao_width_ms"]))
        if "ao_insert_index" in payload:
            self.insert_index_var.set(str(payload["ao_insert_index"]))
        if "sequence_text" in payload:
            self.seq_text.delete("1.0", tk.END)
            self.seq_text.insert("1.0", str(payload["sequence_text"]))

    def _plot_sequence(self) -> None:
        """Parse sequence text and show a simple ON/OFF timeline plot (embedded)."""
        try:
            do_sequence = self._parse_sequence_text()
            insert_index = int(self.insert_index_var.get())
            width_ms = float(self.width_var.get())
        except Exception as e:
            messagebox.showerror("Plot", str(e))
            return

        self._ensure_plot_canvas()
        if self._plot_fig is None or self._plot_canvas is None:
            return

        # Build intervals for each laser.
        # bit0=line4=397, bit1=line5=397_SIG, bit2=line6=729, bit3=line7=854
        lasers = [
            ("854 (b3, line7)", 3),
            ("729 (b2, line6)", 2),
            ("397_SIG (b1, line5)", 1),
            ("397 (b0, line4)", 0),
        ]

        # AO pulse timing (hardware-timed by sample clock)
        if width_ms <= 0:
            raise ValueError("AO width (ms) must be > 0")
        n_high = max(1, int(round((width_ms / 1000.0) * AO_RATE_HZ)))
        ao_high_s = n_high / AO_RATE_HZ
        ao_total_s = (n_high + 2 * AO_EDGE_LOW_SAMPLES) / AO_RATE_HZ
        ao_lead_low_s = AO_EDGE_LOW_SAMPLES / AO_RATE_HZ
        ao_intervals: list[tuple[float, float]] = []

        # Build a timeline of segments: (start, dur, do_value).
        # Real behavior:
        #   set DO -> wait hold_s
        #   if insert_index: run AO waveform (includes low edges)
        #   (During AO, DO stays at the last value)
        segments: list[tuple[float, float, int]] = []
        t = 0.0
        for i, (do_value, hold_s) in enumerate(do_sequence):
            hold_s = float(hold_s)
            do_value_i = int(do_value)
            segments.append((t, hold_s, do_value_i))
            t += hold_s

            if i == insert_index and insert_index >= 0:
                # During AO, DO stays at the current value
                segments.append((t, ao_total_s, do_value_i))
                # Plot only the AO HIGH portion (time still includes LOW edges)
                ao_intervals = [(t + ao_lead_low_s, ao_high_s)]
                t += ao_total_s

        total_s = t

        def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
            if not intervals:
                return []
            merged: list[tuple[float, float]] = [intervals[0]]
            eps = 1e-12
            for start, dur in intervals[1:]:
                last_start, last_dur = merged[-1]
                last_end = last_start + last_dur
                if abs(start - last_end) <= eps:
                    merged[-1] = (last_start, last_dur + dur)
                else:
                    merged.append((start, dur))
            return merged

        laser_intervals: dict[str, list[tuple[float, float]]] = {}
        for label, bit in lasers:
            intervals: list[tuple[float, float]] = []
            mask = 1 << bit
            for start_t, dur_s, do_value in segments:
                if int(do_value) & mask:
                    intervals.append((start_t, float(dur_s)))
            laser_intervals[label] = merge_intervals(intervals)

        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        bar_h = 0.8

        # Plot rows: AO + lasers
        rows: list[tuple[str, list[tuple[float, float]], str]] = []
        rows.append(
            (
                f"AO high (~{ao_high_s*1000.0:.3f} ms, total~{ao_total_s*1000.0:.3f} ms)",
                ao_intervals,
                "tab:red",
            )
        )
        for label, _ in lasers:
            rows.append((label, laser_intervals[label], "tab:blue"))

        for y, (label, intervals, color) in enumerate(reversed(rows)):
            ax.broken_barh(intervals, (y - bar_h / 2, bar_h), facecolors=color)

        ax.set_yticks(list(range(len(rows))))
        ax.set_yticklabels([label for label, _, _ in reversed(rows)])
        ax.set_xlabel("time (s)")
        ax.set_title("Sequence timeline (DO + AO high)")
        ax.set_xlim(0, max(0.0, total_s))
        ax.set_ylim(-1, len(rows))
        ax.grid(True, axis="x", alpha=0.3)

        self._plot_fig.tight_layout()
        self._plot_canvas.draw()

    def _build_manual_tab(self) -> None:
        ttk.Label(self.manual_tab, text="Select lasers then Apply (DO only)").grid(
            row=0, column=0, columnspan=3, sticky=tk.W
        )

        self.v_397 = tk.BooleanVar(value=False)
        self.v_397s = tk.BooleanVar(value=False)
        self.v_729 = tk.BooleanVar(value=False)
        self.v_854 = tk.BooleanVar(value=False)

        ttk.Checkbutton(self.manual_tab, text="397 (line4)", variable=self.v_397).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="397 SIG (line5)", variable=self.v_397s).grid(row=2, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="729 (line6)", variable=self.v_729).grid(row=3, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="854 (line7)", variable=self.v_854).grid(row=4, column=0, sticky=tk.W)

        ttk.Button(self.manual_tab, text="Apply", command=self._apply_manual).grid(row=1, column=1, padx=10)
        ttk.Button(self.manual_tab, text="All Off", command=self._all_off).grid(row=2, column=1, padx=10)

        self.manual_tab.grid_columnconfigure(2, weight=1)

    # ---------------- Sweep tab (queue-based auto sweep) ----------------
    def _build_sweep_tab(self) -> None:
        _limit_blas_threads()

        row = ttk.Frame(self.sweep_tab)
        row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(row, text="Freq start (Hz)").grid(row=0, column=0, sticky=tk.W)
        self.sw_freq_start = tk.StringVar(value="80e6")
        ttk.Entry(row, textvariable=self.sw_freq_start, width=12).grid(row=0, column=1, padx=4)

        ttk.Label(row, text="Freq stop (Hz)").grid(row=0, column=2, sticky=tk.W)
        self.sw_freq_stop = tk.StringVar(value="82e6")
        ttk.Entry(row, textvariable=self.sw_freq_stop, width=12).grid(row=0, column=3, padx=4)

        ttk.Label(row, text="Freq step (Hz)").grid(row=0, column=4, sticky=tk.W)
        self.sw_freq_step = tk.StringVar(value="0.5e6")
        ttk.Entry(row, textvariable=self.sw_freq_step, width=12).grid(row=0, column=5, padx=4)

        ttk.Label(row, text="n_target").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_n_target = tk.StringVar(value="50")
        ttk.Entry(row, textvariable=self.sw_n_target, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))

        ttk.Label(row, text="max_attempt").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))
        self.sw_max_attempt = tk.StringVar(value="200")
        ttk.Entry(row, textvariable=self.sw_max_attempt, width=8).grid(row=1, column=3, padx=4, pady=(6, 0))

        ttk.Label(row, text="settle_s").grid(row=1, column=4, sticky=tk.W, pady=(6, 0))
        self.sw_settle_s = tk.StringVar(value="0.02")
        ttk.Entry(row, textvariable=self.sw_settle_s, width=8).grid(row=1, column=5, padx=4, pady=(6, 0))

        ttk.Label(row, text="Sequence JSON").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_seq_path = tk.StringVar(value="src/shutter_camera_trigger/sequence_examples/minimal_sequence.json")
        ttk.Entry(row, textvariable=self.sw_seq_path, width=48).grid(row=2, column=1, columnspan=4, sticky=tk.W, padx=4, pady=(6, 0))
        ttk.Button(row, text="...", width=3, command=self._pick_seq_json).grid(row=2, column=5, pady=(6, 0))

        ttk.Label(row, text="DAQ mode").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_daq_mode = tk.StringVar(value="dry")
        ttk.Combobox(row, textvariable=self.sw_daq_mode, values=["dry", "real"], width=6, state="readonly").grid(
            row=3, column=1, padx=4, pady=(6, 0)
        )

        ttk.Label(row, text="Camera mode").grid(row=3, column=2, sticky=tk.W, pady=(6, 0))
        self.sw_cam_mode = self.camera_mode_top_var
        ttk.Combobox(row, textvariable=self.sw_cam_mode, values=["dry", "real"], width=6, state="readonly").grid(
            row=3, column=3, padx=4, pady=(6, 0)
        )

        ttk.Label(row, text="DAQ device").grid(row=3, column=4, sticky=tk.W, pady=(6, 0))
        self.sw_device = tk.StringVar(value="Dev3")
        ttk.Entry(row, textvariable=self.sw_device, width=10).grid(row=3, column=5, padx=4, pady=(6, 0))

        ttk.Label(row, text="FG VISA").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_visa = self.fg_resource_var
        ttk.Entry(row, textvariable=self.sw_visa, width=32).grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(6, 0))
        self.sw_no_fg = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="No FG", variable=self.sw_no_fg).grid(row=4, column=4, columnspan=2, sticky=tk.W, pady=(6, 0))

        ttk.Label(row, text="Update interval (s)").grid(row=5, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_update_interval = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.sw_update_interval, width=8).grid(row=5, column=1, padx=4, pady=(6, 0))

        btn_row = ttk.Frame(self.sweep_tab)
        btn_row.pack(fill=tk.X, pady=(8, 8))
        self.sw_start_btn = ttk.Button(btn_row, text="Start sweep", command=self._start_sweep)
        self.sw_start_btn.pack(side=tk.LEFT, padx=4)
        self.sw_stop_btn = ttk.Button(btn_row, text="Stop", command=self._stop_sweep, state=tk.DISABLED)
        self.sw_stop_btn.pack(side=tk.LEFT, padx=4)
        self.sw_status = tk.StringVar(value="Idle")
        ttk.Label(btn_row, textvariable=self.sw_status).pack(side=tk.LEFT, padx=12)

        # Plot area
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            self.sw_fig = Figure(figsize=(7.5, 3.2), dpi=100)
            self.sw_ax = self.sw_fig.add_subplot(111)
            self.sw_canvas = FigureCanvasTkAgg(self.sw_fig, master=self.sweep_tab)
            self.sw_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception:
            self.sw_fig = None
            self.sw_ax = None
            self.sw_canvas = None
            ttk.Label(self.sweep_tab, text="matplotlib not available; real-time plot disabled").pack()

        # Runtime state
        self._sw_running = False
        self._sw_procs: list[Process] = []
        self._sw_queues: dict[str, Queue] = {}
        self._sw_freqs: list[float] = []
        self._sw_results: list[tuple[float, int, int]] = []  # (freq, n_processed, n_bright)
        self._sw_out_dir: Path | None = None
        self._sw_next_update = 0.0

    def _pick_seq_json(self) -> None:
        path = filedialog.askopenfilename(
            title="Select sequence JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if path:
            self.sw_seq_path.set(path)

    def _browse_dry_images(self) -> None:
        """Choose folder for dry camera images (bright*/dark*)."""
        path = filedialog.askdirectory(title="Select dry camera image folder")
        if path:
            self.dry_image_dir_var.set(path)

    def _connect_fg(self) -> None:
        """Open connection to function generator (Rigol DG series)."""
        if self._fg_connected:
            self._disconnect_fg()

        resource = self.fg_resource_var.get().strip()
        if not resource:
            messagebox.showerror("FG", "VISA resource is empty")
            return

        try:
            from src.lib.instruments.rigol_dg import RigolDG, RigolDgConfig

            rig = RigolDG(RigolDgConfig(visa_resource=resource, channel=1, timeout_ms=5000))
            rig.open()
            try:
                _ = rig.idn()
            except Exception:
                # IDN失敗は致命的でないので無視
                pass

            self._fg_handle = rig
            self._fg_resource = resource
            self._fg_connected = True
            self.fg_connect_btn.configure(state=tk.DISABLED)
            self.fg_disconnect_btn.configure(state=tk.NORMAL)
            self.status_var.set(f"FG connected: {resource}")
        except Exception as e:
            self._fg_handle = None
            self._fg_resource = None
            self._fg_connected = False
            messagebox.showerror("FG", str(e))

    def _disconnect_fg(self) -> None:
        """Close FG connection if open."""
        try:
            if self._fg_handle is not None:
                try:
                    self._fg_handle.output(False)
                except Exception:
                    pass
                try:
                    self._fg_handle.close()
                except Exception:
                    pass
        finally:
            self._fg_handle = None
            self._fg_resource = None
            self._fg_connected = False
            try:
                self.fg_connect_btn.configure(state=tk.NORMAL)
                self.fg_disconnect_btn.configure(state=tk.DISABLED)
            except Exception:
                pass

    def _check_camera_connection(self) -> None:
        """Spawn camera worker once to verify connectivity/dry samples."""
        mode = self.camera_mode_top_var.get().strip() or "dry"
        dry_dir = self.dry_image_dir_var.get().strip()
        cfg: dict[str, Any] = {
            "mode": mode,
            "exposure_s": 0.001,
            "frame_timeout_s": 1.0,
            "bootstrap_n": 5,
        }
        if mode == "dry" and dry_dir:
            cfg["dry_image_dir"] = dry_dir

        from src.camera.ion_state_worker import ion_state_worker_main

        cmd_q: Queue = Queue()
        resp_q: Queue = Queue()
        p = Process(target=ion_state_worker_main, args=(cmd_q, resp_q, cfg), daemon=True)
        p.start()
        ok = False
        try:
            ready = resp_q.get(timeout=15)
            if ready.get("ok"):
                ok = True
                dry_samples = ready.get("dry_samples")
                extra = ""
                if dry_samples is not None:
                    extra = f" | dry samples: {dry_samples}"
                messagebox.showinfo("Camera", f"Camera worker ready ({mode}){extra}")
            else:
                messagebox.showerror("Camera", f"Failed: {ready}")
        except Exception as e:
            messagebox.showerror("Camera", f"Failed: {e}")
        finally:
            try:
                cmd_q.put({"cmd": "close"})
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
        self._camera_connected = ok

    def _run_roi_bootstrap(self, daq_cmd_q: Queue, daq_resp_q: Queue, cam_cmd_q: Queue, cam_resp_q: Queue) -> bool:
        """Send simple TTL pulses (camera trigger only) until camera replies or attempts are exhausted."""
        roi_sequence = [
            (ALL_OFF, ROI_IDLE_S),
            (CAMERA_TRIGGER, ROI_PULSE_S),
            (ALL_OFF, ROI_IDLE_S),
        ]
        success = 0
        last_err: str | None = None

        for _attempt in range(ROI_MAX_ATTEMPT):
            try:
                daq_cmd_q.put(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": roi_sequence,
                        "insert_index": -1,
                        "ao_width_ms": 0.0,
                        "ao_rate_hz": AO_RATE_HZ,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )
                daq_resp = daq_resp_q.get(timeout=5)
                if not daq_resp.get("ok"):
                    last_err = f"DAQ: {daq_resp}"
                    continue

                cam_cmd_q.put({"cmd": "get_state", "timeout_s": 1.0})
                cam_resp = cam_resp_q.get(timeout=5)
                if not cam_resp.get("ok"):
                    last_err = f"Camera: {cam_resp}"
                    continue

                success += 1
                if success >= 1:
                    return True
            except Exception as e:
                last_err = str(e)
            time.sleep(max(0.0, ROI_IDLE_S))

        if last_err:
            self.sw_status.set(f"ROI bootstrap failed: {last_err}")
        return False

    def _require_connected(self) -> None:
        if not self._daq_connected:
            raise RuntimeError("Not connected")

    def _connect(self) -> None:
        try:
            device = self.device_var.get().strip() or "Dev3"
            mode = self.device_mode_var.get().strip().lower() or "real"
            self._start_daq_worker(device=device, mode=mode)
            self._daq_connected = True
            self._daq_device = device
            self._daq_mode = mode
            self.status_var.set(f"Connected: {device} ({mode})")
            self.connect_btn.configure(state=tk.DISABLED)
            self.disconnect_btn.configure(state=tk.NORMAL)
        except Exception as e:
            self._daq_connected = False
            self._daq_device = None
            messagebox.showerror("Connect failed", str(e))

    def _disconnect(self) -> None:
        try:
            self._stop_sequence()
        except Exception:
            pass

        try:
            if self._daq_connected:
                try:
                    self._daq_request({"cmd": "set_do", "value": ALL_OFF}, timeout=2.0)
                except Exception:
                    pass
            self._stop_daq_worker()
        except Exception:
            pass

        self._daq_connected = False
        self._daq_device = None
        self.status_var.set("Disconnected")
        self.connect_btn.configure(state=tk.NORMAL)
        self.disconnect_btn.configure(state=tk.DISABLED)

    def _start_daq_worker(self, *, device: str, mode: str) -> None:
        self._stop_daq_worker()

        cmd_q: Queue = Queue()
        resp_q: Queue = Queue()
        worker = daq_worker_dry_main if mode == "dry" else daq_worker_mpq_main
        proc = Process(target=worker, args=(cmd_q, resp_q, {"device": device, "mode": mode}), daemon=True)
        proc.start()

        ready = resp_q.get(timeout=8)
        if not ready.get("ok"):
            raise RuntimeError(f"DAQ worker failed: {ready}")

        self._daq_proc = proc
        self._daq_cmd_q = cmd_q
        self._daq_resp_q = resp_q

    def _stop_daq_worker(self) -> None:
        try:
            if self._daq_cmd_q is not None:
                self._daq_cmd_q.put({"cmd": "close"})
        except Exception:
            pass

        try:
            if self._daq_proc is not None and self._daq_proc.is_alive():
                self._daq_proc.terminate()
        except Exception:
            pass

        self._daq_proc = None
        self._daq_cmd_q = None
        self._daq_resp_q = None
        self._daq_connected = False

    def _daq_request(self, cmd: dict, timeout: float = 5.0) -> dict:
        if not self._daq_connected or self._daq_cmd_q is None or self._daq_resp_q is None:
            raise RuntimeError("Not connected")
        self._daq_cmd_q.put(cmd)
        resp = self._daq_resp_q.get(timeout=timeout)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Invalid DAQ response: {resp!r}")
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "DAQ error"))
        return resp

    def _all_off(self) -> None:
        try:
            self._require_connected()
            self._daq_request({"cmd": "set_do", "value": ALL_OFF})
        except Exception as e:
            messagebox.showerror("DO error", str(e))

    def _apply_manual(self) -> None:
        try:
            self._require_connected()
            value = 0
            if self.v_397.get():
                value |= NM_397
            if self.v_397s.get():
                value |= NM_397_SIG
            if self.v_729.get():
                value |= NM_729
            if self.v_854.get():
                value |= NM_854
            self._daq_request({"cmd": "set_do", "value": int(value)})
        except Exception as e:
            messagebox.showerror("Manual apply error", str(e))

    def _parse_sequence_text(self) -> list[tuple[int, float]]:
        name_to_value = {
            "ALL_OFF": ALL_OFF,
            "NM_397": NM_397,
            "NM_397_SIG": NM_397_SIG,
            "NM_729": NM_729,
            "NM_854": NM_854,
            "NM_729_854": (NM_729 | NM_854),
        }

        raw = self.seq_text.get("1.0", tk.END)
        steps: list[tuple[int, float]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid sequence line: {line!r}")
            key = parts[0]
            hold_s = float(parts[1])
            if hold_s < 0:
                raise ValueError(f"hold_s must be >= 0: {line!r}")

            # Preferred: bitstring like 0101 (length=4)
            if all(ch in "01" for ch in key):
                if len(key) != SEQUENCE_BITS:
                    raise ValueError(
                        f"Bitstring must be {SEQUENCE_BITS} digits ({BITSTRING_HELP}): {line!r}"
                    )
                value = int(key, 2)
            elif key in name_to_value:
                value = name_to_value[key]
            else:
                value = int(key, 0)

            if not (0 <= value <= 0b1111):
                raise ValueError(
                    f"DO value must be 0..15 (4-bit, port0/line4:7): {line!r}"
                )

            steps.append((int(value), float(hold_s)))

        if not steps:
            raise ValueError("Sequence is empty")
        return steps

    def _start_sequence(self) -> None:
        try:
            self._require_connected()
            insert_index = int(self.insert_index_var.get())
            width_ms = float(self.width_var.get())
            do_sequence = self._parse_sequence_text()
        except Exception as e:
            messagebox.showerror("Sequence", str(e))
            return

        self._seq_running = True
        self._seq_thread = threading.Thread(
            target=self._sequence_loop, args=(do_sequence, insert_index, width_ms), daemon=True
        )
        self._seq_thread.start()

        self.status_var.set(f"Connected: {self._daq_device} ({self._daq_mode}) | Sequence running")
        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)

    def _sequence_stopped_ui(self) -> None:
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        if self._daq_connected:
            self.status_var.set(f"Connected: {self._daq_device} ({self._daq_mode})")

    def _stop_sequence(self) -> None:
        self._seq_running = False
        try:
            if self._seq_thread is not None:
                self._seq_thread.join(timeout=0.5)
        except Exception:
            pass
        self._sequence_stopped_ui()

    def _sequence_loop(self, do_sequence: list[tuple[int, float]], insert_index: int, width_ms: float) -> None:
        try:
            while self._seq_running:
                self._daq_request(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": do_sequence,
                        "insert_index": int(insert_index),
                        "ao_width_ms": float(width_ms),
                        "ao_rate_hz": AO_RATE_HZ,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )
        except Exception as e:
            err = str(e)
            self.after(0, lambda msg=err: messagebox.showerror("Sequence", msg))
        finally:
            self._seq_running = False
            self.after(0, self._sequence_stopped_ui)

    def _on_close(self) -> None:
        try:
            self._stop_sweep(clean_only=True)
        except Exception:
            pass
        self._disconnect()
        try:
            self._disconnect_fg()
        except Exception:
            pass
        self.destroy()

    # ---------------- Sweep runtime (queue-based) ----------------
    def _start_sweep(self) -> None:
        if self._sw_running:
            return

        try:
            freq_start = float(eval(self.sw_freq_start.get()))
            freq_stop = float(eval(self.sw_freq_stop.get()))
            freq_step = float(eval(self.sw_freq_step.get()))
            if freq_step == 0:
                raise ValueError("freq_step must be non-zero")
            freqs: list[float] = []
            f = freq_start
            if freq_step > 0:
                while f <= freq_stop + 1e-12:
                    freqs.append(float(f))
                    f += freq_step
            else:
                while f >= freq_stop - 1e-12:
                    freqs.append(float(f))
                    f += freq_step
            if not freqs:
                raise ValueError("No frequencies generated")

            seq_path = Path(self.sw_seq_path.get())
            seq_data = json.loads(seq_path.read_text(encoding="utf-8"))
            do_sequence = self._parse_sequence_text_from_raw(seq_data.get("sequence_text", ""))
            insert_index = int(seq_data.get("ao_insert_index", -1))
            ao_width_ms = float(seq_data.get("ao_width_ms", 1.0))

            n_target = int(self.sw_n_target.get())
            max_attempt = int(self.sw_max_attempt.get())
            settle_s = float(self.sw_settle_s.get())
            update_interval = max(0.2, float(self.sw_update_interval.get()))

            daq_mode = self.sw_daq_mode.get()
            cam_mode = self.sw_cam_mode.get()
            device = self.sw_device.get().strip() or "Dev3"
            visa_res = self.sw_visa.get().strip()
            no_fg = bool(self.sw_no_fg.get())
            dry_image_dir = self.dry_image_dir_var.get().strip()

        except Exception as e:
            messagebox.showerror("Sweep", str(e))
            return

        # disable controls
        self._sw_running = True
        self._toggle_sweep_controls(False)
        self.sw_status.set("Starting...")

        # setup output dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/output/spectrum") / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        self._sw_out_dir = out_dir

        # write config
        cfg = {
            "freqs": freqs,
            "n_target": n_target,
            "max_attempt": max_attempt,
            "settle_s": settle_s,
            "daq_mode": daq_mode,
            "device": device,
            "sequence_json": str(seq_path),
            "insert_index": insert_index,
            "ao_width_ms": ao_width_ms,
            "camera_mode": cam_mode,
            "dry_image_dir": dry_image_dir,
            "roi_bootstrap": {
                "pulse_s": ROI_PULSE_S,
                "idle_s": ROI_IDLE_S,
                "max_attempt": ROI_MAX_ATTEMPT,
            },
        }
        (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        # start workers
        daq_cmd_q: Queue = Queue()
        daq_resp_q: Queue = Queue()
        cam_cmd_q: Queue = Queue()
        cam_resp_q: Queue = Queue()
        self._sw_queues = {"daq_cmd": daq_cmd_q, "daq_resp": daq_resp_q, "cam_cmd": cam_cmd_q, "cam_resp": cam_resp_q}

        if daq_mode == "dry":
            from src.shutter_camera_trigger.daq_worker_dry import daq_worker_dry_main as daq_worker_main
        else:
            from src.shutter_camera_trigger.daq_worker_mpq import daq_worker_mpq_main as daq_worker_main

        from src.camera.ion_state_worker import ion_state_worker_main

        daq_p = Process(target=daq_worker_main, args=(daq_cmd_q, daq_resp_q, {"device": device, "mode": daq_mode}), daemon=True)
        cam_cfg: dict[str, Any] = {
            "mode": cam_mode,
            "exposure_s": 0.001,
            "frame_timeout_s": 1.0,
            "bootstrap_n": 10,
        }
        if dry_image_dir:
            cam_cfg["dry_image_dir"] = dry_image_dir
        cam_p = Process(target=ion_state_worker_main, args=(cam_cmd_q, cam_resp_q, cam_cfg), daemon=True)
        daq_p.start()
        cam_p.start()
        self._sw_procs = [daq_p, cam_p]

        # wait ready
        try:
            daq_ready = daq_resp_q.get(timeout=5)
            cam_ready = cam_resp_q.get(timeout=15)
            if not daq_ready.get("ok"):
                raise RuntimeError(f"DAQ worker failed: {daq_ready}")
            if not cam_ready.get("ok"):
                raise RuntimeError(f"Camera worker failed: {cam_ready}")
        except Exception as e:
            messagebox.showerror("Sweep", f"Worker init failed: {e}")
            self._stop_sweep(clean_only=True)
            return

        # ROI bootstrap: fire camera trigger TTL without 729 and wait for a valid camera reply.
        self.sw_status.set("ROI bootstrap...")
        self.update_idletasks()
        roi_ok = self._run_roi_bootstrap(daq_cmd_q, daq_resp_q, cam_cmd_q, cam_resp_q)
        if not roi_ok:
            messagebox.showerror("Sweep", "ROI bootstrap failed")
            self._stop_sweep(clean_only=True)
            return

        # FG
        rig = None
        rig_owned = False
        if not no_fg:
            if self._fg_connected and self._fg_handle is not None:
                rig = self._fg_handle
                try:
                    rig.output(True)
                except Exception:
                    pass
            elif visa_res:
                try:
                    from src.lib.instruments.rigol_dg import RigolDG, RigolDgConfig

                    rig = RigolDG(RigolDgConfig(visa_resource=visa_res, channel=1, timeout_ms=5000))
                    rig.open()
                    rig.output(True)
                    rig_owned = True
                except Exception as e:
                    messagebox.showwarning("FG", f"FG init failed, continuing without FG: {e}")
                    rig = None

        # open CSVs
        shots_path = out_dir / "shots.csv"
        spec_path = out_dir / "spectrum.csv"
        self._sw_freqs = freqs
        self._sw_results = []
        self._sw_next_update = time.time() + update_interval

        try:
            with shots_path.open("w", newline="", encoding="utf-8") as f_shots, spec_path.open(
                "w", newline="", encoding="utf-8"
            ) as f_spec:
                shots_writer = csv.DictWriter(
                    f_shots,
                    fieldnames=[
                        "t_iso",
                        "step_idx",
                        "freq_hz",
                        "attempt_idx",
                        "processed_idx",
                        "bright",
                        "S_norm",
                        "tau_on",
                        "tau_off",
                        "cam_event",
                    ],
                )
                shots_writer.writeheader()

                spec_writer = csv.DictWriter(f_spec, fieldnames=["step_idx", "freq_hz", "n_processed", "n_bright", "p_bright"])
                spec_writer.writeheader()

                for step_idx, freq in enumerate(freqs):
                    processed = 0
                    n_bright = 0

                    if rig is not None:
                        try:
                            rig.set_frequency_hz(freq)
                            time.sleep(max(0.0, settle_s))
                        except Exception:
                            pass

                    for attempt_idx in range(max_attempt):
                        if not self._sw_running:
                            break
                        if processed >= n_target:
                            break

                        cam_cmd_q.put({"cmd": "get_state", "timeout_s": 1.0})
                        daq_cmd_q.put(
                            {
                                "cmd": "run_sequence_once",
                                "do_sequence": do_sequence,
                                "insert_index": int(insert_index),
                                "ao_width_ms": float(ao_width_ms),
                                "ao_rate_hz": AO_RATE_HZ,
                                "ao_v_high": 5.0,
                                "ao_v_low": 0.0,
                            }
                        )

                        daq_resp = daq_resp_q.get(timeout=5)
                        if not daq_resp.get("ok"):
                            raise RuntimeError(f"DAQ error: {daq_resp}")
                        cam_resp = cam_resp_q.get(timeout=5)
                        if not cam_resp.get("ok"):
                            continue

                        bright = bool(cam_resp.get("bright"))
                        s_norm = cam_resp.get("S_norm")
                        tau_on = cam_resp.get("tau_on")
                        tau_off = cam_resp.get("tau_off")

                        processed += 1
                        if bright:
                            n_bright += 1

                        shots_writer.writerow(
                            {
                                "t_iso": datetime.now().isoformat(timespec="milliseconds"),
                                "step_idx": step_idx,
                                "freq_hz": float(freq),
                                "attempt_idx": attempt_idx,
                                "processed_idx": processed,
                                "bright": int(bright),
                                "S_norm": "" if s_norm is None else float(s_norm),
                                "tau_on": "" if tau_on is None else float(tau_on),
                                "tau_off": "" if tau_off is None else float(tau_off),
                                "cam_event": str(cam_resp.get("event")),
                            }
                        )

                        # periodic UI update (lightweight)
                        now = time.time()
                        if now >= self._sw_next_update:
                            self._sw_next_update = now + update_interval
                            self._update_sw_plot(step_idx, freq, processed, n_bright)
                            self.sw_status.set(
                                f"Running: step {step_idx+1}/{len(freqs)} freq={freq:.3e} Hz proc={processed}/{n_target}"
                            )
                            self.update_idletasks()

                    p_bright = (n_bright / processed) if processed > 0 else 0.0
                    spec_writer.writerow(
                        {
                            "step_idx": step_idx,
                            "freq_hz": float(freq),
                            "n_processed": processed,
                            "n_bright": n_bright,
                            "p_bright": float(p_bright),
                        }
                    )
                    self._sw_results.append((float(freq), processed, n_bright))
                    self._update_sw_plot(step_idx, freq, processed, n_bright)
                    self.sw_status.set(f"Done step {step_idx+1}/{len(freqs)}")
                    self.update_idletasks()

        except Exception as e:
            messagebox.showerror("Sweep", str(e))

        finally:
            if rig is not None:
                try:
                    rig.output(False)
                except Exception:
                    pass
                if rig_owned:
                    try:
                        rig.close()
                    except Exception:
                        pass
            self._stop_sweep(clean_only=True)

    def _stop_sweep(self, clean_only: bool = False) -> None:
        # signal stop
        self._sw_running = False
        # tell workers to close
        try:
            if self._sw_queues.get("daq_cmd"):
                self._sw_queues["daq_cmd"].put({"cmd": "close"})
            if self._sw_queues.get("cam_cmd"):
                self._sw_queues["cam_cmd"].put({"cmd": "close"})
        except Exception:
            pass
        time.sleep(0.2)
        for p in self._sw_procs:
            try:
                if p.is_alive():
                    p.terminate()
            except Exception:
                pass
        self._sw_procs = []
        self._toggle_sweep_controls(True)
        if not clean_only:
            self.sw_status.set("Stopped")
        else:
            self.sw_status.set("Idle")

        # save final plot
        if self._sw_out_dir and self.sw_fig is not None:
            try:
                self.sw_fig.savefig(self._sw_out_dir / "spectrum.png", dpi=120)
            except Exception:
                pass

    def _toggle_sweep_controls(self, enable: bool) -> None:
        self.sw_start_btn.configure(state=(tk.NORMAL if enable else tk.DISABLED))
        self.sw_stop_btn.configure(state=(tk.DISABLED if enable else tk.NORMAL))

        child_state = "!disabled" if enable else "disabled"
        for child in self.sweep_tab.winfo_children():
            if child is self.sw_stop_btn and not enable:
                continue
            try:
                child.state([child_state])
            except Exception:
                try:
                    child.configure(state=("normal" if enable else "disabled"))
                except Exception:
                    pass

    def _parse_sequence_text_from_raw(self, raw: str) -> list[tuple[int, float]]:
        steps: list[tuple[int, float]] = []
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Invalid sequence line: {line!r}")
            key = parts[0]
            hold_s = float(parts[1])
            if hold_s < 0:
                raise ValueError(f"hold_s must be >= 0: {line!r}")
            if all(ch in "01" for ch in key):
                value = int(key, 2)
            else:
                value = int(key, 0)
            if not (0 <= value <= 0b1111):
                raise ValueError(f"DO value must be 0..15: {line!r}")
            steps.append((int(value), float(hold_s)))
        if not steps:
            raise ValueError("Sequence is empty")
        return steps

    def _update_sw_plot(self, step_idx: int, freq: float, processed: int, n_bright: int) -> None:
        if self.sw_ax is None or self.sw_canvas is None:
            return
        if processed <= 0:
            return

        # Append/replace current freq point
        updated = False
        for i, (f, _, _) in enumerate(self._sw_results):
            if abs(f - freq) < 1e-9:
                self._sw_results[i] = (f, processed, n_bright)
                updated = True
                break
        if not updated:
            self._sw_results.append((freq, processed, n_bright))

        xs = [f for f, _, _ in self._sw_results]
        ys = [nb / n if n > 0 else 0.0 for f, n, nb in self._sw_results]

        self.sw_ax.clear()
        self.sw_ax.plot(xs, ys, marker="o")
        self.sw_ax.set_xlabel("freq (Hz)")
        self.sw_ax.set_ylabel("p_bright")
        self.sw_ax.grid(True, alpha=0.3)
        self.sw_fig.tight_layout()
        self.sw_canvas.draw()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
