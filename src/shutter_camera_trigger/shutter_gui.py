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
import queue
import subprocess
import threading
import time
import os
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

import numpy as np

from .daq_worker_dry import daq_worker_dry_main
from .daq_worker_mpq import daq_worker_mpq_main

# -------------------------
# DO bit mapping (port1/line0:3)
# bit0=line0, bit1=line1, bit2=line2, bit3=line3
# -------------------------
ALL_OFF = 0b0000

NM_397 = 0b0001  # line0
NM_397_SIG = 0b0010  # line1
# bit2 (line2) is used as Camera Trigger (DO)
CAMERA_TRIGGER = 0b0100  # line2
NM_729 = CAMERA_TRIGGER  # backward-compatible alias (was 729 shutter)
NM_854 = 0b1000  # line3

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
DEFAULT_FG_AMP_VPP = 0.790
DEFAULT_FG_OFFSET_V = 0.0
DEFAULT_FG_START_HZ = 1_000.0
DEFAULT_FG_STOP_HZ = 10_000.0
DEFAULT_FG_TIME_S = 1.0
DEFAULT_FG_RESOURCE = "USB0::0x1AB1::0x0646::DG9R273500535::INSTR"
FG_AMP_MAX_MVPP = 810.0

# ROI bootstrap (pre-sequence camera TTL check)
ROI_PULSE_S = 0.002
ROI_IDLE_S = 0.002
ROI_MAX_ATTEMPT = 5

# UI
DEFAULT_UI_FONT_SIZE = 12
DEFAULT_DAQ_DEVICE = "Dev1"

# Persist GUI camera-trigger preferences (no manual re-entry on next launch)
GUI_PREFS_PATH = Path("config") / "shutter_gui_prefs.json"

# Persist last worker PIDs so we can clean up after crashes (best-effort).
WORKER_PIDS_PATH = Path("config") / "last_worker_pids.json"


# -------------------------
# Helper: limit BLAS threads (for online run jitter削減)
# -------------------------
def _limit_blas_threads() -> None:
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _format_worker_failure(resp: Any, *, label: str, log_path: str | None = None) -> str:
    """Format worker failure dicts into a human-friendly message.

    The camera/DAQ workers may return large dicts with tracebacks; showing them
    verbatim in a messagebox is noisy. Keep UI actionable.
    """
    msg = ""
    if isinstance(resp, dict):
        event = str(resp.get("event") or "").strip()
        err = resp.get("error")
        if err is None:
            # Best-effort fallback
            err = resp.get("msg") or resp.get("message") or resp
        msg = str(err)
        if event:
            msg = f"{label}: {msg} (event={event})"
        else:
            msg = f"{label}: {msg}"

        # Common actionable hint for Hamamatsu DCAM
        if "NOCAMERA" in msg or "No camera detected" in msg:
            msg += (
                "\n\nDCAM がカメラを検出できていません。\n"
                "- カメラの電源/接続(USB/CameraLink等)\n"
                "- Hamamatsu/DCAM-API ドライバの導入\n"
                "- 他アプリがカメラを掴んでいないか\n"
                "を確認してください。カメラ無しPCで試す場合は Camera mode を dry にしてください。"
            )
    else:
        msg = f"{label}: {resp}"

    if log_path:
        try:
            lp = str(log_path).strip()
        except Exception:
            lp = ""
        if lp:
            msg += f"\n\nLog: {lp}"
    return msg


def _robust_gray_limits(img: Any, *, lo_pct: float = 1.0, hi_pct: float = 99.0) -> tuple[float | None, float | None]:
    """Return (vmin, vmax) for grayscale imshow using robust percentiles.

    This prevents a few hot/saturated pixels from compressing contrast.
    """
    try:
        arr = np.asarray(img)
        if arr.size == 0:
            return (None, None)
        a = np.asarray(arr, dtype=float)
        if not np.isfinite(a).any():
            return (None, None)
        vmin = float(np.nanpercentile(a, float(lo_pct)))
        vmax = float(np.nanpercentile(a, float(hi_pct)))
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            return (None, None)
        if vmax <= vmin:
            return (None, None)
        return (vmin, vmax)
    except Exception:
        return (None, None)

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
        # NOTE: DAQ worker uses a single response queue. Without serializing
        # request/response pairs, concurrent callers can consume each other's
        # responses and appear to hang.
        self._daq_req_lock = threading.Lock()
        self._daq_connected = False
        self._daq_device: str | None = None
        self._daq_mode: str = "real"
        self._seq_thread: threading.Thread | None = None
        self._seq_running = False
        self._seq_stop_polling = False

        self._fg_handle = None
        self._fg_resource: str | None = None
        self._fg_connected = False

        self._camera_connected = False

        # Camera tab plot state
        self.camera_tab: ttk.Frame | None = None
        self._cam_fig = None
        self._cam_ax = None
        self._cam_canvas = None
        self._cam_status = tk.StringVar(value="Idle")

        self._plot_container: ttk.Frame | None = None
        self._plot_placeholder: ttk.Label | None = None
        self._plot_fig = None
        self._plot_canvas = None

        self._build_ui()

        # Restore persisted camera trigger preferences (if any).
        try:
            self._load_camera_trigger_prefs()
        except Exception:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _get_fg_amp_vpp(self) -> float:
        """Return FG amplitude in Vpp parsed from UI (mVpp input)."""
        s = ""
        try:
            s = (self.fg_amp_mvpp_var.get() or "").strip()
            if not s:
                return float(DEFAULT_FG_AMP_VPP)
            mvpp = float(s)
        except Exception as e:
            raise ValueError(f"Invalid FG amplitude (mVpp): {s!r}") from e

        if mvpp <= 0:
            raise ValueError("FG amplitude (mVpp) must be > 0")

        if mvpp > FG_AMP_MAX_MVPP:
            try:
                messagebox.showwarning(
                    "FG",
                    f"FG amp is limited to {FG_AMP_MAX_MVPP:.0f} mVpp. Setting to max.",
                )
            except Exception:
                pass
            try:
                self.fg_amp_mvpp_var.set(str(int(FG_AMP_MAX_MVPP)))
            except Exception:
                pass
            mvpp = float(FG_AMP_MAX_MVPP)
        return mvpp / 1000.0

    def _get_camera_exposure_s(self) -> float:
        """Return camera exposure in seconds parsed from UI (ms input)."""
        s = ""
        try:
            s = (self.camera_exposure_ms_var.get() or "").strip()
            if not s:
                return 0.001
            ms = float(s)
        except Exception as e:
            raise ValueError(f"Invalid exposure (ms): {s!r}") from e

        if ms <= 0:
            raise ValueError("Exposure (ms) must be > 0")
        return ms / 1000.0

    def _camera_subarray_from_ui(self) -> tuple[int, int, int, int] | None:
        """Return subarray as ROI tuple (xw,yw,xs,ys) or None if disabled."""
        try:
            enabled = bool(getattr(self, "camera_subarray_enable_var").get())
        except Exception:
            enabled = False
        if not enabled:
            return None

        def _get_int(var_name: str, label: str) -> int:
            v = (getattr(self, var_name).get() or "").strip()
            if not v:
                raise ValueError(f"Subarray {label} is empty")
            try:
                return int(float(v))
            except Exception as e:
                raise ValueError(f"Invalid subarray {label}: {v!r}") from e

        xs = _get_int("camera_sub_x_var", "X")
        ys = _get_int("camera_sub_y_var", "Y")
        xw = _get_int("camera_sub_w_var", "Width")
        yw = _get_int("camera_sub_h_var", "Height")

        if xs < 0 or ys < 0:
            raise ValueError("Subarray X/Y must be >= 0")
        if xw <= 0 or yw <= 0:
            raise ValueError("Subarray Width/Height must be > 0")
        return (int(xw), int(yw), int(xs), int(ys))

    def _apply_subarray_to_cam_cfg(self, cfg: dict[str, Any]) -> None:
        sub = self._camera_subarray_from_ui()
        if sub is None:
            return
        cfg["subarray"] = [int(sub[0]), int(sub[1]), int(sub[2]), int(sub[3])]

    def _camera_trigger_cfg_from_ui(self) -> dict[str, Any]:
        delay_s_raw = (self.camera_trigger_delay_s_var.get() or "").strip()
        delay_s: float | None = None
        if delay_s_raw:
            try:
                delay_s = float(delay_s_raw)
            except Exception:
                raise ValueError(f"Invalid trigger delay (s): {delay_s_raw!r}")

        cfg: dict[str, Any] = {
            "source": (self.camera_trigger_source_var.get() or "EXTERNAL").strip().upper() or "EXTERNAL",
            "connector": (self.camera_trigger_connector_var.get() or "BNC").strip().upper() or "BNC",
            "polarity": (self.camera_trigger_polarity_var.get() or "POSITIVE").strip().upper() or "POSITIVE",
            "active": (self.camera_trigger_active_var.get() or "EDGE").strip().upper() or "EDGE",
            "mode": (self.camera_trigger_mode_var.get() or "NORMAL").strip().upper() or "NORMAL",
        }
        if delay_s is not None:
            cfg["delay_s"] = float(delay_s)
        return cfg

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

    def _prefs_path(self) -> Path:
        # Store under repo's config/ by default.
        try:
            root = Path(__file__).resolve().parents[2]
            return root / GUI_PREFS_PATH
        except Exception:
            return GUI_PREFS_PATH

    def _load_camera_trigger_prefs(self) -> None:
        p = self._prefs_path()
        if not p.exists():
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        trig = data.get("camera_trigger")
        if not isinstance(trig, dict):
            return

        def _set_str(var: tk.StringVar, key: str) -> None:
            v = trig.get(key)
            if v is None:
                return
            s = str(v).strip()
            if s:
                var.set(s)

        _set_str(self.camera_trigger_source_var, "source")
        _set_str(self.camera_trigger_connector_var, "connector")
        _set_str(self.camera_trigger_polarity_var, "polarity")
        _set_str(self.camera_trigger_active_var, "active")
        _set_str(self.camera_trigger_mode_var, "mode")
        _set_str(self.camera_trigger_delay_s_var, "delay_s")

        try:
            self.camera_verbose_var.set(bool(trig.get("verbose") or False))
        except Exception:
            pass

        # Optional: camera subarray
        sub = data.get("camera_subarray")
        if isinstance(sub, dict):
            try:
                self.camera_subarray_enable_var.set(bool(sub.get("enabled") or False))
            except Exception:
                pass
            for key, var in (
                ("x", self.camera_sub_x_var),
                ("y", self.camera_sub_y_var),
                ("width", self.camera_sub_w_var),
                ("height", self.camera_sub_h_var),
            ):
                try:
                    v = sub.get(key)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s:
                        var.set(s)
                except Exception:
                    pass

    def _save_camera_trigger_prefs(self) -> None:
        p = self._prefs_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        trig = {
            "source": (self.camera_trigger_source_var.get() or "").strip(),
            "connector": (self.camera_trigger_connector_var.get() or "").strip(),
            "polarity": (self.camera_trigger_polarity_var.get() or "").strip(),
            "active": (self.camera_trigger_active_var.get() or "").strip(),
            "mode": (self.camera_trigger_mode_var.get() or "").strip(),
            "delay_s": (self.camera_trigger_delay_s_var.get() or "").strip(),
            "verbose": bool(self.camera_verbose_var.get()),
        }

        sub = {
            "enabled": bool(self.camera_subarray_enable_var.get()),
            "x": (self.camera_sub_x_var.get() or "").strip(),
            "y": (self.camera_sub_y_var.get() or "").strip(),
            "width": (self.camera_sub_w_var.get() or "").strip(),
            "height": (self.camera_sub_h_var.get() or "").strip(),
        }
        p.write_text(json.dumps({"camera_trigger": trig, "camera_subarray": sub}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_ui(self) -> None:
        # Menu bar (Help -> open usage doc)
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Open usage doc", command=self._open_usage_doc)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # Shared vars
        self.fg_resource_var = tk.StringVar(value=DEFAULT_FG_RESOURCE)
        self.fg_amp_mvpp_var = tk.StringVar(value=str(int(DEFAULT_FG_AMP_VPP * 1000)))
        self.camera_mode_top_var = tk.StringVar(value="dry")
        self.camera_exposure_ms_var = tk.StringVar(value="100.0")
        self.dry_image_dir_var = tk.StringVar(value="")

        # Camera trigger settings (applied automatically; users don't need to set env vars in shell)
        self.camera_trigger_source_var = tk.StringVar(
            value="EXTERNAL"
        )
        self.camera_trigger_connector_var = tk.StringVar(
            value="BNC"
        )
        self.camera_trigger_polarity_var = tk.StringVar(
            value="POSITIVE"
        )
        self.camera_trigger_active_var = tk.StringVar(
            value="EDGE"
        )
        self.camera_trigger_mode_var = tk.StringVar(
            value="NORMAL"
        )
        self.camera_trigger_delay_s_var = tk.StringVar(value="")
        self.camera_verbose_var = tk.BooleanVar(value=False)

        # Camera subarray (camera-level ROI for faster/consistent acquisition)
        self.camera_subarray_enable_var = tk.BooleanVar(value=False)
        self.camera_sub_x_var = tk.StringVar(value="0")
        self.camera_sub_y_var = tk.StringVar(value="0")
        self.camera_sub_w_var = tk.StringVar(value="")
        self.camera_sub_h_var = tk.StringVar(value="")

        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Device").grid(row=0, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value=DEFAULT_DAQ_DEVICE)
        ttk.Entry(top, textvariable=self.device_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(top, text="DAQ mode").grid(row=0, column=2, sticky=tk.W)
        self.device_mode_var = tk.StringVar(value="real")
        ttk.Combobox(top, textvariable=self.device_mode_var, values=["real", "dry"], width=6, state="readonly").grid(
            row=0, column=3, sticky=tk.W, padx=5
        )

        ttk.Label(top, text="AO width (ms)").grid(row=0, column=4, sticky=tk.W)
        self.width_var = tk.StringVar(value="15.0")
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

        ttk.Label(top, text="FG amp (mVpp)").grid(row=1, column=6, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.fg_amp_mvpp_var, width=10).grid(row=1, column=7, sticky=tk.W, padx=5, pady=(6, 0))

        ttk.Label(top, text="Camera mode").grid(row=2, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_mode_top_var, values=["dry", "real"], width=6, state="readonly").grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=(6, 0)
        )

        ttk.Label(top, text="Exposure (ms)").grid(row=2, column=2, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.camera_exposure_ms_var, width=10).grid(row=2, column=3, sticky=tk.W, padx=5, pady=(6, 0))

        self.cam_check_btn = ttk.Button(top, text="Camera check", command=self._check_camera_connection)
        self.cam_check_btn.grid(row=2, column=4, padx=5, pady=(6, 0))

        ttk.Label(top, text="Dry images (dry cam)").grid(row=2, column=5, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.dry_image_dir_var, width=30).grid(row=2, column=6, columnspan=2, sticky=tk.W, padx=5, pady=(6, 0))
        ttk.Button(top, text="...", width=3, command=self._browse_dry_images).grid(row=2, column=8, pady=(6, 0))

        ttk.Label(top, text="Cam trig").grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_trigger_source_var, values=["EXTERNAL", "INTERNAL"], width=9, state="readonly").grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=(6, 0)
        )
        ttk.Label(top, text="Conn").grid(row=3, column=2, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_trigger_connector_var, values=["BNC", "MULTI", "INTERFACE"], width=9, state="readonly").grid(
            row=3, column=3, sticky=tk.W, padx=5, pady=(6, 0)
        )
        ttk.Label(top, text="Pol").grid(row=3, column=4, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_trigger_polarity_var, values=["POSITIVE", "NEGATIVE"], width=9, state="readonly").grid(
            row=3, column=5, sticky=tk.W, padx=5, pady=(6, 0)
        )
        ttk.Label(top, text="Act").grid(row=3, column=6, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_trigger_active_var, values=["EDGE", "LEVEL"], width=7, state="readonly").grid(
            row=3, column=7, sticky=tk.W, padx=5, pady=(6, 0)
        )

        ttk.Label(top, text="Mode").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
        ttk.Combobox(top, textvariable=self.camera_trigger_mode_var, values=["NORMAL", "START"], width=9, state="readonly").grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=(6, 0)
        )
        ttk.Label(top, text="Delay (s)").grid(row=4, column=2, sticky=tk.W, pady=(6, 0))
        ttk.Entry(top, textvariable=self.camera_trigger_delay_s_var, width=10).grid(row=4, column=3, sticky=tk.W, padx=5, pady=(6, 0))
        ttk.Checkbutton(top, text="Cam verbose", variable=self.camera_verbose_var).grid(row=4, column=4, sticky=tk.W, padx=5, pady=(6, 0))

        # Subarray settings (applied on next camera worker start: snap/check/sweep)
        sub = ttk.LabelFrame(top, text="Subarray")
        sub.grid(row=5, column=0, columnspan=9, sticky=tk.W + tk.E, pady=(8, 0))

        ttk.Checkbutton(sub, text="Enable", variable=self.camera_subarray_enable_var).grid(row=0, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Label(sub, text="X").grid(row=0, column=1, sticky=tk.W)
        ttk.Entry(sub, textvariable=self.camera_sub_x_var, width=8).grid(row=0, column=2, sticky=tk.W, padx=(2, 10))
        ttk.Label(sub, text="Y").grid(row=0, column=3, sticky=tk.W)
        ttk.Entry(sub, textvariable=self.camera_sub_y_var, width=8).grid(row=0, column=4, sticky=tk.W, padx=(2, 10))
        ttk.Label(sub, text="W").grid(row=0, column=5, sticky=tk.W)
        ttk.Entry(sub, textvariable=self.camera_sub_w_var, width=8).grid(row=0, column=6, sticky=tk.W, padx=(2, 10))
        ttk.Label(sub, text="H").grid(row=0, column=7, sticky=tk.W)
        ttk.Entry(sub, textvariable=self.camera_sub_h_var, width=8).grid(row=0, column=8, sticky=tk.W, padx=(2, 10))

        try:
            sub.grid_columnconfigure(9, weight=1)
        except Exception:
            pass

        self.status_var = tk.StringVar(value="Disconnected")
        ttk.Label(top, textvariable=self.status_var).grid(row=6, column=0, columnspan=9, sticky=tk.W, pady=(8, 0))

        top.grid_columnconfigure(9, weight=1)

        nb = ttk.Notebook(self)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.seq_tab = ttk.Frame(nb, padding=10)
        self.manual_tab = ttk.Frame(nb, padding=10)
        self.sweep_tab = ttk.Frame(nb, padding=10)
        self.camera_tab = ttk.Frame(nb, padding=10)
        nb.add(self.seq_tab, text="Sequence")
        nb.add(self.manual_tab, text="Manual")
        nb.add(self.sweep_tab, text="Sweep")
        nb.add(self.camera_tab, text="Camera")

        self._build_sequence_tab()
        self._build_manual_tab()
        self._build_sweep_tab()
        self._build_camera_tab()

    def _build_camera_tab(self) -> None:
        if self.camera_tab is None:
            return

        top = ttk.Frame(self.camera_tab)
        top.pack(fill=tk.X, pady=(0, 8))

        ttk.Button(top, text="Snap", command=self._camera_snap).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, textvariable=self._cam_status).pack(side=tk.LEFT, padx=12)

        # Plot area
        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            self._cam_fig = Figure(figsize=(7.5, 4.6), dpi=100)
            self._cam_ax = self._cam_fig.add_subplot(111)
            self._cam_canvas = FigureCanvasTkAgg(self._cam_fig, master=self.camera_tab)
            self._cam_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception:
            self._cam_fig = None
            self._cam_ax = None
            self._cam_canvas = None
            ttk.Label(self.camera_tab, text="matplotlib not available; camera plot disabled").pack()

    def _camera_snap(self) -> None:
        """Send TTL -> acquire one frame -> save .npy -> plot (no sweep)."""
        if self._cam_ax is None or self._cam_canvas is None:
            messagebox.showerror("Camera", "matplotlib is required for plotting")
            return

        # DAQ is needed to output TTL on real hardware.
        if not self._daq_connected:
            messagebox.showerror("Camera", "DAQ is not connected. Please Connect first.")
            return

        mode = (self.camera_mode_top_var.get().strip() or "dry").lower()
        exposure_s = float(self._get_camera_exposure_s())
        trig_cfg = self._camera_trigger_cfg_from_ui()
        trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
        dry_image_dir = self.dry_image_dir_var.get().strip()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/output/camera_snap") / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        cfg: dict[str, Any] = {
            "mode": mode,
            "exposure_s": float(exposure_s),
            "frame_timeout_s": max(1.0, float(exposure_s) * 4.0 + 0.5),
            "bootstrap_n": 1,
            "trigger": dict(trig_cfg),
            "verbose": bool(self.camera_verbose_var.get()),
        }
        try:
            self._apply_subarray_to_cam_cfg(cfg)
        except Exception as e:
            messagebox.showerror("Subarray", str(e))
            return
        if dry_image_dir:
            cfg["dry_image_dir"] = dry_image_dir
        try:
            cfg["log_path"] = str(out_dir / "camera_worker.log")
        except Exception:
            pass

        # One-shot ROI-friendly trigger pulse: keep 397 ON.
        pulse_seq = [(NM_397, ROI_IDLE_S), (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S), (NM_397, ROI_IDLE_S)]

        def _worker() -> None:
            from src.camera.ion_state_worker import ion_state_worker_main

            cam_cmd_q: Queue = Queue()
            cam_resp_q: Queue = Queue()
            cam_p = Process(target=ion_state_worker_main, args=(cam_cmd_q, cam_resp_q, cfg), daemon=True)
            cam_p.start()

            def _cleanup() -> None:
                try:
                    cam_cmd_q.put({"cmd": "close"})
                except Exception:
                    pass
                try:
                    cam_p.join(timeout=3.0)
                    if cam_p.is_alive():
                        cam_p.terminate()
                        cam_p.join(timeout=1.0)
                except Exception:
                    pass

            try:
                self.after(0, lambda: self._cam_status.set("Snap: starting camera..."))

                cam_ready: dict[str, Any] | None = None
                if mode == "real" and trig_src in ("EXTERNAL", "EXT", "2", ""):
                    # Prime until ready (bootstrap waits for external triggers).
                    deadline = time.time() + 15.0
                    while time.time() < deadline:
                        try:
                            cam_ready = cam_resp_q.get_nowait()
                            break
                        except Exception:
                            pass
                        try:
                            self._daq_request(
                                {
                                    "cmd": "run_sequence_once",
                                    "do_sequence": pulse_seq,
                                    "insert_index": -1,
                                    "ao_width_ms": 0.0,
                                    "ao_rate_hz": AO_RATE_HZ,
                                    "ao_v_high": 5.0,
                                    "ao_v_low": 0.0,
                                },
                                timeout=3.0,
                            )
                        except Exception:
                            time.sleep(0.05)
                        time.sleep(0.01)

                if cam_ready is None:
                    cam_ready = cam_resp_q.get(timeout=15.0)
                if not cam_ready.get("ok"):
                    raise RuntimeError(
                        _format_worker_failure(
                            cam_ready,
                            label="Camera worker init failed",
                            log_path=str(cfg.get("log_path") or "") or None,
                        )
                    )

                # Request a frame, then fire one TTL pulse.
                cam_cmd_q.put({"cmd": "get_frame", "timeout_s": 1.0})
                try:
                    self._daq_request(
                        {
                            "cmd": "run_sequence_once",
                            "do_sequence": pulse_seq,
                            "insert_index": -1,
                            "ao_width_ms": 0.0,
                            "ao_rate_hz": AO_RATE_HZ,
                            "ao_v_high": 5.0,
                            "ao_v_low": 0.0,
                        },
                        timeout=3.0,
                    )
                except Exception:
                    # Even if TTL fails (dry), we can still try to read a frame.
                    pass

                cam_resp = cam_resp_q.get(timeout=15.0)
                if not cam_resp.get("ok"):
                    raise RuntimeError(
                        _format_worker_failure(
                            cam_resp,
                            label="Camera frame failed",
                            log_path=str(cfg.get("log_path") or "") or None,
                        )
                    )
                if cam_resp.get("event") != "frame":
                    raise RuntimeError(
                        _format_worker_failure(
                            cam_resp,
                            label="Unexpected camera response",
                            log_path=str(cfg.get("log_path") or "") or None,
                        )
                    )

                frame = np.asarray(cam_resp.get("frame"))
                npy_path = out_dir / "snap.npy"
                np.save(npy_path, frame)

                roi = cam_resp.get("roi")
                if roi is None:
                    # Best-effort ROI suggestion from the image.
                    try:
                        from src.camera.lib.analysis_profiles import generate_rois_from_image

                        rois = generate_rois_from_image(np.asarray(frame), plot=False)
                        if rois:
                            roi = list(rois[0])
                    except Exception:
                        roi = None

                def _ui_update() -> None:
                    self._cam_ax.clear()
                    vmin, vmax = _robust_gray_limits(frame)
                    self._cam_ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
                    self._cam_ax.set_title(f"snap | {mode} | saved: {npy_path}")
                    self._cam_ax.set_axis_off()

                    if isinstance(roi, (list, tuple)) and len(roi) == 4:
                        try:
                            xw, yw, xs, ys = map(int, roi)
                            from matplotlib.patches import Rectangle

                            rect = Rectangle((xs, ys), xw, yw, fill=False, edgecolor="tab:red", linewidth=2)
                            self._cam_ax.add_patch(rect)
                        except Exception:
                            pass

                    self._cam_fig.tight_layout()
                    self._cam_canvas.draw()
                    self._cam_status.set("Snap: done")

                self.after(0, _ui_update)

            except Exception as e:
                self.after(0, lambda msg=str(e): messagebox.showerror("Camera", msg))
                self.after(0, lambda: self._cam_status.set("Snap: failed"))
            finally:
                _cleanup()

        threading.Thread(target=_worker, daemon=True).start()

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
                    "ao_width_ms": 15.0,
                    "ao_insert_index": 1,
                    "sequence_text": DEFAULT_SEQUENCE_TEXT,
                },
                "minimal": {
                    "ao_width_ms": 15.0,
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
                "ao_width_ms": 15.0,
                "ao_insert_index": 1,
                "sequence_text": DEFAULT_SEQUENCE_TEXT,
            }
        if "minimal" not in seqs:
            seqs["minimal"] = {
                "ao_width_ms": 15.0,
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

        # Build intervals for each laser / trigger line.
        # bit0=line0=397, bit1=line1=397_SIG, bit2=line2=Camera trigger, bit3=line3=854
        lasers = [
            ("854 (b3, line3)", 3, "tab:blue"),
            ("397_SIG (b1, line1)", 1, "tab:blue"),
            ("397 (b0, line0)", 0, "tab:blue"),
            ("Camera trigger (b2, line2)", 2, "tab:green"),
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
        for label, bit, _color in lasers:
            intervals: list[tuple[float, float]] = []
            mask = 1 << bit
            for start_t, dur_s, do_value in segments:
                if int(do_value) & mask:
                    intervals.append((start_t, float(dur_s)))
            laser_intervals[label] = merge_intervals(intervals)

        self._plot_fig.clear()
        ax = self._plot_fig.add_subplot(111)
        bar_h = 0.8

        # Plot rows: AO (729nm) + lasers/triggers (camera trigger in green at the bottom)
        rows: list[tuple[str, list[tuple[float, float]], str]] = []
        rows.append(
            (
                f"729 nm (AO high ~{ao_high_s*1000.0:.3f} ms, total~{ao_total_s*1000.0:.3f} ms)",
                ao_intervals,
                "tab:red",
            )
        )
        for label, _bit, color in lasers:
            rows.append((label, laser_intervals[label], color))

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

        ttk.Checkbutton(self.manual_tab, text="397 (line0)", variable=self.v_397).grid(row=1, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="397 SIG (line1)", variable=self.v_397s).grid(row=2, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="Camera trigger (line2)", variable=self.v_729).grid(row=3, column=0, sticky=tk.W)
        ttk.Checkbutton(self.manual_tab, text="854 (line3)", variable=self.v_854).grid(row=4, column=0, sticky=tk.W)

        ttk.Button(self.manual_tab, text="Apply", command=self._apply_manual).grid(row=1, column=1, padx=10)
        ttk.Button(self.manual_tab, text="All Off", command=self._all_off).grid(row=2, column=1, padx=10)

        self.manual_tab.grid_columnconfigure(2, weight=1)

    # ---------------- Sweep tab (queue-based auto sweep) ----------------
    def _build_sweep_tab(self) -> None:
        _limit_blas_threads()

        row = ttk.Frame(self.sweep_tab)
        row.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(row, text="Freq start (Hz)").grid(row=0, column=0, sticky=tk.W)
        self.sw_freq_start = tk.StringVar(value="199e6")
        ttk.Entry(row, textvariable=self.sw_freq_start, width=12).grid(row=0, column=1, padx=4)

        ttk.Label(row, text="Freq stop (Hz)").grid(row=0, column=2, sticky=tk.W)
        self.sw_freq_stop = tk.StringVar(value="201e6")
        ttk.Entry(row, textvariable=self.sw_freq_stop, width=12).grid(row=0, column=3, padx=4)

        ttk.Label(row, text="Freq step (Hz)").grid(row=0, column=4, sticky=tk.W)
        self.sw_freq_step = tk.StringVar(value="0.5e6")
        ttk.Entry(row, textvariable=self.sw_freq_step, width=12).grid(row=0, column=5, padx=4)

        ttk.Label(row, text="n_target").grid(row=1, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_n_target = tk.StringVar(value="50")
        ttk.Entry(row, textvariable=self.sw_n_target, width=8).grid(row=1, column=1, padx=4, pady=(6, 0))

        ttk.Label(row, text="max_attempt").grid(row=1, column=2, sticky=tk.W, pady=(6, 0))
        self.sw_max_attempt = tk.StringVar(value="100")
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
        self.sw_device = tk.StringVar(value=DEFAULT_DAQ_DEVICE)
        ttk.Entry(row, textvariable=self.sw_device, width=10).grid(row=3, column=5, padx=4, pady=(6, 0))

        ttk.Label(row, text="FG VISA").grid(row=4, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_visa = self.fg_resource_var
        ttk.Entry(row, textvariable=self.sw_visa, width=32).grid(row=4, column=1, columnspan=3, sticky=tk.W, padx=4, pady=(6, 0))
        self.sw_no_fg = tk.BooleanVar(value=True)
        ttk.Checkbutton(row, text="No FG", variable=self.sw_no_fg).grid(row=4, column=4, columnspan=2, sticky=tk.W, pady=(6, 0))

        ttk.Label(row, text="FG amp (mVpp)").grid(row=5, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_fg_amp_mvpp = self.fg_amp_mvpp_var
        ttk.Entry(row, textvariable=self.sw_fg_amp_mvpp, width=10).grid(row=5, column=1, padx=4, pady=(6, 0))

        ttk.Label(row, text="Update interval (s)").grid(row=6, column=0, sticky=tk.W, pady=(6, 0))
        self.sw_update_interval = tk.StringVar(value="1.0")
        ttk.Entry(row, textvariable=self.sw_update_interval, width=8).grid(row=6, column=1, padx=4, pady=(6, 0))

        btn_row = ttk.Frame(self.sweep_tab)
        btn_row.pack(fill=tk.X, pady=(8, 8))
        self.sw_roi_btn = ttk.Button(btn_row, text="1) ROI check", command=self._sw_roi_check)
        self.sw_roi_btn.pack(side=tk.LEFT, padx=4)

        self.sw_thr_btn = ttk.Button(btn_row, text="2) Threshold", command=self._sw_threshold_check, state=tk.DISABLED)
        self.sw_thr_btn.pack(side=tk.LEFT, padx=4)

        self.sw_start_btn = ttk.Button(btn_row, text="3) Start spectrum", command=self._start_sweep, state=tk.DISABLED)
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
        self._sw_prepared = False
        self._sw_threshold_done = False
        self._sw_procs: list[Process] = []
        self._sw_queues: dict[str, Queue] = {}
        self._sw_freqs: list[float] = []
        self._sw_results: list[tuple[float, int, int]] = []  # (freq, n_processed, n_bright)
        self._sw_out_dir: Path | None = None
        self._sw_next_update = 0.0

        # Cached session parameters after ROI/threshold steps.
        self._sw_session: dict[str, Any] | None = None

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
                rig.set_amplitude_vpp(self._get_fg_amp_vpp())
            except Exception:
                # 振幅設定に失敗しても接続自体は継続
                pass
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
                    # Best-effort: return front panel to LOCAL control.
                    self._fg_handle.local()
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
        # Try to clean up stale workers from previous crashed runs.
        self._cleanup_stale_workers()

        mode = self.camera_mode_top_var.get().strip() or "dry"
        dry_dir = self.dry_image_dir_var.get().strip()
        exposure_s = self._get_camera_exposure_s()

        trig_cfg = self._camera_trigger_cfg_from_ui()
        trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"

        # For EXTERNAL trigger cameras, we can make this check succeed by
        # temporarily priming the camera trigger TTL via DAQ while the camera
        # worker performs bootstrap.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/output") / "camera_check" / ts
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        cfg: dict[str, Any] = {
            "mode": mode,
            "exposure_s": float(exposure_s),
            "frame_timeout_s": max(1.0, float(exposure_s) * 4.0 + 0.5),
            "bootstrap_n": 5,
            "trigger": dict(trig_cfg),
            "verbose": bool(self.camera_verbose_var.get()),
        }
        try:
            self._apply_subarray_to_cam_cfg(cfg)
        except Exception as e:
            messagebox.showerror("Subarray", str(e))
            return
        try:
            cfg["log_path"] = str(out_dir / "camera_worker.log")
        except Exception:
            pass
        if mode == "dry" and dry_dir:
            cfg["dry_image_dir"] = dry_dir

        def _worker() -> None:
            import queue as _queue
            import threading as _threading
            import time as _time

            from src.camera.ion_state_worker import ion_state_worker_main

            cmd_q: Queue = Queue()
            resp_q: Queue = Queue()
            p = Process(target=ion_state_worker_main, args=(cmd_q, resp_q, cfg), daemon=True)

            # Optional: DAQ priming (external trigger)
            prime_stop = _threading.Event()
            prime_thread: _threading.Thread | None = None

            tmp_daq_proc: Process | None = None
            tmp_daq_cmd_q: Queue | None = None
            tmp_daq_resp_q: Queue | None = None

            def _start_tmp_daq() -> tuple[Queue, Queue, Process]:
                device = self.device_var.get().strip() or DEFAULT_DAQ_DEVICE
                daq_mode = self.device_mode_var.get().strip().lower() or "real"
                if daq_mode != "real":
                    raise RuntimeError(
                        "Camera check in EXTERNAL trigger mode requires DAQ mode 'real' (to output TTL)."
                    )

                dq: Queue = Queue()
                rq: Queue = Queue()
                proc = Process(
                    target=daq_worker_mpq_main,
                    args=(dq, rq, {"device": device, "mode": daq_mode}),
                    daemon=True,
                )
                proc.start()
                ready = rq.get(timeout=8)
                if not ready.get("ok"):
                    raise RuntimeError(f"DAQ worker failed: {ready}")
                return dq, rq, proc

            def _prime_loop_using_existing() -> None:
                # Keep 397 ON while priming external-trigger camera.
                roi_sequence = [
                    (NM_397, ROI_IDLE_S),
                    (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S),
                    (NM_397, ROI_IDLE_S),
                ]
                while not prime_stop.is_set():
                    try:
                        self._daq_request(
                            {
                                "cmd": "run_sequence_once",
                                "do_sequence": roi_sequence,
                                "insert_index": -1,
                                "ao_width_ms": 0.0,
                                "ao_rate_hz": AO_RATE_HZ,
                                "ao_v_high": 5.0,
                                "ao_v_low": 0.0,
                            },
                            timeout=2.0,
                        )
                    except Exception:
                        pass
                    _time.sleep(0.01)

            def _prime_loop_using_tmp(dq: Queue, rq: Queue) -> None:
                # Keep 397 ON while priming external-trigger camera.
                roi_sequence = [
                    (NM_397, ROI_IDLE_S),
                    (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S),
                    (NM_397, ROI_IDLE_S),
                ]
                while not prime_stop.is_set():
                    try:
                        dq.put(
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
                    except Exception:
                        pass

                    # Drain a single response if available (avoid queue growth)
                    try:
                        rq.get(timeout=0.1)
                    except _queue.Empty:
                        pass
                    except Exception:
                        pass
                    _time.sleep(0.01)

            want_prime = (mode == "real") and (trig_src in ("EXTERNAL", "EXT", "2", ""))
            if want_prime:
                try:
                    if self._daq_connected:
                        prime_thread = _threading.Thread(target=_prime_loop_using_existing, daemon=True)
                        prime_thread.start()
                    else:
                        dq, rq, proc = _start_tmp_daq()
                        tmp_daq_cmd_q, tmp_daq_resp_q, tmp_daq_proc = dq, rq, proc
                        prime_thread = _threading.Thread(
                            target=_prime_loop_using_tmp,
                            args=(tmp_daq_cmd_q, tmp_daq_resp_q),
                            daemon=True,
                        )
                        prime_thread.start()
                except Exception as e:
                    # If priming cannot be started, fail fast with a helpful message.
                    def _ui_fail(msg=str(e)) -> None:
                        messagebox.showerror("Camera", f"Failed to start DAQ priming for EXTERNAL trigger.\n{msg}")

                    self.after(0, _ui_fail)
                    return

            p.start()

            try:
                self._write_last_worker_pids(
                    {
                        "t_iso": datetime.now().isoformat(timespec="seconds"),
                        "cam_pid": int(getattr(p, "pid", 0) or 0),
                    }
                )
            except Exception:
                pass

            ok = False
            ui_title = "Camera"
            ui_msg = ""
            frame_np: Any | None = None
            frame_path: str | None = None
            try:
                # External-trigger bootstrap may take longer (depends on exposure).
                ready = resp_q.get(timeout=max(15.0, float(cfg.get("bootstrap_n", 5)) * (float(exposure_s) + 0.05) + 5.0))
                if ready.get("ok"):
                    ok = True
                    dry_samples = ready.get("dry_samples")
                    extra = ""
                    if dry_samples is not None:
                        extra = f" | dry samples: {dry_samples}"
                    # Try to grab a single frame for visual confirmation.
                    try:
                        # For dry mode ROI check, prefer roi_test if available.
                        prefer = ""
                        if mode == "dry" and dry_dir:
                            try:
                                prefer = str((Path(dry_dir) / "roi_test.npy"))
                            except Exception:
                                prefer = ""
                        cmd = {"cmd": "get_frame", "timeout_s": max(2.0, float(exposure_s) * 4.0 + 0.5)}
                        if prefer:
                            cmd["prefer_sample"] = prefer
                        cmd_q.put(cmd)
                        fr = resp_q.get(timeout=10.0)
                        if isinstance(fr, dict) and fr.get("ok") and fr.get("event") == "frame":
                            frame_np = fr.get("frame")
                            try:
                                import numpy as _np

                                frame_arr = _np.asarray(frame_np)
                                frame_path = str(out_dir / "frame.npy")
                                _np.save(frame_path, frame_arr)
                            except Exception:
                                frame_path = None
                    except Exception:
                        frame_np = None
                        frame_path = None

                    if frame_path:
                        ui_msg = f"Camera check OK ({mode}){extra}\nSaved: {frame_path}"
                    else:
                        ui_msg = f"Camera check OK ({mode}){extra}"
                    ui_kind = "info"
                else:
                    ui_msg = _format_worker_failure(
                        ready,
                        label="Camera worker failed",
                        log_path=str(cfg.get("log_path") or "") or None,
                    )
                    ui_kind = "error"
            except Exception as e:
                ui_msg = _format_worker_failure(
                    e,
                    label="Camera check failed",
                    log_path=str(cfg.get("log_path") or "") or None,
                )
                ui_kind = "error"
            finally:
                prime_stop.set()
                try:
                    cmd_q.put({"cmd": "close"})
                except Exception:
                    pass
                try:
                    # Give the worker a chance to run its cleanup (camera close/uninit)
                    # before falling back to terminate.
                    p.join(timeout=3.0)
                    if p.is_alive():
                        p.terminate()
                        p.join(timeout=1.0)
                except Exception:
                    pass

                # Clear pid record after this check.
                try:
                    self._write_last_worker_pids({})
                except Exception:
                    pass

                # Close temporary DAQ (if we started one)
                try:
                    if tmp_daq_cmd_q is not None:
                        tmp_daq_cmd_q.put({"cmd": "close"})
                except Exception:
                    pass
                try:
                    if tmp_daq_proc is not None and tmp_daq_proc.is_alive():
                        tmp_daq_proc.join(timeout=2.0)
                        if tmp_daq_proc.is_alive():
                            tmp_daq_proc.terminate()
                            tmp_daq_proc.join(timeout=1.0)
                except Exception:
                    pass

            def _ui() -> None:
                self._camera_connected = ok
                # Update Camera tab plot if we captured a frame and matplotlib is available.
                try:
                    if frame_np is not None and self._cam_ax is not None and self._cam_canvas is not None and self._cam_fig is not None:
                        self._cam_ax.clear()
                        vmin, vmax = _robust_gray_limits(frame_np)
                        self._cam_ax.imshow(frame_np, cmap="gray", vmin=vmin, vmax=vmax)
                        self._cam_ax.set_title("camera_check")
                        self._cam_ax.set_axis_off()
                        self._cam_fig.tight_layout()
                        self._cam_canvas.draw()
                except Exception:
                    pass
                if ui_kind == "info":
                    messagebox.showinfo(ui_title, ui_msg)
                else:
                    messagebox.showerror(ui_title, ui_msg)

            self.after(0, _ui)

        threading.Thread(target=_worker, daemon=True).start()

    def _run_roi_bootstrap(self, daq_cmd_q: Queue, daq_resp_q: Queue, cam_cmd_q: Queue, cam_resp_q: Queue) -> bool:
        """Send simple TTL pulses (camera trigger only) until camera replies or attempts are exhausted."""
        # Keep 397 ON during ROI/bootstrap so ions remain cooled/visible.
        roi_sequence = [
            (NM_397, ROI_IDLE_S),
            (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S),
            (NM_397, ROI_IDLE_S),
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
            device = self.device_var.get().strip() or DEFAULT_DAQ_DEVICE
            mode = self.device_mode_var.get().strip().lower() or "real"
            self._start_daq_worker(device=device, mode=mode)
            self._daq_connected = True
            self._daq_device = device
            self._daq_mode = mode

            # Outside sequences, keep 397 ON by default (cooling safety).
            try:
                self._daq_request({"cmd": "set_do", "value": int(NM_397)}, timeout=2.0)
            except Exception:
                pass

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
                self._join_with_ui(self._daq_proc, timeout=2.0)
                if self._daq_proc.is_alive():
                    self._daq_proc.terminate()
                    self._join_with_ui(self._daq_proc, timeout=1.0)
        except Exception:
            pass

        self._daq_proc = None
        self._daq_cmd_q = None
        self._daq_resp_q = None
        self._daq_connected = False

    def _daq_request(self, cmd: dict, timeout: float = 5.0) -> dict:
        # Serialize to keep request/response pairing correct.
        with self._daq_req_lock:
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
            # 397 nm should normally stay ON outside sequences.
            do_all_off = False
            try:
                do_all_off = bool(
                    messagebox.askyesno(
                        "All Off",
                        "397 nm はシーケンス外では基本ON推奨です（冷却が止まります）。\n\n本当に全てOFFにしますか？",
                        parent=self,
                    )
                )
            except Exception:
                do_all_off = False
            self._daq_request({"cmd": "set_do", "value": int(ALL_OFF if do_all_off else NM_397)})
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
                raise ValueError(f"DO value must be 0..15 (4-bit, port1/line0:3): {line!r}")

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
            # Return to cooling state when not running a sequence.
            try:
                self._daq_request({"cmd": "set_do", "value": int(NM_397)}, timeout=2.0)
            except Exception:
                pass
            self.status_var.set(f"Connected: {self._daq_device} ({self._daq_mode})")

    def _stop_sequence(self) -> None:
        self._seq_running = False

        # Don't block the Tk main thread while the worker thread is still
        # waiting for a DAQ response. Poll asynchronously instead.
        if self._seq_thread is None:
            self._sequence_stopped_ui()
            return

        try:
            alive = self._seq_thread.is_alive()
        except Exception:
            alive = False

        if not alive:
            self._sequence_stopped_ui()
            return

        # Prevent starting a second sequence thread while stopping.
        try:
            self.start_btn.configure(state=tk.DISABLED)
            self.stop_btn.configure(state=tk.DISABLED)
        except Exception:
            pass

        if not self._seq_stop_polling:
            self._seq_stop_polling = True
            self.after(100, self._poll_sequence_stop)

    def _poll_sequence_stop(self) -> None:
        try:
            t = self._seq_thread
            alive = bool(t and t.is_alive())
        except Exception:
            alive = False

        if alive:
            self.after(100, self._poll_sequence_stop)
            return

        self._seq_stop_polling = False
        self._sequence_stopped_ui()

    def _sequence_loop(self, do_sequence: list[tuple[int, float]], insert_index: int, width_ms: float) -> None:
        try:
            # Expected runtime per iteration (best-effort) to avoid premature timeouts.
            est_s = 0.0
            try:
                est_s = float(sum(float(hold_s) for _, hold_s in do_sequence))
            except Exception:
                est_s = 0.0
            req_timeout = max(5.0, est_s + 2.0)

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
                    },
                    timeout=req_timeout,
                )
        except Exception as e:
            err = str(e)
            self.after(0, lambda msg=err: messagebox.showerror("Sequence", msg))
        finally:
            self._seq_running = False
            self.after(0, self._sequence_stopped_ui)

    def _on_close(self) -> None:
        try:
            self._save_camera_trigger_prefs()
        except Exception:
            pass
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

    def _ui_pump(self) -> None:
        """Process Tk events to avoid UI freeze during long operations."""
        try:
            self.update()
        except Exception:
            pass

    def _mpq_get_with_ui(self, q: Queue, timeout: float, *, label: str = "response", poll_s: float = 0.02):
        """Queue.get(timeout=...) that keeps the Tk UI responsive.

        IMPORTANT: When it times out, raise a RuntimeError with a readable message
        (queue.Empty stringifies to an empty string and is confusing in dialogs).
        """
        deadline = time.time() + float(timeout)
        while True:
            if not self._sw_running:
                raise RuntimeError("Stopped")
            try:
                return q.get_nowait()
            except queue.Empty:
                if time.time() >= deadline:
                    raise RuntimeError(f"Timeout waiting for {label} ({timeout:.1f}s)")
                self._ui_pump()
                time.sleep(poll_s)

    def _join_with_ui(self, p: Process, timeout: float, *, poll_s: float = 0.02) -> None:
        """Process.join(timeout=...) that keeps the Tk UI responsive."""
        deadline = time.time() + float(timeout)
        while True:
            try:
                if not p.is_alive():
                    return
            except Exception:
                return
            if time.time() >= deadline:
                return
            self._ui_pump()
            time.sleep(poll_s)

    def _read_last_worker_pids(self) -> dict:
        try:
            if WORKER_PIDS_PATH.exists():
                return json.loads(WORKER_PIDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _write_last_worker_pids(self, data: dict) -> None:
        try:
            WORKER_PIDS_PATH.parent.mkdir(parents=True, exist_ok=True)
            WORKER_PIDS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _get_cmdline_for_pid(self, pid: int) -> str:
        """Best-effort process command line lookup (Windows)."""
        if pid <= 0 or getattr(os, "name", "") != "nt":
            return ""
        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; if($p){{$p.CommandLine}}",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
            return (r.stdout or "").strip()
        except Exception:
            return ""

    def _taskkill_pid(self, pid: int, *, force: bool = False) -> bool:
        if pid <= 0 or getattr(os, "name", "") != "nt":
            return False
        try:
            args = ["taskkill", "/PID", str(pid), "/T"]
            if force:
                args.append("/F")
            r = subprocess.run(args, capture_output=True, text=True, timeout=4)
            return r.returncode == 0
        except Exception:
            return False

    def _cleanup_stale_workers(self) -> None:
        """Try to release camera locks left by crashed runs (best-effort).

        If a previous sweep died, its python worker might still be alive and
        holding the camera. We only kill PIDs that look like our own workers
        (guardrail: check cmdline).
        """
        try:
            data = self._read_last_worker_pids()
            pids: list[int] = []
            for k in ("cam_pid", "daq_pid"):
                try:
                    v = int(data.get(k, 0) or 0)
                except Exception:
                    v = 0
                if v > 0:
                    pids.append(v)

            if not pids:
                return

            for pid in pids:
                cmdline = (self._get_cmdline_for_pid(pid) or "").lower()
                looks_like_ours = any(m in cmdline for m in ("ion_state_worker", "daq_worker", "shutter_gui"))
                if not looks_like_ours:
                    continue

                # Try gentle termination first; escalate to /F.
                if not self._taskkill_pid(pid, force=False):
                    self._taskkill_pid(pid, force=True)

            # Clear after attempting cleanup.
            self._write_last_worker_pids({})
        except Exception:
            pass

    # ---------------- Sweep runtime (queue-based) ----------------

    def _sw_refresh_buttons(self) -> None:
        """Update Sweep tab button enabled/disabled states based on stage."""
        try:
            if not self._sw_running:
                self.sw_stop_btn.configure(state=tk.DISABLED)
                self.sw_roi_btn.configure(state=tk.NORMAL)
                self.sw_thr_btn.configure(state=(tk.NORMAL if self._sw_prepared else tk.DISABLED))
                self.sw_start_btn.configure(state=(tk.NORMAL if self._sw_threshold_done else tk.DISABLED))
            else:
                # During an active session, only Stop is always allowed.
                self.sw_stop_btn.configure(state=tk.NORMAL)
                self.sw_roi_btn.configure(state=(tk.NORMAL if self._sw_prepared else tk.DISABLED))
                self.sw_thr_btn.configure(state=(tk.NORMAL if self._sw_prepared else tk.DISABLED))
                self.sw_start_btn.configure(state=(tk.NORMAL if self._sw_threshold_done else tk.DISABLED))
        except Exception:
            pass

    def _sw_prepare_session(self) -> bool:
        """Start DAQ+camera workers and run ROI bootstrap, but do not start frequency sweep."""
        if self._sw_prepared:
            return True
        if self._sw_running:
            return False

        # If a previous run crashed, workers may still hold exclusive resources.
        self._cleanup_stale_workers()

        trig_cfg = self._camera_trigger_cfg_from_ui()

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
            ao_width_ms = float(seq_data.get("ao_width_ms", 15.0))

            n_target = int(self.sw_n_target.get())
            max_attempt = int(self.sw_max_attempt.get())
            settle_s = float(self.sw_settle_s.get())
            update_interval = max(0.2, float(self.sw_update_interval.get()))

            daq_mode = self.sw_daq_mode.get()
            cam_mode = self.sw_cam_mode.get()
            cam_exposure_s = self._get_camera_exposure_s()
            device = self.sw_device.get().strip() or DEFAULT_DAQ_DEVICE
            visa_res = self.sw_visa.get().strip()
            no_fg = bool(self.sw_no_fg.get())
            fg_amp_vpp = self._get_fg_amp_vpp()
            dry_image_dir = self.dry_image_dir_var.get().strip()

        except Exception as e:
            messagebox.showerror("Sweep", str(e))
            return False

        # Real camera requires real DAQ to provide hardware TTL triggers.
        if cam_mode == "real" and daq_mode != "real":
            messagebox.showerror("Sweep", "Camera mode is real but DAQ mode is not real. Set DAQ mode to real.")
            return False

        # disable controls and mark session running
        self._sw_running = True
        self._toggle_sweep_controls(False)
        self.sw_status.set("Starting session...")
        self._sw_refresh_buttons()

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
            "camera_exposure_s": float(cam_exposure_s),
            "fg_amp_mvpp": float(fg_amp_vpp) * 1000.0,
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
            "exposure_s": float(cam_exposure_s),
            "frame_timeout_s": max(1.0, float(cam_exposure_s) * 4.0 + 0.5),
            "bootstrap_n": 10,
            "trigger": dict(trig_cfg),
            "verbose": bool(self.camera_verbose_var.get()),
        }
        try:
            self._apply_subarray_to_cam_cfg(cam_cfg)
        except Exception as e:
            messagebox.showerror("Subarray", str(e))
            self._stop_sweep(clean_only=True)
            return False
        try:
            cam_cfg["log_path"] = str(out_dir / "camera_worker.log")
        except Exception:
            pass
        if dry_image_dir:
            cam_cfg["dry_image_dir"] = dry_image_dir
        cam_p = Process(target=ion_state_worker_main, args=(cam_cmd_q, cam_resp_q, cam_cfg), daemon=True)
        daq_p.start()
        self._sw_procs = [daq_p, cam_p]

        # wait DAQ ready first
        try:
            daq_ready = self._mpq_get_with_ui(daq_resp_q, timeout=5, label="DAQ ready")
            if not daq_ready.get("ok"):
                raise RuntimeError(f"DAQ worker failed: {daq_ready}")
        except Exception as e:
            messagebox.showerror("Sweep", f"Worker init failed ({type(e).__name__}): {e}")
            self._stop_sweep(clean_only=True)
            return False

        cam_p.start()

        # Record PIDs so we can clean up if the GUI crashes.
        try:
            self._write_last_worker_pids(
                {
                    "t_iso": datetime.now().isoformat(timespec="seconds"),
                    "daq_pid": int(getattr(daq_p, "pid", 0) or 0),
                    "cam_pid": int(getattr(cam_p, "pid", 0) or 0),
                }
            )
        except Exception:
            pass

        # Prime external-trigger camera during bootstrap.
        cam_ready: dict[str, Any] | None = None
        trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
        if cam_mode == "real" and trig_src in ("EXTERNAL", "EXT", "2", ""):
            self.sw_status.set("Camera priming...")
            self._ui_pump()
            time.sleep(0.2)

            prime_deadline = time.time() + 30.0
            prime_seq_one = [(NM_397, ROI_IDLE_S), (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S), (NM_397, ROI_IDLE_S)]

            while time.time() < prime_deadline:
                try:
                    cam_ready = cam_resp_q.get_nowait()
                    break
                except Exception:
                    pass

                try:
                    daq_cmd_q.put(
                        {
                            "cmd": "run_sequence_once",
                            "do_sequence": prime_seq_one,
                            "insert_index": -1,
                            "ao_width_ms": 0.0,
                            "ao_rate_hz": AO_RATE_HZ,
                            "ao_v_high": 5.0,
                            "ao_v_low": 0.0,
                        }
                    )
                    _ = self._mpq_get_with_ui(daq_resp_q, timeout=5, label="DAQ prime response")
                except Exception:
                    time.sleep(0.05)

                self._ui_pump()
                time.sleep(0.01)

        # wait camera ready
        try:
            if cam_ready is None:
                cam_ready = self._mpq_get_with_ui(cam_resp_q, timeout=30, label="Camera ready")
            if not cam_ready.get("ok"):
                raise RuntimeError(
                    _format_worker_failure(
                        cam_ready,
                        label="Camera worker init failed",
                        log_path=str((self._sw_out_dir / "camera_worker.log") if self._sw_out_dir else "") or None,
                    )
                )
        except Exception as e:
            messagebox.showerror("Sweep", f"Worker init failed ({type(e).__name__}): {e}")
            self._stop_sweep(clean_only=True)
            return False

        # ROI bootstrap
        self.sw_status.set("ROI bootstrap...")
        self._ui_pump()
        roi_ok = self._run_roi_bootstrap(daq_cmd_q, daq_resp_q, cam_cmd_q, cam_resp_q)
        if not roi_ok:
            messagebox.showerror("Sweep", "ROI bootstrap failed")
            self._stop_sweep(clean_only=True)
            return False

        # Cache session parameters
        self._sw_session = {
            "freqs": freqs,
            "do_sequence": do_sequence,
            "insert_index": insert_index,
            "ao_width_ms": ao_width_ms,
            "n_target": n_target,
            "max_attempt": max_attempt,
            "settle_s": settle_s,
            "update_interval": update_interval,
            "daq_mode": daq_mode,
            "cam_mode": cam_mode,
            "device": device,
            "visa_res": visa_res,
            "no_fg": no_fg,
            "fg_amp_vpp": fg_amp_vpp,
            "trig_cfg": dict(trig_cfg),
            "cam_exposure_s": float(cam_exposure_s),
            "seq_path": str(seq_path),
        }

        self._sw_prepared = True
        self._sw_threshold_done = False
        self.sw_status.set("Session ready. Step 1: ROI check.")
        self._sw_refresh_buttons()
        return True

    def _sw_roi_check(self) -> None:
        if not self._sw_prepare_session():
            return
        if self._sw_out_dir is None:
            return
        if self.sw_fig is None or self.sw_canvas is None:
            return

        daq_cmd_q = self._sw_queues.get("daq_cmd")
        daq_resp_q = self._sw_queues.get("daq_resp")
        cam_cmd_q = self._sw_queues.get("cam_cmd")
        cam_resp_q = self._sw_queues.get("cam_resp")
        if not (daq_cmd_q and daq_resp_q and cam_cmd_q and cam_resp_q):
            return

        # ROI確認: 397のみ開けて、カメラトリガを1回。
        pulse_seq = [(NM_397, ROI_IDLE_S), (NM_397 | CAMERA_TRIGGER, ROI_PULSE_S), (NM_397, ROI_IDLE_S)]

        try:
            self.sw_status.set("ROI: acquiring frame...")
            self._ui_pump()

            cam_cmd = {"cmd": "get_frame", "timeout_s": 1.0}
            try:
                if self._sw_session and self._sw_session.get("cam_mode") == "dry":
                    cam_cmd["prefer_sample"] = "data/input/dry_samples/roi_test.npy"
            except Exception:
                pass
            cam_cmd_q.put(cam_cmd)
            daq_cmd_q.put(
                {
                    "cmd": "run_sequence_once",
                    "do_sequence": pulse_seq,
                    "insert_index": -1,
                    "ao_width_ms": 0.0,
                    "ao_rate_hz": AO_RATE_HZ,
                    "ao_v_high": 5.0,
                    "ao_v_low": 0.0,
                }
            )
            _ = self._mpq_get_with_ui(daq_resp_q, timeout=5, label="DAQ ROI response")
            cam_resp = self._mpq_get_with_ui(cam_resp_q, timeout=15, label="Camera ROI frame")
            if not cam_resp.get("ok"):
                raise RuntimeError(
                    _format_worker_failure(
                        cam_resp,
                        label="Camera frame failed",
                        log_path=str((self._sw_out_dir / "camera_worker.log") if self._sw_out_dir else "") or None,
                    )
                )
            frame = np.asarray(cam_resp.get("frame"))

            # Step 1: determine ROI from this single frame (user should press while in bright state).
            roi = None
            try:
                from src.camera.lib.analysis_profiles import generate_rois_from_image
                from src.camera.lib.image_ops import crop_roi

                rois = generate_rois_from_image(np.asarray(frame), plot=False)
                best = None
                best_sum = None
                for r in rois or []:
                    if not (isinstance(r, (list, tuple)) and len(r) == 4):
                        continue
                    xw, yw, xs, ys = map(int, r)
                    crop = crop_roi(np.asarray(frame), (xw, yw, xs, ys))
                    if crop.size == 0:
                        continue
                    s = float(np.sum(crop))
                    if best_sum is None or s > best_sum:
                        best_sum = s
                        best = [int(xw), int(yw), int(xs), int(ys)]
                if best is not None:
                    roi = best
            except Exception:
                roi = None

            # Fallback to worker-provided ROI if present.
            if roi is None:
                r = cam_resp.get("roi")
                if isinstance(r, (list, tuple)) and len(r) == 4:
                    try:
                        roi = [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
                    except Exception:
                        roi = None

            # ROI check assumes the user triggers it in a bright state.
            # Do not gate ROI locking by any bright/dark heuristic here.

            if self._sw_session is not None:
                self._sw_session["roi"] = roi

            # Propagate ROI to camera worker so get_state uses the same ROI scalar as Step 2.
            try:
                cam_cmd_q.put({"cmd": "set_roi", "roi": list(roi) if roi is not None else None})
                _ = self._mpq_get_with_ui(cam_resp_q, timeout=5, label="Camera set_roi")
            except Exception:
                pass

            # Save snapshot
            try:
                np.save(self._sw_out_dir / "roi_check.npy", frame)
            except Exception:
                pass

            # Plot image only (photon distributions belong to Step 2: Threshold)
            self.sw_fig.clear()
            try:
                # 2x2 layout: image on the left (spans rows), profiles on right.
                gs = self.sw_fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
                ax_img = self.sw_fig.add_subplot(gs[:, 0])
                ax_x = self.sw_fig.add_subplot(gs[0, 1])
                ax_y = self.sw_fig.add_subplot(gs[1, 1])

                # Keep the persistent axis reference in sync after figure.clear().
                self.sw_ax = ax_img

                vmin, vmax = _robust_gray_limits(frame)
                ax_img.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
                ax_img.set_title("ROI check")
                ax_img.set_axis_off()
                if isinstance(roi, (list, tuple)) and len(roi) == 4:
                    try:
                        xw, yw, xs, ys = map(int, roi)
                        from matplotlib.patches import Rectangle

                        ax_img.add_patch(Rectangle((xs, ys), xw, yw, fill=False, edgecolor="tab:red", linewidth=2))
                    except Exception:
                        pass

                # Fit profiles and overlay curves (best-effort).
                try:
                    from src.camera.lib.analysis_profiles import lorentz_fit_profiles

                    results = lorentz_fit_profiles(np.asarray(frame), plot=False) or {}
                    horiz = results.get("horizontal") or {}
                    vert = results.get("vertical") or {}

                    # Horizontal profile (sum over y)
                    if isinstance(horiz, dict) and horiz.get("profile") is not None:
                        x_prof = np.asarray(horiz.get("profile"), dtype=float)
                        x_axis = np.asarray(horiz.get("x"), dtype=float) if horiz.get("x") is not None else np.arange(len(x_prof))
                        ax_x.plot(x_axis, x_prof, color="tab:blue", linewidth=1.0, label="profile")
                        if horiz.get("fitted") is not None:
                            ax_x.plot(x_axis, np.asarray(horiz.get("fitted"), dtype=float), color="tab:orange", linewidth=1.5, label="fit")
                        centers = horiz.get("centers")
                        fwhms = horiz.get("fwhms")
                        if isinstance(centers, (list, tuple)) and centers:
                            for i, c in enumerate(centers[:5]):
                                try:
                                    ax_x.axvline(float(c), color="tab:red", alpha=0.6, linewidth=1.0)
                                except Exception:
                                    pass
                        title = "X profile"
                        try:
                            if isinstance(fwhms, (list, tuple)) and fwhms:
                                title += f" (FWHM~{float(np.mean([float(w) for w in fwhms])):.1f}px)"
                        except Exception:
                            pass
                        ax_x.set_title(title)
                        ax_x.grid(True, alpha=0.2)
                        ax_x.tick_params(labelsize=8)
                        try:
                            ax_x.legend(fontsize=7, loc="best")
                        except Exception:
                            pass
                    else:
                        ax_x.set_title("X profile (fit failed)")
                        ax_x.set_axis_off()

                    # Vertical profile (sum over x)
                    if isinstance(vert, dict) and vert.get("profile") is not None:
                        y_prof = np.asarray(vert.get("profile"), dtype=float)
                        y_axis = np.asarray(vert.get("x"), dtype=float) if vert.get("x") is not None else np.arange(len(y_prof))
                        ax_y.plot(y_axis, y_prof, color="tab:blue", linewidth=1.0, label="profile")
                        if vert.get("fitted") is not None:
                            ax_y.plot(y_axis, np.asarray(vert.get("fitted"), dtype=float), color="tab:orange", linewidth=1.5, label="fit")
                        try:
                            yc = float(vert.get("center"))
                            ax_y.axvline(yc, color="tab:red", alpha=0.6, linewidth=1.0)
                        except Exception:
                            pass
                        title = "Y profile"
                        try:
                            if vert.get("fwhm") is not None:
                                title += f" (FWHM~{float(vert.get('fwhm')):.1f}px)"
                        except Exception:
                            pass
                        ax_y.set_title(title)
                        ax_y.grid(True, alpha=0.2)
                        ax_y.tick_params(labelsize=8)
                        try:
                            ax_y.legend(fontsize=7, loc="best")
                        except Exception:
                            pass
                    else:
                        ax_y.set_title("Y profile (fit failed)")
                        ax_y.set_axis_off()
                except Exception:
                    # If scipy/fit isn't available, just keep the image.
                    ax_x.set_title("profiles unavailable")
                    ax_x.set_axis_off()
                    ax_y.set_axis_off()

                self.sw_fig.tight_layout()
                self.sw_canvas.draw()
            except Exception:
                # Last-resort fallback: image only.
                self.sw_fig.clear()
                ax = self.sw_fig.add_subplot(111)
                self.sw_ax = ax
                vmin, vmax = _robust_gray_limits(frame)
                ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title("ROI check")
                ax.set_axis_off()
                self.sw_fig.tight_layout()
                self.sw_canvas.draw()

            if roi is None:
                self.sw_status.set("ROI: failed to detect ROI. Retry Step 1.")
            else:
                self.sw_status.set("ROI: locked. Step 2: Threshold.")
            self._sw_refresh_buttons()
        except Exception as e:
            messagebox.showerror("Sweep", str(e))

    def _sw_threshold_check(self) -> None:
        if not self._sw_prepared or not self._sw_running or not self._sw_session:
            messagebox.showerror("Sweep", "Run '1) ROI check' first.")
            return

        roi = self._sw_session.get("roi")
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            messagebox.showerror("Sweep", "ROI is not set. Run '1) ROI check' first.")
            return

        if self.sw_fig is None or self.sw_canvas is None:
            return

        daq_cmd_q = self._sw_queues.get("daq_cmd")
        daq_resp_q = self._sw_queues.get("daq_resp")
        cam_cmd_q = self._sw_queues.get("cam_cmd")
        cam_resp_q = self._sw_queues.get("cam_resp")
        if not (daq_cmd_q and daq_resp_q and cam_cmd_q and cam_resp_q):
            return

        do_sequence = self._sw_session["do_sequence"]
        # TTL sequence only (no AO, no frequency sweep)
        cal_ao_width_ms = 0.0
        cal_insert_index = -1
        n = int(self._sw_session.get("n_target") or 50)
        max_attempt = int(self._sw_session.get("max_attempt") or max(100, n))

        # Frame acquisition timeout must cover the whole shot duration.
        # If timeout is too short (e.g. fixed 1s) and the DO sequence is longer,
        # the camera worker will repeatedly return timeouts and we collect 0 samples.
        try:
            cam_exposure_s = float(self._sw_session.get("cam_exposure_s") or 0.001)
        except Exception:
            cam_exposure_s = 0.001
        seq_s = 0.0
        try:
            for step in (do_sequence or []):
                if isinstance(step, (list, tuple)) and len(step) >= 2:
                    seq_s += float(step[1])
        except Exception:
            seq_s = 0.0
        # generous margin for DAQ jitter / scheduling
        shot_timeout_s = max(1.5, float(seq_s) + float(cam_exposure_s) + 0.8)

        # Acquire frames using the selected sequence; then classify post-hoc.
        # Classification scalar S: mean value in ROI (no exposure normalization, no background subtraction).
        samples: list[float] = []
        profiles: list[np.ndarray] = []  # per-shot 1D photon-count profile (integrated over y-axis)
        last_cam_event: str | None = None
        last_cam_error: str | None = None
        cam_timeout_count = 0
        try:
            self.sw_status.set("Threshold: acquiring frames...")
            self._ui_pump()

            for attempt_idx in range(max_attempt):
                if not self._sw_running:
                    raise RuntimeError("Stopped")
                if len(samples) >= n:
                    break

                # Get a frame for this shot and compute S from the Step-1 ROI.
                cam_cmd_q.put({"cmd": "get_frame", "timeout_s": float(shot_timeout_s)})
                daq_cmd_q.put(
                    {
                        "cmd": "run_sequence_once",
                        "do_sequence": do_sequence,
                        "insert_index": int(cal_insert_index),
                        "ao_width_ms": float(cal_ao_width_ms),
                        "ao_rate_hz": AO_RATE_HZ,
                        "ao_v_high": 5.0,
                        "ao_v_low": 0.0,
                    }
                )

                daq_resp = self._mpq_get_with_ui(daq_resp_q, timeout=5, label="DAQ response")
                if not daq_resp.get("ok"):
                    raise RuntimeError(f"DAQ error: {daq_resp}")
                cam_resp = self._mpq_get_with_ui(cam_resp_q, timeout=15, label="Camera frame")
                if not cam_resp.get("ok"):
                    last_cam_event = str(cam_resp.get("event") or "") or None
                    last_cam_error = str(cam_resp.get("error") or "") or None
                    if (cam_resp.get("event") == "timeout"):
                        cam_timeout_count += 1
                    continue

                frame = np.asarray(cam_resp.get("frame"))
                from src.camera.lib.image_ops import crop_roi

                crop = crop_roi(np.asarray(frame), roi)
                if crop.size == 0:
                    continue

                try:
                    s = float(np.mean(np.asarray(crop, dtype=float)))
                    samples.append(s)
                    profiles.append(np.asarray(np.sum(np.asarray(crop, dtype=float), axis=0), dtype=float))
                except Exception:
                    continue

                if len(samples) % 10 == 0:
                    self.sw_status.set(f"Threshold: {len(samples)}/{n} frames")
                    self._ui_pump()

            if len(samples) < max(5, min(10, n)):
                detail = f"Too few samples: {len(samples)}"
                if len(samples) == 0:
                    detail += f" | seq_s~{seq_s:.3f}s exposure_s~{cam_exposure_s:.3f}s get_frame_timeout_s~{shot_timeout_s:.3f}s"
                    if cam_timeout_count:
                        detail += f" | camera_timeouts={cam_timeout_count}"
                    if last_cam_event:
                        detail += f" | last_cam_event={last_cam_event}"
                    if last_cam_error:
                        detail += f" | last_cam_error={last_cam_error}"
                raise RuntimeError(detail)

            from src.camera.lib.thresholding import quick_threshold_from_samples

            th = quick_threshold_from_samples(list(samples))
            tau = float(th["tau"])
            # Disable hysteresis: use a single threshold.
            tau_on = float(tau)
            tau_off = float(tau)

            # Post-hoc classification using tau
            bright_samples = [float(v) for v in samples if float(v) > tau]
            dark_samples = [float(v) for v in samples if float(v) <= tau]

            bright_profiles = [profiles[i] for i, v in enumerate(samples) if float(v) > tau]
            dark_profiles = [profiles[i] for i, v in enumerate(samples) if float(v) <= tau]

            # "Agreement": self-consistency metric. With hysteresis disabled, this should be ~100%.
            try:
                from src.camera.lib.thresholding import classify_hysteresis

                prev: bool | None = None
                agree = 0
                total = 0
                for v in samples:
                    v_f = float(v)
                    simple = bool(v_f > tau)
                    hys = bool(classify_hysteresis(v_f, prev_state_bright=prev, tau_on=tau_on, tau_off=tau_off))
                    prev = hys
                    agree += int(simple == hys)
                    total += 1
                acc = (float(agree) / float(total)) if total > 0 else 0.0
            except Exception:
                acc = 0.0

            # Plot:
            #  (top) photon-count distributions (integrated over y-axis), per-column samples
            #  (bottom) roi_mean distribution (the scalar used for tau)
            self.sw_fig.clear()
            ax_ph = self.sw_fig.add_subplot(211)
            ax_s = self.sw_fig.add_subplot(212)
            # Keep the persistent axis reference in sync after figure.clear().
            # Use the bottom axis as the "current" one.
            self.sw_ax = ax_s

            def _concat_profiles(ps: list[np.ndarray]) -> np.ndarray:
                arrs = []
                for p in ps:
                    a = np.asarray(p, dtype=float)
                    a = a[np.isfinite(a)]
                    if a.size:
                        arrs.append(a)
                return np.concatenate(arrs) if arrs else np.asarray([], dtype=float)

            light_counts = _concat_profiles(bright_profiles)
            dark_counts = _concat_profiles(dark_profiles)
            combined = np.concatenate([c for c in (light_counts, dark_counts) if c.size > 0])
            if combined.size == 0:
                raise RuntimeError("No valid photon-count samples")

            # NOTE: The histogram below is for 1D profiles integrated over y-axis
            # (i.e., per-column sums). The threshold tau is computed on roi_mean
            # (mean over all ROI pixels), so convert tau to this axis by scaling
            # with ROI height (yw): tau_plot ~= tau * yw.
            try:
                tau_plot = float(tau) * float(yw)
            except Exception:
                tau_plot = float(tau)

            start = int(np.floor(float(np.nanmin(combined))))
            end = int(np.ceil(float(np.nanmax(combined))))
            # Ensure the threshold line is within the plotted range.
            try:
                start = int(min(start, np.floor(float(tau_plot))))
                end = int(max(end, np.ceil(float(tau_plot))))
            except Exception:
                pass
            bin_edges = np.arange(start - 0.5, end + 1.5, 1)

            if light_counts.size > 0:
                mean_light = float(np.mean(light_counts))
                ax_ph.hist(
                    light_counts,
                    bins=bin_edges,
                    density=True,
                    alpha=0.6,
                    color="tab:orange",
                    edgecolor="none",
                    label=f"Light (mean={mean_light:.2f})",
                )
                ax_ph.axvline(mean_light, color="tab:orange", linestyle="--")
            if dark_counts.size > 0:
                mean_dark = float(np.mean(dark_counts))
                ax_ph.hist(
                    dark_counts,
                    bins=bin_edges,
                    density=True,
                    alpha=0.6,
                    color="navy",
                    edgecolor="none",
                    label=f"Dark (mean={mean_dark:.2f})",
                )
                ax_ph.axvline(mean_dark, color="navy", linestyle="--")

            # Plot threshold in the same axis unit as this histogram (per-column sum).
            try:
                ax_ph.axvline(
                    float(tau_plot),
                    color="tab:red",
                    linestyle="-",
                    linewidth=2,
                    label=f"Threshold (tau*yw={float(tau_plot):.2f})",
                )
            except Exception:
                pass

            ax_ph.set_xlabel("Photon Count (per-column sum; integer bins)")
            ax_ph.set_ylabel("Probability density")
            ax_ph.set_title(f"Photon Distribution (integrated over y-axis) | agree={acc*100:.1f}%")
            # loc="best" can be slow; use a fixed location for snappy UI.
            ax_ph.legend(loc="upper right")
            ax_ph.grid(True, alpha=0.3)

            # Bottom: roi_mean distribution used for tau
            try:
                s_all = np.asarray(samples, dtype=float)
                s_all = s_all[np.isfinite(s_all)]
            except Exception:
                s_all = np.asarray([], dtype=float)

            if s_all.size > 0:
                try:
                    s_bright = np.asarray(bright_samples, dtype=float)
                    s_dark = np.asarray(dark_samples, dtype=float)
                except Exception:
                    s_bright = np.asarray([], dtype=float)
                    s_dark = np.asarray([], dtype=float)

                try:
                    s_min = float(np.nanmin(s_all))
                    s_max = float(np.nanmax(s_all))
                    s_min = min(s_min, float(tau))
                    s_max = max(s_max, float(tau))
                    bins_s = max(10, min(80, int(np.sqrt(s_all.size)) * 4))
                    edges_s = np.linspace(s_min, s_max, bins_s + 1)
                except Exception:
                    edges_s = 50

                if s_bright.size > 0:
                    ax_s.hist(
                        s_bright,
                        bins=edges_s,
                        density=True,
                        alpha=0.6,
                        color="tab:orange",
                        edgecolor="none",
                        label=f"roi_mean bright (n={int(s_bright.size)})",
                    )
                    ax_s.axvline(float(np.mean(s_bright)), color="tab:orange", linestyle="--")
                if s_dark.size > 0:
                    ax_s.hist(
                        s_dark,
                        bins=edges_s,
                        density=True,
                        alpha=0.6,
                        color="navy",
                        edgecolor="none",
                        label=f"roi_mean dark (n={int(s_dark.size)})",
                    )
                    ax_s.axvline(float(np.mean(s_dark)), color="navy", linestyle="--")

                ax_s.axvline(float(tau), color="tab:red", linestyle="-", linewidth=2, label=f"tau={float(tau):.3g}")

            ax_s.set_xlabel("roi_mean (used for tau)")
            ax_s.set_ylabel("Probability density")
            ax_s.set_title("ROI-mean distribution")
            ax_s.legend(loc="upper right")
            ax_s.grid(True, alpha=0.3)

            self.sw_fig.tight_layout()
            self.sw_canvas.draw()

            # Save threshold info
            if self._sw_out_dir is not None:
                try:
                    (self._sw_out_dir / "threshold.json").write_text(
                        json.dumps(
                            {
                                "bright_samples_n": len(bright_samples),
                                "dark_samples_n": len(dark_samples),
                                "samples_n": int(len(samples)),
                                "roi": list(roi) if isinstance(roi, (list, tuple)) else None,
                                "sample_metric": "roi_mean",
                                "threshold": th,
                                "agreement": acc,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                except Exception:
                    pass

            # Apply to camera worker
            apply_ok = bool(
                messagebox.askyesno(
                    "Threshold",
                    f"Apply threshold?\nmode={th.get('mode')}\nagreement={acc*100:.1f}% (hysteresis OFF)\n\n metric=roi_mean\n tau={tau:.3g}",
                    parent=self,
                )
            )
            if apply_ok:
                cam_cmd_q.put({"cmd": "set_threshold", "tau_on": float(tau_on), "tau_off": float(tau_off)})
                ack = self._mpq_get_with_ui(cam_resp_q, timeout=5, label="Camera set_threshold")
                if not ack.get("ok"):
                    raise RuntimeError(f"set_threshold failed: {ack}")
                self._sw_threshold_done = True
                self.sw_status.set(f"Threshold applied. agreement={acc*100:.1f}%. Step 3: Start spectrum.")
                self._sw_refresh_buttons()
            else:
                self.sw_status.set("Threshold plotted (not applied).")

        except Exception as e:
            messagebox.showerror("Sweep", str(e))

    def _start_sweep(self) -> None:
        # Stage 3: start spectrum acquisition only after ROI + threshold confirmation.
        if not self._sw_prepared or not self._sw_threshold_done or not self._sw_session:
            messagebox.showerror("Sweep", "Run '1) ROI check' and '2) Threshold' first.")
            return

        if not self._sw_running:
            # Should not happen (session should be active), but guard anyway.
            if not self._sw_prepare_session():
                return

        # Use prepared session and workers
        freqs: list[float] = list(self._sw_session["freqs"])
        do_sequence = self._sw_session["do_sequence"]
        insert_index = int(self._sw_session["insert_index"])
        ao_width_ms = float(self._sw_session["ao_width_ms"])

        n_target = int(self._sw_session["n_target"])
        max_attempt = int(self._sw_session["max_attempt"])
        settle_s = float(self._sw_session["settle_s"])
        update_interval = float(self._sw_session["update_interval"])

        daq_cmd_q: Queue = self._sw_queues["daq_cmd"]
        daq_resp_q: Queue = self._sw_queues["daq_resp"]
        cam_cmd_q: Queue = self._sw_queues["cam_cmd"]
        cam_resp_q: Queue = self._sw_queues["cam_resp"]

        visa_res = str(self._sw_session.get("visa_res") or "")
        no_fg = bool(self._sw_session.get("no_fg"))
        fg_amp_vpp = float(self._sw_session.get("fg_amp_vpp") or self._get_fg_amp_vpp())

        # FG
        rig = None
        rig_owned = False
        if not no_fg:
            if self._fg_connected and self._fg_handle is not None:
                rig = self._fg_handle
                try:
                    try:
                        rig.set_amplitude_vpp(fg_amp_vpp)
                    except Exception:
                        pass
                    rig.output(True)
                except Exception:
                    pass
            elif visa_res:
                try:
                    from src.lib.instruments.rigol_dg import RigolDG, RigolDgConfig

                    rig = RigolDG(RigolDgConfig(visa_resource=visa_res, channel=1, timeout_ms=5000))
                    rig.open()
                    try:
                        rig.set_amplitude_vpp(fg_amp_vpp)
                    except Exception:
                        pass
                    rig.output(True)
                    rig_owned = True
                except Exception as e:
                    messagebox.showwarning("FG", f"FG init failed, continuing without FG: {e}")
                    rig = None

        # open CSVs
        out_dir = self._sw_out_dir or Path("data/output/spectrum") / datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._sw_out_dir = out_dir
        shots_path = out_dir / "shots.csv"
        spec_path = out_dir / "spectrum.csv"
        self._sw_freqs = freqs
        self._sw_results = []
        self._sw_next_update = time.time() + update_interval

        # Reset plot area for Step 3 (single axis), so it doesn't inherit Step 2 subplots.
        try:
            if self.sw_fig is not None and self.sw_canvas is not None:
                self.sw_fig.clear()
                self.sw_ax = self.sw_fig.add_subplot(111)
                self.sw_fig.tight_layout()
                self.sw_canvas.draw()
        except Exception:
            pass

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

                        daq_resp = self._mpq_get_with_ui(daq_resp_q, timeout=5, label="DAQ response")
                        if not daq_resp.get("ok"):
                            raise RuntimeError(f"DAQ error: {daq_resp}")
                        cam_resp = self._mpq_get_with_ui(cam_resp_q, timeout=5, label="Camera response")
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
                            self._ui_pump()

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
                    self._ui_pump()

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
        # Best-effort: keep 397 ON when leaving sweep.
        try:
            if self._sw_queues.get("daq_cmd"):
                self._sw_queues["daq_cmd"].put({"cmd": "set_do", "value": int(NM_397)})
        except Exception:
            pass
        # tell workers to close
        try:
            if self._sw_queues.get("daq_cmd"):
                self._sw_queues["daq_cmd"].put({"cmd": "close"})
            if self._sw_queues.get("cam_cmd"):
                self._sw_queues["cam_cmd"].put({"cmd": "close"})
        except Exception:
            pass

        # Prefer graceful shutdown so camera resources are properly released.
        # Order is [daq_p, cam_p]. Give camera longer.
        for i, p in enumerate(self._sw_procs):
            try:
                timeout = 2.0 if i == 0 else 6.0
                self._join_with_ui(p, timeout=timeout)
            except Exception:
                pass

        for p in self._sw_procs:
            try:
                if p.is_alive():
                    p.terminate()
                    self._join_with_ui(p, timeout=1.0)
            except Exception:
                pass
        self._sw_procs = []

        # Clear pid record after stopping sweep.
        try:
            self._write_last_worker_pids({})
        except Exception:
            pass

        # Reset staged-session flags
        self._sw_prepared = False
        self._sw_threshold_done = False
        self._sw_session = None

        self._toggle_sweep_controls(True)
        if not clean_only:
            self.sw_status.set("Stopped")
        else:
            self.sw_status.set("Idle")

        self._sw_refresh_buttons()

        # save final plot
        if self._sw_out_dir and self.sw_fig is not None:
            try:
                self.sw_fig.savefig(self._sw_out_dir / "spectrum.png", dpi=120)
            except Exception:
                pass

    def _toggle_sweep_controls(self, enable: bool) -> None:
        # Stage buttons
        if enable:
            try:
                self.sw_roi_btn.configure(state=tk.NORMAL)
                self.sw_thr_btn.configure(state=tk.DISABLED)
                self.sw_start_btn.configure(state=tk.DISABLED)
                self.sw_stop_btn.configure(state=tk.DISABLED)
            except Exception:
                pass
        else:
            try:
                self.sw_roi_btn.configure(state=tk.DISABLED)
                self.sw_thr_btn.configure(state=tk.DISABLED)
                self.sw_start_btn.configure(state=tk.DISABLED)
                self.sw_stop_btn.configure(state=tk.NORMAL)
            except Exception:
                pass

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
        # tight_layout() is expensive; avoid calling it on every update.
        self.sw_canvas.draw()


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
