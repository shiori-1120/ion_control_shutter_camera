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

import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from multiprocessing import Process

from .clients.daq_client import DaqClient
from .gui_support.prefs import resolve_repo_relative_path
from .gui_support.app_lifecycle import apply_default_fonts, load_camera_prefs, on_close
from .gui_support.camera_worker_manager import stop_camera_worker
from .gui_support.dialogs import pick_seq_json
from .config.device_registry import load_device_registry
from .gui_support.device_registry_ui import load_device_registry_ui, save_device_registry_ui
from .gui_support.logging_setup import init_app_logging
from .gui_support.log_panel import build_log_panel
from .gui_support.ui_state import init_ui_state
from .gui_support.validators import parse_fg_amp_vpp_safe
from .gui_support.docs import open_usage_doc
from .sweep.ui_tab import build_sweep_tab
from .sweep.ui_actions import roi_check, start_sweep, stop_sweep, threshold_check
from .gui_tabs.camera_tab import build_camera_tab, camera_check, camera_snap
from .gui_tabs.diagnostics_tab import build_diagnostics_tab
from .gui_tabs.sequence_tab import build_sequence_tab
from .gui_tabs.manual_tab import build_manual_tab
from .gui_tabs.top_bar import build_top_bar
from .daq.controller import connect_daq, disconnect_daq
from .fg.controller import connect_fg, disconnect_fg

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
DEVICE_REGISTRY_PATH = Path("config") / "device_registry.json"

# Persist last worker PIDs so we can clean up after crashes (best-effort).
WORKER_PIDS_PATH = Path("config") / "last_worker_pids.json"


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.worker_pids_path = resolve_repo_relative_path(__file__, WORKER_PIDS_PATH)
        self._prefs_path = resolve_repo_relative_path(__file__, GUI_PREFS_PATH)
        self._device_registry_path = resolve_repo_relative_path(__file__, DEVICE_REGISTRY_PATH)
        logs_root = "logs"
        try:
            registry = load_device_registry(self._device_registry_path)
            if registry.io_paths.logs_root:
                logs_root = str(registry.io_paths.logs_root)
        except Exception:
            pass
        self._log_ctx = init_app_logging(logs_root=logs_root)
        self._logger = self._log_ctx.logger
        try:
            self._logger.info("app_start")
        except Exception:
            pass
        apply_default_fonts(self, size=DEFAULT_UI_FONT_SIZE)
        self.title("Shutter/Camera Trigger")
        # Keep it reasonably sized so the embedded plot is readable.
        self.geometry("900x650")

        self._daq_proc: Process | None = None
        self._daq = DaqClient()
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
        self.diag_tab: ttk.Frame | None = None
        self.diag_info_tab: ttk.Frame | None = None
        self.setup_tab: ttk.Frame | None = None
        self.run_tab: ttk.Frame | None = None
        self._cam_fig = None
        self._cam_ax = None
        self._cam_canvas = None
        self._cam_status = tk.StringVar(value="Idle")

        self._plot_container: ttk.Frame | None = None
        self._plot_placeholder: ttk.Label | None = None
        self._plot_fig = None


        self._build_main_layout()

        # Restore persisted camera trigger preferences (if any).
        try:
            load_camera_prefs(self, prefs_path=self._prefs_path)
        except Exception:
            pass
        def _before_close() -> None:
            try:
                save_device_registry_ui(self, self._device_registry_path)
            finally:
                stop_camera_worker(self)
        self.protocol(
            "WM_DELETE_WINDOW",
            lambda: on_close(
                self,
                prefs_path=self._prefs_path,
                before_close_cb=_before_close,
                stop_sweep_cb=lambda: stop_sweep(self, clean_only=True),
                disconnect_daq_cb=lambda: disconnect_daq(self, all_off=ALL_OFF),
                disconnect_fg_cb=lambda: disconnect_fg(self),
            ),
        )
        build_log_panel(self)


    def _build_main_layout(self) -> None:
        # メニューバー
        menubar = tk.Menu(self)
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Open usage doc", command=lambda: open_usage_doc(self, source_file=__file__))
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)

        # メイン横並びフレーム
        main_frame = ttk.Frame(self)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 左: Notebook (従来のUI)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        init_ui_state(
            self,
            default_daq_device=DEFAULT_DAQ_DEVICE,
            default_fg_resource=DEFAULT_FG_RESOURCE,
            default_fg_amp_mvpp=str(int(DEFAULT_FG_AMP_VPP * 1000)),
            default_seq_path="src/shutter_camera_trigger/sequence_examples/minimal_sequence.json",
        )
        load_device_registry_ui(self, self._device_registry_path)
        nb = ttk.Notebook(left_frame)
        nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_tab = ttk.Frame(nb, padding=10)
        self.run_tab = ttk.Frame(nb, padding=10)
        self.diag_tab = ttk.Frame(nb, padding=10)
        nb.add(self.setup_tab, text="Setup")
        nb.add(self.run_tab, text="Run")
        nb.add(self.diag_tab, text="Diagnostics")

        ttk.Label(self.setup_tab, text="Setup & configuration", font=("", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        build_top_bar(
            self,
            parent=self.setup_tab,
            connect_cb=lambda: connect_daq(self, default_daq_device=DEFAULT_DAQ_DEVICE, nm_397=NM_397),
            disconnect_cb=lambda: disconnect_daq(self, all_off=ALL_OFF),
            fg_connect_cb=lambda: connect_fg(
                self,
                get_amp_vpp=lambda: parse_fg_amp_vpp_safe(
                    self,
                    max_mvpp=FG_AMP_MAX_MVPP,
                    default_vpp=DEFAULT_FG_AMP_VPP,
                ),
            ),
            fg_disconnect_cb=lambda: disconnect_fg(self),
            pick_seq_json_cb=lambda: pick_seq_json(self),
        )

        ttk.Label(self.run_tab, text="Run controls", font=("", 11, "bold")).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(self.run_tab, text="Settings are configured in Setup.").pack(anchor=tk.W, pady=(0, 6))

        run_nb = ttk.Notebook(self.run_tab)
        run_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.seq_tab = ttk.Frame(run_nb, padding=10)
        self.sweep_tab = ttk.Frame(run_nb, padding=10)
        self.camera_tab = ttk.Frame(run_nb, padding=10)
        run_nb.add(self.sweep_tab, text="Sweep")
        run_nb.add(self.seq_tab, text="Sequence")
        run_nb.add(self.camera_tab, text="Camera")

        diag_nb = ttk.Notebook(self.diag_tab)
        diag_nb.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.diag_info_tab = ttk.Frame(diag_nb, padding=10)
        self.manual_tab = ttk.Frame(diag_nb, padding=10)
        diag_nb.add(self.diag_info_tab, text="Diagnostics")
        diag_nb.add(self.manual_tab, text="Manual")

        ttk.Label(self.diag_info_tab, text="Diagnostics & logs", font=("", 11, "bold")).pack(
            anchor=tk.W, pady=(0, 6)
        )

        self._build_sequence_tab()
        self._build_manual_tab()
        self._build_sweep_tab()
        self._build_camera_tab()
        self._build_diagnostics_tab()

        # 右: ログパネル
        from .gui_support.log_panel import build_log_panel
        log_panel_frame = ttk.Frame(main_frame)
        log_panel_frame.pack(side=tk.RIGHT, fill=tk.Y)
        # app._log_panel_parent を使ってbuild_log_panelに親を渡す
        self._log_panel_parent = log_panel_frame
        build_log_panel(self)

    def _build_camera_tab(self) -> None:
        build_camera_tab(self)

    def _build_sequence_tab(self) -> None:
        build_sequence_tab(
            self,
            bitstring_help=BITSTRING_HELP,
            default_seq_path=Path("src/shutter_camera_trigger/sequence_examples/minimal_sequence.json"),
            seq_bits=SEQUENCE_BITS,
            all_off=ALL_OFF,
            nm_397=NM_397,
            nm_397_sig=NM_397_SIG,
            nm_729=NM_729,
            nm_854=NM_854,
            ao_rate_hz=AO_RATE_HZ,
        )

    def _build_manual_tab(self) -> None:
        build_manual_tab(
            self,
            all_off_value=ALL_OFF,
            nm_397=NM_397,
            nm_397_sig=NM_397_SIG,
            nm_729=NM_729,
            nm_854=NM_854,
        )

    def _build_sweep_tab(self) -> None:
        build_sweep_tab(
            self,
            default_daq_device=DEFAULT_DAQ_DEVICE,
            ao_rate_hz=AO_RATE_HZ,
            nm_397=NM_397,
            camera_trigger=CAMERA_TRIGGER,
            roi_pulse_s=ROI_PULSE_S,
            roi_idle_s=ROI_IDLE_S,
            roi_max_attempt=ROI_MAX_ATTEMPT,
            roi_check_cb=lambda: roi_check(self, default_daq_device=DEFAULT_DAQ_DEVICE),
            threshold_check_cb=lambda: threshold_check(self),
            start_sweep_cb=lambda: start_sweep(
                self,
                default_daq_device=DEFAULT_DAQ_DEVICE,
                fg_amp_max_mvpp=FG_AMP_MAX_MVPP,
                default_fg_amp_vpp=DEFAULT_FG_AMP_VPP,
            ),
            stop_sweep_cb=lambda: stop_sweep(self),
        )

    def _build_diagnostics_tab(self) -> None:
        build_diagnostics_tab(
            self,
            camera_check_cb=lambda: camera_check(
                self,
                default_daq_device=DEFAULT_DAQ_DEVICE,
                nm_397=NM_397,
                camera_trigger=CAMERA_TRIGGER,
                roi_pulse_s=ROI_PULSE_S,
                roi_idle_s=ROI_IDLE_S,
                ao_rate_hz=AO_RATE_HZ,
            ),
            camera_snap_cb=lambda: camera_snap(
                self,
                nm_397=NM_397,
                camera_trigger=CAMERA_TRIGGER,
                roi_pulse_s=ROI_PULSE_S,
                roi_idle_s=ROI_IDLE_S,
                ao_rate_hz=AO_RATE_HZ,
            ),
        )

def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()


