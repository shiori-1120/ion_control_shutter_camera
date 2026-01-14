from __future__ import annotations

from datetime import datetime
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Any, Callable
import queue
import threading
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import numpy as np

from ..gui_support.image_utils import robust_gray_limits
from ..gui_support.diagnostics import set_last_error
from ..gui_support.validators import (
    apply_subarray_to_cam_cfg,
    parse_camera_trigger_cfg,
    parse_exposure_s_safe,
)
from ..gui_support.worker_cleanup import cleanup_stale_workers, write_last_worker_pids
from ..gui_support.worker_messages import format_worker_failure
from ..hardware import CameraWorkerDevice, DaqClientDevice, DaqQueueDevice, DaqSequenceCommand
from ..config.device_registry import resolve_output_root
from ..sweep.session_config import write_manifest_json
from ..workers.camera_worker_process import start_camera_worker_process, stop_worker_process
from ..workers.daq_worker_process import start_daq_worker_process


def _resolve_output_root(app: Any) -> Path:
    try:
        root = getattr(app, "output_root", None)
        if root:
            return Path(root)
    except Exception:
        pass
    return resolve_output_root()


def build_camera_tab(app: Any) -> None:
    if app.camera_tab is None:
        return

    top = ttk.Frame(app.camera_tab)
    top.pack(fill=tk.X, pady=(0, 8))

    ttk.Label(top, text="Use Diagnostics > Diagnostics tools to run camera actions.").pack(side=tk.LEFT, padx=4)
    ttk.Label(top, textvariable=app._cam_status).pack(side=tk.LEFT, padx=12)

    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        app._cam_fig = Figure(figsize=(7.5, 4.6), dpi=100)
        app._cam_ax = app._cam_fig.add_subplot(111)
        app._cam_canvas = FigureCanvasTkAgg(app._cam_fig, master=app.camera_tab)
        app._cam_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    except Exception:
        app._cam_fig = None
        app._cam_ax = None
        app._cam_canvas = None
        ttk.Label(app.camera_tab, text="matplotlib not available; camera plot disabled").pack()


def get_camera_instance(app):
    """app._qCMOSにカメラインスタンスを保持し、なければ生成・openする。"""
    from src.camera.lib.ControlDevice import Control_qCMOScamera
    if getattr(app, '_qCMOS', None) is None:
        cam = Control_qCMOScamera(verbose=True)
        cam.OpenCamera_GetHandle()
        app._qCMOS = cam
    return app._qCMOS


def release_camera_instance(app):
    """アプリ終了時にカメラを解放する。"""
    cam = getattr(app, '_qCMOS', None)
    if cam is not None:
        try:
            cam.StopCapture()
        except Exception:
            pass
        try:
            cam.ReleaseBuf()
        except Exception:
            pass
        try:
            import time as _time
            _time.sleep(0.1)
            cam.CloseUninitCamera()
        except Exception:
            pass
        app._qCMOS = None


def camera_check(
    app: Any,
    *,
    default_daq_device: str,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    ao_rate_hz: float,
) -> None:
    """カメラ接続確認だけを行うシンプルなチェック"""
    def _worker() -> None:
        import traceback
        ui_msg = ""
        try:
            cam = get_camera_instance(app)
            ui_msg = "カメラ接続OK"
        except Exception as e:
            ui_msg = f"カメラ接続エラー: {e}\n{traceback.format_exc(limit=2)}"
        finally:
            try:
                release_camera_instance(app)
            except Exception:
                pass
        app.after(0, lambda: messagebox.showinfo("Camera", ui_msg))
        set_last_error(app, label="Camera", message=ui_msg)

    threading.Thread(target=_worker, daemon=True).start()


def camera_snap(
    app: Any,
    *,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    ao_rate_hz: float,
) -> None:
    """カメラスナップ（画像取得のみ）を行うシンプルな処理"""

    def _worker() -> None:
        import numpy as np
        import traceback
        from pathlib import Path
        from datetime import datetime
        import time as _time


        try:
            exposure_s = float(parse_exposure_s_safe(app))
            cam = get_camera_instance(app)
            cam.SetParameters(exposure_s)
            cam.StartCapture()

            from ..hardware import DaqClientDevice
            daq = DaqClientDevice(app._daq)
            pulse_width = max(roi_pulse_s, 0.01)  # 10ms以上
            max_attempt = 5
            ok = False
            err = None
            for attempt in range(max_attempt):
                daq.set_do(camera_trigger)
                _time.sleep(pulse_width)
                daq.set_do(0)
                ok, err = cam.wait_for_frame_ready(0.5)
                if ok:
                    break
                _time.sleep(0.05)  # 50ms待機
            if not ok:
                raise RuntimeError(f"frame not ready after {max_attempt} TTLs: {err}")
            _, frame = cam.GetLastFrame()
            arr = np.asarray(frame)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = _resolve_output_root(app) / "camera_snap" / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / "snap.npy"
            np.save(npy_path, arr)
            app._cam_img = arr
            app._cam_status.set(f"Snap: OK shape={arr.shape}")
            app._cam_canvas.draw()
            ui_msg = f"画像を{npy_path}に保存しました。 shape={arr.shape}"
        except Exception as e:
            ui_msg = f"カメラスナップエラー: {e}\n{traceback.format_exc(limit=2)}"
            app._cam_status.set(f"Snap: error {e}")
            set_last_error(app, label="Camera", message=str(e))
        finally:
            try:
                release_camera_instance(app)
            except Exception:
                pass
        app.after(0, lambda: messagebox.showinfo("Camera", ui_msg))

    threading.Thread(target=_worker, daemon=True).start()
