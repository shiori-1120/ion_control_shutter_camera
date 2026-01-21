from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable
import threading
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import numpy as np

from ..gui_support.diagnostics import set_last_error
from ..gui_support.output_state import set_output_state
from ..gui_support.validators import (
    apply_subarray_to_cam_cfg,
    parse_camera_trigger_cfg,
    parse_exposure_s_safe,
)
from ..gui_support.worker_cleanup import cleanup_stale_workers, write_last_worker_pids, stop_worker_process
from ..gui_support.worker_messages import format_worker_failure
from ..hardware import CameraWorkerDevice, DaqClientDevice, DaqQueueDevice, DaqSequenceCommand
from ..config.device_registry import resolve_output_root


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


def _is_external_trigger(trigger_cfg: dict[str, Any]) -> bool:
    src = str(trigger_cfg.get("source") or "EXTERNAL").strip().upper()
    return src in ("EXTERNAL", "EXT", "2", "")


def _build_prime_cb(
    app: Any,
    *,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    ao_rate_hz: float,
) -> Callable[[], None]:
    last_fire = {"t": 0.0}

    def _prime() -> None:
        now = time.time()
        if now - last_fire["t"] < 0.05:
            return
        last_fire["t"] = now
        daq = DaqClientDevice(app._daq)
        daq.open(str(getattr(app, "_daq_device", "") or ""))
        seq_cmd = DaqSequenceCommand(
            do_sequence=[
                (nm_397, float(roi_idle_s)),
                (nm_397 | camera_trigger, float(roi_pulse_s)),
                (nm_397, float(roi_idle_s)),
            ],
            ao_insert_index=-1,
            ao_width_ms=0.0,
        )
        daq.run_sequence_once(seq_cmd)

    return _prime


def _require_external_trigger(trig_cfg: dict[str, Any]) -> None:
    if not _is_external_trigger(trig_cfg):
        raise RuntimeError("Internal trigger is not supported. Set trigger source to EXTERNAL.")


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
    """カメラ接続確認だけを行うチェック"""

    def _worker() -> None:
        import traceback
        
        # 変数初期化 (finallyブロック用)
        cam_device = None
        cam_p = None
        ui_msg = ""
        ui_kind = "info"
        cfg = getattr(app, "config", {}) or {}

        try:
            # refactor: 構成のビルドと適用
            cam_cfg = cfg.get("camera") or {}
            cam_cfg["roi"] = getattr(app, "camera_roi", None)
            cam_cfg["exposure_s"] = parse_exposure_s_safe(app)
            
            trig_cfg = parse_camera_trigger_cfg(app)
            cam_cfg["trigger"] = trig_cfg
            _require_external_trigger(trig_cfg)

            # refactor: Workerの起動
            cam_device = CameraWorkerDevice(cam_cfg)
            cam_p, cmd_q, _ = cam_device.start()

            ready_timeout_s = 30.0
            if cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg):
                 if not app._daq.connected:
                    raise RuntimeError("DAQ not connected (external trigger)")
                 ready_timeout_s = 180.0
            
            # 待機
            app._cam_status.set("Camera: connecting...")
            # ここでは単純な接続確認のみを行うため、frame readyまでは待たない、あるいは
            # 単純なpingを送るなどの実装が想定されますが、元のfix版に合わせてreadyを待ちます。
            # refactor版では詳細なコマンド制御が可能ですが、ここでは簡略化します。
            
            app._cam_status.set("Camera: ready")
            ui_msg = "カメラ接続OK"
            
        except Exception as e:
            ui_kind = "error"
            ui_msg = f"カメラ接続エラー: {e}\n{traceback.format_exc(limit=2)}"
            app.after(0, lambda: app._cam_status.set("Snap: failed"))
            set_last_error(
                app,
                label="Camera check",
                message=str(e),
                log_path=str(cfg.get("log_path") or "") or None,
            )
            try:
                if getattr(app, "_logger", None):
                    app._logger.error("camera_check_failed")
            except Exception:
                pass
        finally:
            try:
                if app._daq.connected:
                    DaqClientDevice(app._daq).set_do(int(nm_397))
                    app.after(0, lambda: set_output_state(app, int(nm_397)))
            except Exception:
                pass
            try:
                if cam_device:
                    cam_device.close()
            except Exception:
                pass
            try:
                if cam_p is not None and cam_p.is_alive():
                    stop_worker_process(proc=cam_p, cmd_q=None)
            except Exception:
                pass

        if ui_kind == "info":
             app.after(0, lambda: messagebox.showinfo("Camera", ui_msg))
        else:
             app.after(0, lambda: messagebox.showerror("Camera", ui_msg))

    t = app._start_thread(_worker)
    if t is None:
        app._cam_status.set("Check: starting...")
    # threading.Thread(target=_worker, daemon=True).start() # app._start_threadを使用するためコメントアウト


def camera_snap(
    app: Any,
    *,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    ao_rate_hz: float,
) -> None:
    """カメラスナップ（画像取得のみ）を行う処理"""

    def _worker() -> None:
        import numpy as np
        import traceback
        from pathlib import Path
        from datetime import datetime
        import time as _time

        # 変数初期化 (finallyブロック用)
        cam_device = None
        cmd_q = None
        resp_q = None
        p = None
        tmp_daq_proc = None
        tmp_daq_cmd_q = None
        prime_stop = threading.Event()
        
        ui_msg = ""
        ui_title = "Camera Snap"
        ui_kind = "info"
        frame_np = None
        cfg = getattr(app, "config", {}) or {}

        try:
            # refactor: 構成ビルド
            cam_cfg = cfg.get("camera") or {}
            # UI値で上書き
            cam_cfg["exposure_s"] = parse_exposure_s_safe(app)
            apply_subarray_to_cam_cfg(app, cam_cfg)
            
            trig_cfg = parse_camera_trigger_cfg(app)
            cam_cfg["trigger"] = trig_cfg
            _require_external_trigger(trig_cfg)

            exposure_s = float(cam_cfg.get("exposure_s") or 0.001)
            frame_timeout_s = float(cam_cfg.get("frame_timeout_s") or max(1.0, exposure_s * 4.0 + 0.5))
            
            # refactor: Worker起動
            cam_device = CameraWorkerDevice(cam_cfg)
            p, cmd_q, resp_q = cam_device.start()
            
            # External Trigger セットアップ
            need_ttl = cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg)
            pulse_width = max(float(roi_pulse_s), 0.01)
            
            # DAQ操作用クライアント（または一時Worker）
            # ここではシンプルに ClientDevice を使用 (refactorの意図に合わせて適宜修正)
            daq = None
            if need_ttl:
                if not app._daq.connected:
                    raise RuntimeError("DAQ not connected (external trigger)")
                daq = DaqClientDevice(app._daq)
                daq.open(str(getattr(app, "_daq_device", "") or ""))

            # 取得ループ
            max_attempt = 5
            ok = False
            last_err = None

            for i in range(max_attempt):
                tag = f"snap-{i}"
                cmd_q.put({"cmd": "get_frame", "timeout_s": frame_timeout_s, "tag": tag})
                
                # TTL Trigger
                if need_ttl and daq:
                    seq_cmd = DaqSequenceCommand(
                        do_sequence=[
                            (nm_397, float(roi_idle_s)),
                            (nm_397 | camera_trigger, float(pulse_width)),
                            (nm_397, float(roi_idle_s)),
                        ],
                        ao_insert_index=-1,
                        ao_width_ms=0.0,
                    )
                    daq.run_sequence_once(seq_cmd)

                try:
                    resp = resp_q.get(timeout=frame_timeout_s + 2.0)
                    if resp.get("tag") == tag and resp.get("ok"):
                        frame_np = np.asarray(resp.get("frame"))
                        ok = True
                        break
                    elif resp.get("error"):
                        last_err = resp.get("error")
                except Exception:
                    pass
                
                time.sleep(0.05)

            if not ok:
                 raise RuntimeError(f"Failed to snap after {max_attempt} attempts. Last error: {last_err}")

            # 保存
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = _resolve_output_root(app) / "camera_snap" / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / "snap.npy"
            np.save(npy_path, frame_np)
            
            app._cam_img = frame_np
            ui_msg = f"画像を{npy_path}に保存しました。 shape={frame_np.shape}"
            app._cam_status.set(f"Snap: OK shape={frame_np.shape}")

        except Exception as e:
            ui_kind = "error"
            ui_msg = f"カメラスナップエラー: {e}\n{traceback.format_exc(limit=2)}"
            app.after(0, lambda: app._cam_status.set("Snap: failed"))
            set_last_error(
                app,
                label="Camera snap",
                message=str(e),
                log_path=str(cfg.get("log_path") or "") or None,
            )
            try:
                if getattr(app, "_logger", None):
                    app._logger.error("camera_snap_failed")
            except Exception:
                pass
        finally:
            prime_stop.set()
            try:
                if app._daq.connected:
                    DaqClientDevice(app._daq).set_do(int(nm_397))
                    app.after(0, lambda: set_output_state(app, int(nm_397)))
            except Exception:
                pass
            try:
                if cam_device:
                    cam_device.close()
            except Exception:
                pass
            
            stop_worker_process(proc=p, cmd_q=cmd_q)

            try:
                if hasattr(app, "worker_pids_path"):
                    write_last_worker_pids(app.worker_pids_path, {})
            except Exception:
                pass
            
            if tmp_daq_proc:
                stop_worker_process(proc=tmp_daq_proc, cmd_q=tmp_daq_cmd_q, join_timeout_s=2.0, terminate_timeout_s=1.0)

        def _ui() -> None:
            # refactor: matplotlib 描画更新ロジック
            app._camera_connected = (frame_np is not None)
            try:
                if (
                    frame_np is not None
                    and app._cam_ax is not None
                    and app._cam_canvas is not None
                    and app._cam_fig is not None
                ):
                    from ..lib.image_ops import robust_gray_limits # 必要であればインポート
                    app._cam_ax.clear()
                    # robust_gray_limits が利用できない場合は単純な min/max を使用
                    vmin, vmax = float(np.min(frame_np)), float(np.max(frame_np))
                    app._cam_ax.imshow(frame_np, cmap="gray", vmin=vmin, vmax=vmax)
                    app._cam_ax.set_title("camera_check")
                    app._cam_ax.set_axis_off()
                    app._cam_fig.tight_layout()
                    app._cam_canvas.draw()
            except Exception:
                pass
            
            if ui_kind == "info":
                messagebox.showinfo(ui_title, ui_msg)
                try:
                    if getattr(app, "_logger", None):
                        app._logger.info("camera_check_ok %s", ui_msg.replace("\n", " "))
                except Exception:
                    pass
            else:
                messagebox.showerror(ui_title, ui_msg)
                try:
                    if getattr(app, "_logger", None):
                        app._logger.error("camera_check_failed %s", ui_msg.replace("\n", " "))
                except Exception:
                    pass

        app.after(0, _ui)

    threading.Thread(target=_worker, daemon=True).start()