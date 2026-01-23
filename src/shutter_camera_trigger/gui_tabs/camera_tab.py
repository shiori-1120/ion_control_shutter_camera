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

from ..gui_support.image_utils import robust_gray_limits
from ..gui_support.diagnostics import set_last_error
from ..gui_support.camera_worker_manager import build_cam_cfg, ensure_camera_worker
from ..gui_support.camera_capture import acquire_frame_with_ttl
from ..gui_support.validators import parse_exposure_s_safe
from ..hardware import DaqClientDevice, DaqSequenceCommand
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


def plot_camera_frame(
    app: Any,
    frame: Any,
    *,
    title: str | None = None,
    roi: list[int] | None = None,
    bg_roi: list[int] | None = None,
) -> None:
    if app._cam_fig is None or app._cam_canvas is None:
        return
    ax = app._cam_ax
    if ax is None:
        return
    ax.clear()
    vmin, vmax = robust_gray_limits(frame)
    ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    try:
        from matplotlib.patches import Rectangle

        def _draw_box(box: list[int] | None, color: str) -> None:
            if not box or len(box) != 4:
                return
            xw, yw, xs, ys = [int(v) for v in box]
            ax.add_patch(
                Rectangle(
                    (xs, ys),
                    xw,
                    yw,
                    fill=False,
                    edgecolor=color,
                    linewidth=1.2,
                )
            )

        _draw_box(roi, "tab:orange")
        _draw_box(bg_roi, "tab:green")
    except Exception:
        pass
    try:
        app._cam_fig.tight_layout()
    except Exception:
        pass
    app._cam_canvas.draw()


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
    """カメラ接続確認だけを行うシンプルなチェック"""

    def _worker() -> None:
        import traceback
        ui_msg = ""
        try:
            cam_cfg = build_cam_cfg(app)
            trig_cfg = cam_cfg.get("trigger") or {}
            _require_external_trigger(trig_cfg)
            ready_timeout_s = 30.0
            prime_cb = None
            if cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg):
                if not app._daq.connected:
                    raise RuntimeError("DAQ not connected (external trigger)")
                ready_timeout_s = 180.0
                prime_cb = _build_prime_cb(
                    app,
                    nm_397=nm_397,
                    camera_trigger=camera_trigger,
                    roi_pulse_s=roi_pulse_s,
                    roi_idle_s=roi_idle_s,
                    ao_rate_hz=ao_rate_hz,
                )
            ensure_camera_worker(app, cam_cfg=cam_cfg, ready_timeout_s=ready_timeout_s, prime_cb=prime_cb)
            app._cam_status.set("Camera: ready")
            ui_msg = "カメラ接続OK"
        except Exception as e:
            ui_msg = f"カメラ接続エラー: {e}\n{traceback.format_exc(limit=2)}"
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
            cam_cfg = build_cam_cfg(app)
            exposure_s = float(cam_cfg.get("exposure_s") or parse_exposure_s_safe(app))
            frame_timeout_s = float(cam_cfg.get("frame_timeout_s") or max(1.0, exposure_s * 4.0 + 0.5))
            trig_cfg = cam_cfg.get("trigger") or {}
            _require_external_trigger(trig_cfg)
            ready_timeout_s = 30.0
            prime_cb = None
            if cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg):
                if not app._daq.connected:
                    raise RuntimeError("DAQ not connected (external trigger)")
                ready_timeout_s = 180.0
                prime_cb = _build_prime_cb(
                    app,
                    nm_397=nm_397,
                    camera_trigger=camera_trigger,
                    roi_pulse_s=roi_pulse_s,
                    roi_idle_s=roi_idle_s,
                    ao_rate_hz=ao_rate_hz,
                )
            _, cmd_q, _ = ensure_camera_worker(app, cam_cfg=cam_cfg, ready_timeout_s=ready_timeout_s, prime_cb=prime_cb)
            resp_q = getattr(app, "_cam_worker_resp_q", None)
            if resp_q is None:
                raise RuntimeError("Camera worker response queue missing")
            need_ttl = cam_cfg.get("mode") == "real" and _is_external_trigger(trig_cfg)
            daq = None
            pulse_width = max(float(roi_pulse_s), 0.01)
            if need_ttl:
                daq = DaqClientDevice(app._daq)
                daq.open(str(getattr(app, "_daq_device", "") or ""))
            max_attempt = 5

            def _send_get_frame(timeout_s: float, prefer_sample: str | None) -> None:
                cmd_q.put({"cmd": "get_frame", "timeout_s": float(timeout_s), "tag": f"snap-{_send_get_frame.attempt}"})

            _send_get_frame.attempt = -1  # type: ignore[attr-defined]

            def _run_ttl() -> None:
                if need_ttl and daq is not None:
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

            def _wait_resp(timeout_s: float, _label: str) -> dict:
                return resp_q.get(timeout=float(timeout_s))

            def _send_get_frame_with_count(timeout_s: float, prefer_sample: str | None) -> None:
                _send_get_frame.attempt += 1  # type: ignore[attr-defined]
                _send_get_frame(timeout_s, prefer_sample)

            resp_pack = acquire_frame_with_ttl(
                send_get_frame=_send_get_frame_with_count,
                run_ttl=_run_ttl,
                wait_resp=_wait_resp,
                max_attempt=max_attempt,
                frame_timeout_s=frame_timeout_s,
                resp_timeout_s=frame_timeout_s + 2.0,
                prefer_sample_path=None,
                sleep_s=0.05,
                log_cb=None,
            )
            if not resp_pack.get("ok"):
                raise RuntimeError(f"frame not ready after {max_attempt} TTLs: {resp_pack.get('error')}")
            resp = resp_pack.get("resp") or {}
            arr = np.asarray(resp.get("frame"))
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = _resolve_output_root(app) / "camera_snap" / ts
            out_dir.mkdir(parents=True, exist_ok=True)
            npy_path = out_dir / "snap.npy"
            np.save(npy_path, arr)
            app._cam_img = arr
            ui_msg = f"画像を{npy_path}に保存しました。 shape={arr.shape}"
            app.after(
                0,
                lambda: (
                    app._cam_status.set(f"Snap: OK shape={arr.shape}"),
                    plot_camera_frame(
                        app,
                        arr,
                        title=f"Snap {ts}",
                        roi=resp.get("roi"),
                        bg_roi=resp.get("bg_roi"),
                    ),
                ),
            )
        except Exception as e:
            ui_msg = f"カメラスナップエラー: {e}\n{traceback.format_exc(limit=2)}"
            app._cam_status.set(f"Snap: error {e}")
            set_last_error(app, label="Camera", message=str(e))
        app.after(0, lambda: messagebox.showinfo("Camera", ui_msg))

    threading.Thread(target=_worker, daemon=True).start()
