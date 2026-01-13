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
from ..gui_support.validators import (
    apply_subarray_to_cam_cfg,
    parse_camera_trigger_cfg,
    parse_exposure_s_safe,
)
from ..gui_support.worker_cleanup import cleanup_stale_workers, write_last_worker_pids
from ..gui_support.worker_messages import format_worker_failure
from ..workers.camera_worker_process import start_camera_worker_process, stop_worker_process
from ..workers.daq_worker_process import start_daq_worker_process


def build_camera_tab(app: Any, *, camera_snap_cb: Callable[[], None]) -> None:
    if app.camera_tab is None:
        return

    top = ttk.Frame(app.camera_tab)
    top.pack(fill=tk.X, pady=(0, 8))

    ttk.Button(top, text="Snap", command=camera_snap_cb).pack(side=tk.LEFT, padx=4)
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


def camera_snap(
    app: Any,
    *,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    ao_rate_hz: float,
) -> None:
    """Send TTL -> acquire one frame -> save .npy -> plot (no sweep)."""
    if app._cam_ax is None or app._cam_canvas is None:
        messagebox.showerror("Camera", "matplotlib is required for plotting")
        return

    if not app._daq.connected:
        messagebox.showerror("Camera", "DAQ is not connected. Please Connect first.")
        return

    mode = (app.camera_mode_top_var.get().strip() or "dry").lower()
    exposure_s = float(parse_exposure_s_safe(app))
    trig_cfg = parse_camera_trigger_cfg(app)
    trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"
    try:
        if getattr(app, "_logger", None):
            app._logger.info(
                "camera_check_start mode=%s exposure_s=%.4f trig_src=%s",
                mode,
                float(exposure_s),
                trig_src,
            )
    except Exception:
        pass
    log_ctx = getattr(app, "_log_ctx", None)
    daq_log_path = None
    run_id = None
    if log_ctx is not None:
        try:
            daq_log_path = str(log_ctx.log_dir / "daq_worker.log")
            run_id = str(log_ctx.run_id)
        except Exception:
            daq_log_path = None
            run_id = None
    dry_image_dir = app.dry_image_dir_var.get().strip()
    try:
        if getattr(app, "_logger", None):
            app._logger.info(
                "camera_snap_start mode=%s exposure_s=%.4f trig_src=%s dry_dir=%s",
                mode,
                float(exposure_s),
                trig_src,
                dry_image_dir,
            )
    except Exception:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data/output/camera_snap") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg: dict[str, Any] = {
        "mode": mode,
        "exposure_s": float(exposure_s),
        "frame_timeout_s": max(1.0, float(exposure_s) * 4.0 + 0.5),
        "bootstrap_n": 1,
        "trigger": dict(trig_cfg),
        "verbose": bool(app.camera_verbose_var.get()),
    }
    try:
        apply_subarray_to_cam_cfg(app, cfg)
    except Exception as e:
        messagebox.showerror("Subarray", str(e))
        return
    if dry_image_dir:
        cfg["dry_image_dir"] = dry_image_dir
    log_ctx = getattr(app, "_log_ctx", None)
    try:
        if log_ctx is not None:
            cfg["log_path"] = str(log_ctx.log_dir / "camera_worker.log")
            cfg["run_id"] = str(log_ctx.run_id)
        else:
            cfg["log_path"] = str(out_dir / "camera_worker.log")
    except Exception:
        pass

    pulse_seq = [(nm_397, roi_idle_s), (nm_397 | camera_trigger, roi_pulse_s), (nm_397, roi_idle_s)]

    def _worker() -> None:
        cam_p, cam_cmd_q, cam_resp_q = start_camera_worker_process(cfg=cfg)

        try:
            app.after(0, lambda: app._cam_status.set("Snap: starting camera..."))

            cam_ready: dict[str, Any] | None = None
            if mode == "real" and trig_src in ("EXTERNAL", "EXT", "2", ""):
                deadline = time.time() + 15.0
                while time.time() < deadline:
                    try:
                        cam_ready = cam_resp_q.get_nowait()
                        break
                    except Exception:
                        pass
                    try:
                        app._daq.request(
                            {
                                "cmd": "run_sequence_once",
                                "do_sequence": pulse_seq,
                                "insert_index": -1,
                                "ao_width_ms": 0.0,
                                "ao_rate_hz": float(ao_rate_hz),
                                "ao_v_high": 5.0,
                                "ao_v_low": 0.0,
                            },
                            timeout=3.0,
                        )
                    except Exception:
                        time.sleep(0.05)
                    time.sleep(0.01)

            if cam_ready is None:
                try:
                    cam_ready = cam_resp_q.get(timeout=15)
                except Exception as e:
                    raise RuntimeError(f"Camera ready timeout: {e}")

            if not cam_ready.get("ok"):
                raise RuntimeError(f"Camera ready failed: {cam_ready}")

            app.after(0, lambda: app._cam_status.set("Snap: ready, pulsing..."))
            app._daq.request(
                {
                    "cmd": "run_sequence_once",
                    "do_sequence": pulse_seq,
                    "insert_index": -1,
                    "ao_width_ms": 0.0,
                    "ao_rate_hz": float(ao_rate_hz),
                    "ao_v_high": 5.0,
                    "ao_v_low": 0.0,
                },
                timeout=5.0,
            )

            cam_cmd_q.put({"cmd": "get_frame", "timeout_s": max(2.0, float(exposure_s) * 5.0)})
            cam_resp = cam_resp_q.get(timeout=20)
            if not cam_resp.get("ok"):
                raise RuntimeError(f"Camera frame failed: {cam_resp}")

            frame = np.asarray(cam_resp.get("frame"))
            npy_path = out_dir / "snap.npy"
            np.save(npy_path, frame)

            roi = cam_resp.get("roi")
            if roi is None:
                try:
                    from src.camera.lib.analysis_profiles import generate_rois_from_image

                    rois = generate_rois_from_image(np.asarray(frame), plot=False)
                    if rois:
                        roi = list(rois[0])
                except Exception:
                    roi = None

            try:
                if isinstance(roi, (list, tuple)) and len(roi) == 4:
                    xw, yw, xs, ys = map(int, roi)
                else:
                    xw = yw = xs = ys = None
            except Exception:
                xw = yw = xs = ys = None

            vmin, vmax = robust_gray_limits(frame)
            app._cam_ax.clear()
            app._cam_ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            app._cam_ax.set_title(f"snap | {mode} | saved: {npy_path}")
            app._cam_ax.set_axis_off()

            if xw is not None:
                try:
                    from matplotlib.patches import Rectangle

                    rect = Rectangle((xs, ys), xw, yw, fill=False, edgecolor="tab:red", linewidth=2)
                    app._cam_ax.add_patch(rect)
                except Exception:
                    pass

            app._cam_fig.tight_layout()
            app._cam_canvas.draw()
            app._cam_status.set("Snap: done")
            try:
                if getattr(app, "_logger", None):
                    app._logger.info("camera_snap_done saved=%s", str(npy_path))
            except Exception:
                pass
        except Exception:
            app.after(0, lambda: app._cam_status.set("Snap: failed"))
            try:
                if getattr(app, "_logger", None):
                    app._logger.error("camera_snap_failed")
            except Exception:
                pass
        finally:
            try:
                cam_cmd_q.put({"cmd": "close"})
            except Exception:
                pass
            try:
                if cam_p is not None and cam_p.is_alive():
                    cam_p.join(timeout=2.0)
                    if cam_p.is_alive():
                        cam_p.terminate()
                        cam_p.join(timeout=1.0)
            except Exception:
                pass

    t = app._start_thread(_worker)
    if t is None:
        app._cam_status.set("Snap: starting...")


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
    """Spawn camera worker once to verify connectivity/dry samples."""
    cleanup_stale_workers(app.worker_pids_path)

    mode = app.camera_mode_top_var.get().strip() or "dry"
    dry_dir = app.dry_image_dir_var.get().strip()
    exposure_s = parse_exposure_s_safe(app)

    trig_cfg = parse_camera_trigger_cfg(app)
    trig_src = str(trig_cfg.get("source") or "EXTERNAL").strip().upper() or "EXTERNAL"

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
        "verbose": bool(app.camera_verbose_var.get()),
    }
    try:
        apply_subarray_to_cam_cfg(app, cfg)
    except Exception as e:
        messagebox.showerror("Subarray", str(e))
        return
    log_ctx = getattr(app, "_log_ctx", None)
    try:
        if log_ctx is not None:
            cfg["log_path"] = str(log_ctx.log_dir / "camera_worker.log")
            cfg["run_id"] = str(log_ctx.run_id)
        else:
            cfg["log_path"] = str(out_dir / "camera_worker.log")
    except Exception:
        pass
    if mode == "dry" and dry_dir:
        cfg["dry_image_dir"] = dry_dir

    def _worker() -> None:
        p, cmd_q, resp_q = start_camera_worker_process(cfg=cfg)

        prime_stop = threading.Event()
        prime_thread: threading.Thread | None = None

        tmp_daq_proc: Process | None = None
        tmp_daq_cmd_q: Queue | None = None
        tmp_daq_resp_q: Queue | None = None

        def _start_tmp_daq() -> tuple[Queue, Queue, Process]:
            device = app.device_var.get().strip() or default_daq_device
            daq_mode = app.device_mode_var.get().strip().lower() or "real"
            if daq_mode != "real":
                raise RuntimeError(
                    "Camera check in EXTERNAL trigger mode requires DAQ mode 'real' (to output TTL)."
                )

            proc, dq, rq = start_daq_worker_process(
                device=device,
                mode=daq_mode,
                log_path=daq_log_path,
                run_id=run_id,
            )
            return dq, rq, proc

        def _prime_loop_using_existing() -> None:
            roi_sequence = [
                (nm_397, roi_idle_s),
                (nm_397 | camera_trigger, roi_pulse_s),
                (nm_397, roi_idle_s),
            ]
            while not prime_stop.is_set():
                try:
                    app._daq.request(
                        {
                            "cmd": "run_sequence_once",
                            "do_sequence": roi_sequence,
                            "insert_index": -1,
                            "ao_width_ms": 0.0,
                            "ao_rate_hz": float(ao_rate_hz),
                            "ao_v_high": 5.0,
                            "ao_v_low": 0.0,
                        },
                        timeout=2.0,
                    )
                except Exception:
                    pass
                time.sleep(0.01)

        def _prime_loop_using_tmp(dq: Queue, rq: Queue) -> None:
            roi_sequence = [
                (nm_397, roi_idle_s),
                (nm_397 | camera_trigger, roi_pulse_s),
                (nm_397, roi_idle_s),
            ]
            while not prime_stop.is_set():
                try:
                    dq.put(
                        {
                            "cmd": "run_sequence_once",
                            "do_sequence": roi_sequence,
                            "insert_index": -1,
                            "ao_width_ms": 0.0,
                            "ao_rate_hz": float(ao_rate_hz),
                            "ao_v_high": 5.0,
                            "ao_v_low": 0.0,
                        }
                    )
                except Exception:
                    pass

                try:
                    rq.get(timeout=0.1)
                except queue.Empty:
                    pass
                except Exception:
                    pass
                time.sleep(0.01)

        want_prime = (mode == "real") and (trig_src in ("EXTERNAL", "EXT", "2", ""))
        if want_prime:
            try:
                if app._daq.connected:
                    prime_thread = threading.Thread(target=_prime_loop_using_existing, daemon=True)
                    prime_thread.start()
                else:
                    dq, rq, proc = _start_tmp_daq()
                    tmp_daq_cmd_q, tmp_daq_resp_q, tmp_daq_proc = dq, rq, proc
                    prime_thread = threading.Thread(
                        target=_prime_loop_using_tmp,
                        args=(tmp_daq_cmd_q, tmp_daq_resp_q),
                        daemon=True,
                    )
                    prime_thread.start()
            except Exception as e:
                def _ui_fail(msg=str(e)) -> None:
                    messagebox.showerror("Camera", f"Failed to start DAQ priming for EXTERNAL trigger.\n{msg}")

                app.after(0, _ui_fail)
                return

        try:
            write_last_worker_pids(
                app.worker_pids_path,
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
            ready = resp_q.get(
                timeout=max(15.0, float(cfg.get("bootstrap_n", 5)) * (float(exposure_s) + 0.05) + 5.0)
            )
            if ready.get("ok"):
                ok = True
                dry_samples = ready.get("dry_samples")
                extra = ""
                if dry_samples is not None:
                    extra = f" | dry samples: {dry_samples}"
                try:
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
                ui_msg = format_worker_failure(
                    ready,
                    label="Camera worker failed",
                    log_path=str(cfg.get("log_path") or "") or None,
                )
                ui_kind = "error"
        except Exception as e:
            ui_msg = format_worker_failure(
                e,
                label="Camera check failed",
                log_path=str(cfg.get("log_path") or "") or None,
            )
            ui_kind = "error"
        finally:
            prime_stop.set()
            stop_worker_process(proc=p, cmd_q=cmd_q)

            try:
                write_last_worker_pids(app.worker_pids_path, {})
            except Exception:
                pass

            stop_worker_process(proc=tmp_daq_proc, cmd_q=tmp_daq_cmd_q, join_timeout_s=2.0, terminate_timeout_s=1.0)

        def _ui() -> None:
            app._camera_connected = ok
            try:
                if (
                    frame_np is not None
                    and app._cam_ax is not None
                    and app._cam_canvas is not None
                    and app._cam_fig is not None
                ):
                    app._cam_ax.clear()
                    vmin, vmax = robust_gray_limits(frame_np)
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
