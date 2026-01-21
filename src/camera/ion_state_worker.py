"""Camera + ion state classification worker.

Design goals (for this project):
- separate process from DAQ to reduce timing jitter impact
- classify only bright/dark (frame content itself is not important)
- allow dropping frames (we always take the latest frame)

The worker can run in two modes:
- dry-run: generate synthetic bright/dark with noise (for software bring-up)
- real: use Control_qCMOScamera (DCAM) and wait for FRAMEREADY

Communication:
- cmd_q receives dict commands
- resp_q returns dict results

Important:
- On Windows, this module is meant to be used via multiprocessing spawn.
"""

from __future__ import annotations

import os
import queue
import time
import traceback
import threading
from pathlib import Path
from multiprocessing.queues import Queue
from typing import Any

from .worker_commands import handle_roi_threshold_cmd
from .worker_dry import handle_dry_command, load_dry_samples
from .worker_logging import setup_worker_logging
from .worker_real import init_real_camera
from .worker_utils import as_roi_tuple, limit_blas_threads


def ion_state_worker_main(cmd_q: Queue, resp_q: Queue, cfg: dict[str, Any]) -> None:
    limit_blas_threads()

    log, log_debug, log_worker_env, close_log, cam_verbose = setup_worker_logging(cfg)

    # The legacy camera stack expects `import lib.*` to resolve to src/camera/lib.
    # When running as a module from repo root, we need to prepend src/camera to sys.path.
    from pathlib import Path
    import sys

    camera_dir = Path(__file__).resolve().parent
    if str(camera_dir) not in sys.path:
        sys.path.insert(0, str(camera_dir))

    log(
        f"worker start | pid={os.getpid()} | mode={cfg.get('mode')} | "
        f"exposure_s={cfg.get('exposure_s')} | frame_timeout_s={cfg.get('frame_timeout_s')} | "
        f"bootstrap_n={cfg.get('bootstrap_n')}"
    )
    log_worker_env()

    mode = str(cfg.get("mode") or "dry")  # 'dry' | 'real'

    roi = cfg.get("roi")
    bg_roi = cfg.get("bg_roi")
    exposure_s = float(cfg.get("exposure_s") or 0.001)
    frame_timeout_s = float(cfg.get("frame_timeout_s") or 1.0)
    bootstrap_n = int(cfg.get("bootstrap_n") or 10)

    tau_on = cfg.get("tau_on")
    tau_off = cfg.get("tau_off")

    trigger_cfg = cfg.get("trigger")
    log(f"trigger_cfg={trigger_cfg} | cam_verbose={cam_verbose}")

    prev_state: bool | None = None
    cam: Any | None = None
    acq_stop = threading.Event()
    acq_thread: threading.Thread | None = None
    frame_cv = threading.Condition()
    latest_frame: Any | None = None
    latest_frame_ts: float | None = None
    latest_frame_seq = -1
    last_frame_error: str | None = None

    # Imported lazily (real mode only)
    np = None  # type: ignore
    normalize_count = None  # type: ignore
    classify_hysteresis = None  # type: ignore

    def send(msg: dict[str, Any]) -> None:
        try:
            resp_q.put(msg)
        except Exception:
            pass

    roi_t = as_roi_tuple(roi)
    bg_roi_t = as_roi_tuple(bg_roi)
    subarray_t = as_roi_tuple(cfg.get("subarray"))

    dry_samples = load_dry_samples(cfg.get("dry_image_dir"))

    try:
        if mode == "dry":
            log("mode=dry -> sending ready")
            send({"ok": True, "event": "ready", "mode": "dry", "dry_samples": len(dry_samples)})

        elif mode == "real":
            log("mode=real -> importing camera stack")
            # Import the real camera/analysis stack only in real mode.
            import numpy as np

            from .lib.analysis_profiles import generate_rois_from_image
            from .lib.ControlDevice import Control_qCMOScamera
            from .lib.image_ops import crop_roi
            from .lib.thresholding import bootstrap_threshold_from_stream, classify_hysteresis, normalize_count

            # store for command loop
            locals_np = np
            locals_norm = normalize_count
            locals_cls = classify_hysteresis

            log("creating Control_qCMOScamera")
            cam = Control_qCMOScamera(trigger_cfg=trigger_cfg, verbose=cam_verbose)
            log("OpenCamera_GetHandle")
            cam.OpenCamera_GetHandle()
            # Full frame by default. If subarray is configured, apply it at camera level.
            log("SetParameters")
            if subarray_t is not None:
                xw, yw, xs, ys = map(int, subarray_t)
                cam.SetParameters(exposure_s, xw, yw, xs, ys)
                log(f"subarray applied: x={xs} y={ys} w={xw} h={yw}")
            else:
                cam.SetParameters(exposure_s)
            log("StartCapture")
            cam.StartCapture()

            def _store_frame(frame_any: Any) -> None:
                nonlocal latest_frame, latest_frame_ts, last_frame_error, latest_frame_seq
                with frame_cv:
                    latest_frame = frame_any
                    latest_frame_ts = time.time()
                    latest_frame_seq += 1
                    last_frame_error = None
                    frame_cv.notify_all()

            def _store_error(err_msg: str) -> None:
                nonlocal last_frame_error
                with frame_cv:
                    last_frame_error = err_msg
                    frame_cv.notify_all()

            def _wait_latest(timeout_s: float, *, min_seq: int) -> tuple[Any | None, str | None]:
                deadline = time.time() + float(timeout_s)
                with frame_cv:
                    while latest_frame is None or latest_frame_seq < int(min_seq):
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            return None, last_frame_error
                        frame_cv.wait(timeout=remaining)
                    return latest_frame, last_frame_error

            # Bootstrap ROI + thresholds if missing
            frames: list[np.ndarray] = []
            for i in range(max(1, bootstrap_n)):
                log_debug(f"bootstrap wait_for_frame_ready {i+1}/{max(1, bootstrap_n)}")
                ok, err = cam.wait_for_frame_ready(frame_timeout_s)
                if not ok:
                    raise RuntimeError(f"Camera timeout during bootstrap: {err}")
                _, frame = cam.GetLastFrame()
                frames.append(np.asarray(frame))

            if roi_t is None:
                rois = generate_rois_from_image(np.asarray(frames[-1]), plot=False)
                if not rois:
                    raise RuntimeError("Failed to auto-detect ROI from image")
                # single-ion: pick ROI with max sum
                best = None
                best_sum = None
                for r in rois:
                    r_t = as_roi_tuple(r)
                    if r_t is None:
                        continue
                    xw, yw, xs, ys = r_t
                    cropped = crop_roi(frames[-1], (xw, yw, xs, ys))
                    s = float(np.sum(cropped))
                    if best_sum is None or s > best_sum:
                        best_sum = s
                        best = r_t
                if best is None:
                    raise RuntimeError("Failed to select a ROI")
                roi_t = best

            if (tau_on is None) or (tau_off is None):
                th = bootstrap_threshold_from_stream(
                    frames,
                    roi_t,
                    bg_roi=bg_roi_t,
                    exposure_s_list=[exposure_s] * len(frames),
                    sample_n=min(bootstrap_n, len(frames)),
                )
                tau_on = float(th["tau_on"])
                tau_off = float(th["tau_off"])

            send(
                {
                    "ok": True,
                    "event": "ready",
                    "mode": "real",
                    "roi": list(roi_t),
                    "bg_roi": (list(bg_roi_t) if bg_roi_t else None),
                    "tau_on": float(tau_on),
                    "tau_off": float(tau_off),
                    "exposure_s": float(exposure_s),
                }
            )
            log("sent ready")

            # expose to command loop
            np = locals_np
            normalize_count = locals_norm
            classify_hysteresis = locals_cls

            def _acq_loop() -> None:
                log("acq_thread start")
                while not acq_stop.is_set():
                    try:
                        ok, err = cam.wait_for_frame_ready(frame_timeout_s)
                        if not ok:
                            _store_error(str(err))
                            continue
                        _, frame = cam.GetLastFrame()
                        frame_np = np.asarray(frame) if np is not None else frame
                        _store_frame(frame_np)
                    except Exception as e:
                        _store_error(str(e))
                        time.sleep(0.001)
                log("acq_thread stop")

            acq_thread = threading.Thread(target=_acq_loop, daemon=True)
            acq_thread.start()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        while True:
            try:
                cmd = cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if not isinstance(cmd, dict):
                continue

            name = cmd.get("cmd")
            cmd_tag = cmd.get("tag")
            log_debug(f"cmd={name}")
            if name in ("quit", "close"):
                log("closing")
                send({"ok": True, "event": "closing"})
                if cam is not None:
                    log("cleanup (on close cmd): StopCapture/ReleaseBuf/CloseUninitCamera")
                    try:
                        cam.StopCapture()
                    except Exception:
                        pass
                    try:
                        cam.ReleaseBuf()
                    except Exception:
                        pass
                    try:
                        cam.CloseUninitCamera()
                    except Exception:
                        pass
                    cam = None
                break

            handled, resp, state = handle_roi_threshold_cmd(
                name,
                cmd,
                roi_t=roi_t,
                bg_roi_t=bg_roi_t,
                tau_on=tau_on,
                tau_off=tau_off,
                prev_state=prev_state,
                log=log,
            )
            if handled:
                roi_t, bg_roi_t, tau_on, tau_off, prev_state = state
                if resp is not None:
                    send(resp)
                continue

            if name == "set_subarray":
                try:
                    sub_new = as_roi_tuple(cmd.get("subarray"))
                    if mode == "dry":
                        subarray_t = sub_new
                        send({"ok": True, "event": "subarray", "subarray": (list(subarray_t) if subarray_t else None)})
                        log(f"set_subarray (dry) {subarray_t}")
                        continue
                    if cam is None:
                        raise RuntimeError("Camera worker is not configured")
                    log(f"set_subarray {sub_new}")
                    try:
                        cam.StopCapture()
                    except Exception:
                        pass
                    try:
                        cam.ReleaseBuf()
                    except Exception:
                        pass
                    if sub_new is not None:
                        xw, yw, xs, ys = map(int, sub_new)
                        cam.SetParameters(exposure_s, xw, yw, xs, ys)
                        subarray_t = (xw, yw, xs, ys)
                    else:
                        cam.SetParameters(exposure_s)
                        subarray_t = None
                    cam.StartCapture()
                    send({"ok": True, "event": "subarray", "subarray": (list(subarray_t) if subarray_t else None)})
                except Exception as e:
                    log(f"set_subarray error {type(e).__name__}: {e}")
                    send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})
                continue

            if name not in ("get_state", "get_frame"):
                send({"ok": False, "event": "error", "error": f"unknown cmd: {name}"})
                continue

            timeout_s = float(cmd.get("timeout_s") or frame_timeout_s)
            log_debug(f"{name} timeout_s={timeout_s}")

            try:
                if mode == "dry":
                    resp, prev_state = handle_dry_command(
                        name=name,
                        cmd=cmd,
                        dry_samples=dry_samples,
                        subarray_t=subarray_t,
                        roi_t=roi_t,
                        bg_roi_t=bg_roi_t,
                        tau_on=tau_on,
                        tau_off=tau_off,
                        prev_state=prev_state,
                    )
                    send(resp)
                    continue

                if cam is None:
                    raise RuntimeError("Camera worker is not configured")

                with frame_cv:
                    prev_seq = int(latest_frame_seq)
                frame_np, last_err = _wait_latest(timeout_s, min_seq=prev_seq + 1)
                if frame_np is None:
                    log(f"frame_timeout err={last_err}")
                    resp = {"ok": False, "event": "timeout", "error": str(last_err or "timeout")}
                    if cmd_tag is not None:
                        resp["tag"] = cmd_tag
                    send(resp)
                    continue


                _, frame = cam.GetLastFrame()
                frame_np = np.asarray(frame) if np is not None else frame
                S_norm: float | None = None
                bright: bool | None = None
                if (normalize_count is not None) and (roi_t is not None):
                    norm = normalize_count(frame_np, roi_t, bg_roi=bg_roi_t, exposure_s=exposure_s)
                    S_norm = float(norm["S_norm"])
                    if (tau_on is not None) and (tau_off is not None):
                        if abs(float(tau_on) - float(tau_off)) < 1e-12:
                            bright = bool(S_norm > float(tau_on))
                            prev_state = bool(bright)
                        elif classify_hysteresis is not None:
                            bright = bool(
                                classify_hysteresis(
                                    S_norm,
                                    prev_state_bright=prev_state,
                                    tau_on=float(tau_on),
                                    tau_off=float(tau_off),
                                )
                            )
                            prev_state = bool(bright)

                if name == "get_frame":
                    resp = {
                        "ok": True,
                        "event": "frame",
                        "frame": frame_np,
                        "roi": list(roi_t) if roi_t else None,
                        "bg_roi": list(bg_roi_t) if bg_roi_t else None,
                        "bright": bright,
                        "S_norm": S_norm,
                        "tau_on": float(tau_on) if tau_on is not None else None,
                        "tau_off": float(tau_off) if tau_off is not None else None,
                        "exposure_s": float(exposure_s),
                    }
                    if cmd_tag is not None:
                        resp["tag"] = cmd_tag
                    send(resp)
                else:
                    resp = {
                        "ok": True,
                        "event": "state",
                        "bright": bool(bright) if bright is not None else False,
                        "S_norm": S_norm,
                        "tau_on": float(tau_on) if tau_on is not None else None,
                        "tau_off": float(tau_off) if tau_off is not None else None,
                    }
                    if cmd_tag is not None:
                        resp["tag"] = cmd_tag
                    send(resp)

            except Exception as e:
                resp = {"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)}
                if cmd_tag is not None:
                    resp["tag"] = cmd_tag
                send(resp)

    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc(limit=12)}")
        send({"ok": False, "event": "fatal", "error": str(e), "traceback": traceback.format_exc(limit=12)})
    finally:
        try:
            if cam is not None:
                log("cleanup: StopCapture/ReleaseBuf/CloseUninitCamera")
                acq_stop.set()
                try:
                    if acq_thread is not None:
                        acq_thread.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    cam.StopCapture()
                except Exception:
                    pass
                try:
                    cam.ReleaseBuf()
                except Exception:
                    pass
                try:
                    cam.CloseUninitCamera()
                except Exception:
                    pass
        except Exception:
            pass

        close_log()
