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
            try:
                init = init_real_camera(
                    cfg=cfg,
                    roi_t=roi_t,
                    bg_roi_t=bg_roi_t,
                    tau_on=tau_on,
                    tau_off=tau_off,
                    subarray_t=subarray_t,
                    log=log,
                    log_debug=log_debug,
                )
                cam = init["cam"]
                roi_t = init["roi_t"]
                bg_roi_t = init["bg_roi_t"]
                tau_on = init["tau_on"]
                tau_off = init["tau_off"]
                np = init["np"]
                normalize_count = init["normalize_count"]
                classify_hysteresis = init["classify_hysteresis"]
                send(init["ready_msg"])
                log("sent ready")
            except Exception as e:
                log("[Camera init exception detected]")
                log_worker_env()
                log(f"Exception: {e}\n{traceback.format_exc()}")
                raise
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

                log_debug("wait_for_frame_ready")
                ok, err = cam.wait_for_frame_ready(timeout_s)
                if not ok:
                    log(f"frame_timeout err={err}")
                    resp = {"ok": False, "event": "timeout", "error": str(err)}
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
