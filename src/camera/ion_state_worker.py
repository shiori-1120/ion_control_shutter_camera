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
import random
import time
import traceback
from pathlib import Path
from multiprocessing.queues import Queue
from typing import Any


def _limit_blas_threads() -> None:
    # Safety for online operation: avoid NumPy/SciPy consuming all cores and starving DAQ.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def ion_state_worker_main(cmd_q: Queue, resp_q: Queue, cfg: dict[str, Any]) -> None:
    _limit_blas_threads()

    log_path = cfg.get("log_path")
    _log_file: Any | None = None

    def log(msg: str) -> None:
        nonlocal _log_file
        if not log_path:
            return
        try:
            if _log_file is None:
                p = Path(str(log_path))
                p.parent.mkdir(parents=True, exist_ok=True)
                _log_file = open(p, "a", encoding="utf-8")
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            _log_file.write(f"[{ts}] {msg}\n")
            _log_file.flush()
        except Exception:
            pass

    # The legacy camera stack expects `import lib.*` to resolve to src/camera/lib.
    # When running as a module from repo root, we need to prepend src/camera to sys.path.
    import sys
    from pathlib import Path

    camera_dir = Path(__file__).resolve().parent
    if str(camera_dir) not in sys.path:
        sys.path.insert(0, str(camera_dir))

    log(f"worker start | pid={os.getpid()} | mode={cfg.get('mode')} | exposure_s={cfg.get('exposure_s')} | frame_timeout_s={cfg.get('frame_timeout_s')} | bootstrap_n={cfg.get('bootstrap_n')}")

    mode = str(cfg.get("mode") or "dry")  # 'dry' | 'real'

    roi = cfg.get("roi")
    bg_roi = cfg.get("bg_roi")
    exposure_s = float(cfg.get("exposure_s") or 0.001)
    frame_timeout_s = float(cfg.get("frame_timeout_s") or 1.0)
    bootstrap_n = int(cfg.get("bootstrap_n") or 10)

    tau_on = cfg.get("tau_on")
    tau_off = cfg.get("tau_off")

    trigger_cfg = cfg.get("trigger")
    cam_verbose = bool(cfg.get("verbose") or cfg.get("camera_verbose") or False)

    log(f"trigger_cfg={trigger_cfg} | cam_verbose={cam_verbose}")

    prev_state: bool | None = None

    cam: Any | None = None

    def _to_uint8_image(arr: Any) -> Any:
        """Convert arbitrary array-like image to uint8 in [0,255] for dry-mode tests."""
        try:
            import numpy as _np

            x = _np.asarray(arr)
            if x.size == 0:
                return x.astype(_np.uint8)

            # If already uint8, keep.
            if x.dtype == _np.uint8:
                return x

            # Handle float images in [0,1]
            x_f = x.astype(float)
            finite = x_f[_np.isfinite(x_f)]
            if finite.size == 0:
                return _np.zeros_like(x_f, dtype=_np.uint8)

            vmin = float(finite.min())
            vmax = float(finite.max())

            if 0.0 <= vmin and vmax <= 1.0:
                y = _np.clip(x_f, 0.0, 1.0) * 255.0
                return _np.asarray(_np.rint(y), dtype=_np.uint8)

            # If already roughly in [0,255], just clip.
            if -1.0 <= vmin and vmax <= 256.0:
                y = _np.clip(x_f, 0.0, 255.0)
                return _np.asarray(_np.rint(y), dtype=_np.uint8)

            # Otherwise, normalize robustly (percentiles) then scale to [0,255].
            p1 = float(_np.percentile(finite, 1))
            p99 = float(_np.percentile(finite, 99))
            if not _np.isfinite(p1) or not _np.isfinite(p99) or abs(p99 - p1) < 1e-12:
                y = _np.clip(x_f, 0.0, 255.0)
                return _np.asarray(_np.rint(y), dtype=_np.uint8)

            y = (x_f - p1) / (p99 - p1)
            y = _np.clip(y, 0.0, 1.0) * 255.0
            return _np.asarray(_np.rint(y), dtype=_np.uint8)
        except Exception:
            return arr

    # Imported lazily (real mode only)
    np = None  # type: ignore
    normalize_count = None  # type: ignore
    classify_hysteresis = None  # type: ignore

    def send(msg: dict[str, Any]) -> None:
        try:
            resp_q.put(msg)
        except Exception:
            pass

    def as_roi_tuple(x: Any) -> tuple[int, int, int, int] | None:
        if x is None:
            return None
        if isinstance(x, (list, tuple)) and len(x) == 4:
            return (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
        return None

    roi_t = as_roi_tuple(roi)
    bg_roi_t = as_roi_tuple(bg_roi)

    dry_samples: list[tuple[Any, bool, str]] = []
    dry_dir = cfg.get("dry_image_dir")
    if dry_dir:
        try:
            import numpy as np

            try:
                from PIL import Image  # type: ignore
            except Exception:
                Image = None  # type: ignore

            base = Path(dry_dir)
            if base.exists() and base.is_dir():
                def load_img(p: Path) -> Any:
                    if p.suffix.lower() == ".npy":
                        return np.load(p)
                    if Image is None:
                        raise RuntimeError("Pillow not available to load image")
                    return np.asarray(Image.open(p).convert("L"))

                def try_load(stem: str, is_bright: bool) -> None:
                    for ext in ("png", "jpg", "jpeg", "bmp", "tif", "tiff", "npy"):
                        for candidate in base.glob(f"{stem}*.{ext}"):
                            try:
                                arr = load_img(candidate)
                                dry_samples.append((arr, is_bright, candidate.name))
                                return
                            except Exception:
                                continue

                try_load("bright", True)
                # Used for dry ROI check (should look bright).
                try_load("roi_test", True)
                try_load("dark", False)
        except Exception:
            dry_samples = []

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
            from .lib.thresholding import bootstrap_threshold_from_stream, classify_hysteresis, normalize_count

            # store for command loop
            locals_np = np
            locals_norm = normalize_count
            locals_cls = classify_hysteresis

            log("creating Control_qCMOScamera")
            cam = Control_qCMOScamera(trigger_cfg=trigger_cfg, verbose=cam_verbose)
            log("OpenCamera_GetHandle")
            cam.OpenCamera_GetHandle()
            # Full frame by default (ROI can be applied in software)
            log("SetParameters")
            cam.SetParameters(exposure_s)
            log("StartCapture")
            cam.StartCapture()

            # Bootstrap ROI + thresholds if missing
            frames: list[np.ndarray] = []
            for i in range(max(1, bootstrap_n)):
                log(f"bootstrap wait_for_frame_ready {i+1}/{max(1, bootstrap_n)}")
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
                    cropped = frames[-1][ys : ys + yw, xs : xs + xw]
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
            if name in ("quit", "close"):
                send({"ok": True, "event": "closing"})
                break

            if name == "set_roi":
                try:
                    roi_new = as_roi_tuple(cmd.get("roi"))
                    if roi_new is None:
                        raise ValueError("set_roi requires roi=[xw,yw,xs,ys]")
                    roi_t = roi_new
                    # Reset hysteresis memory when ROI changes.
                    prev_state = None
                    send({"ok": True, "event": "roi", "roi": list(roi_t)})
                except Exception as e:
                    send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})
                continue

            if name == "set_bg_roi":
                try:
                    bg_new = as_roi_tuple(cmd.get("bg_roi"))
                    bg_roi_t = bg_new
                    prev_state = None
                    send({"ok": True, "event": "bg_roi", "bg_roi": (list(bg_roi_t) if bg_roi_t else None)})
                except Exception as e:
                    send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})
                continue

            if name == "set_threshold":
                try:
                    tau_on_new = cmd.get("tau_on")
                    tau_off_new = cmd.get("tau_off")
                    tau_new = cmd.get("tau")

                    if tau_new is not None:
                        tau = float(tau_new)
                        # Hysteresis disabled: use a single threshold.
                        tau_on_new = float(tau)
                        tau_off_new = float(tau)

                    if tau_on_new is None or tau_off_new is None:
                        raise ValueError("set_threshold requires tau or (tau_on and tau_off)")

                    tau_on = float(tau_on_new)
                    tau_off = float(tau_off_new)
                    prev_state = None

                    send({"ok": True, "event": "threshold", "tau_on": float(tau_on), "tau_off": float(tau_off)})
                except Exception as e:
                    send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})
                continue

            if name not in ("get_state", "get_frame"):
                send({"ok": False, "event": "error", "error": f"unknown cmd: {name}"})
                continue

            timeout_s = float(cmd.get("timeout_s") or frame_timeout_s)

            try:
                if mode == "dry":
                    if name == "get_frame":
                        # Force a specific sample if requested (useful for ROI check).
                        # This works even when dry_image_dir is not configured.
                        try:
                            prefer = cmd.get("prefer_sample")
                            if isinstance(prefer, str) and prefer.strip():
                                p = Path(prefer)
                                if p.exists() and p.is_file():
                                    import numpy as _np  # local import

                                    arr: Any
                                    if p.suffix.lower() == ".npy":
                                        arr = _np.load(p)
                                    else:
                                        try:
                                            from PIL import Image  # type: ignore

                                            arr = _np.asarray(Image.open(p).convert("L"))
                                        except Exception:
                                            arr = _np.load(p)
                                    frame = _to_uint8_image(arr)
                                    frame = _np.asarray(frame)
                                    send(
                                        {
                                            "ok": True,
                                            "event": "frame",
                                            "frame": frame,
                                            "bright": True,
                                            "S_norm": float(_np.mean(frame)) if frame.size else 0.0,
                                            "tau_on": None,
                                            "tau_off": None,
                                            "sample": str(p),
                                        }
                                    )
                                    continue
                        except Exception:
                            pass

                        # Best-effort: return a representative sample frame.
                        if dry_samples:
                            import numpy as _np  # local import

                            prefer = cmd.get("prefer_sample")
                            pick = None
                            if isinstance(prefer, str) and prefer.strip():
                                prefer_path = Path(prefer)
                                prefer_name = prefer_path.name.lower()
                                prefer_stem = prefer_path.stem.lower()
                                for a, b, n in dry_samples:
                                    if not isinstance(n, str):
                                        continue
                                    n_l = n.lower()
                                    if n_l == prefer_name:
                                        pick = (a, b, n)
                                        break
                                    try:
                                        if Path(n_l).stem == prefer_stem:
                                            pick = (a, b, n)
                                            break
                                    except Exception:
                                        continue

                            if pick is None:
                                pick = random.choice(dry_samples)

                            arr, bright_label, sample_name = pick
                            frame = _to_uint8_image(arr)
                            frame = _np.asarray(frame)
                            send(
                                {
                                    "ok": True,
                                    "event": "frame",
                                    "frame": frame,
                                    "bright": bool(bright_label),
                                    "S_norm": float(_np.mean(frame)) if frame.size else 0.0,
                                    "tau_on": None,
                                    "tau_off": None,
                                    "sample": sample_name,
                                }
                            )
                            continue
                        # synthetic fallback
                        import numpy as _np  # local import

                        is_bright = (random.random() < 0.5)
                        base = 180.0 if is_bright else 40.0
                        noise = _np.random.normal(loc=0.0, scale=18.0, size=(256, 256))
                        frame_f = base + noise
                        frame = _np.asarray(_np.clip(_np.rint(frame_f), 0, 255), dtype=_np.uint8)
                        send({"ok": True, "event": "frame", "frame": frame, "bright": is_bright, "S_norm": float(_np.mean(frame)), "tau_on": None, "tau_off": None})
                        continue

                    if dry_samples:
                        import numpy as np  # local import; used only when samples exist

                        arr, bright_label, name = random.choice(dry_samples)
                        try:
                            frame = np.asarray(_to_uint8_image(arr))
                            if roi_t is not None:
                                xw, yw, xs, ys = map(int, roi_t)
                                crop = frame[ys : ys + yw, xs : xs + xw]
                                s_norm = float(np.mean(crop)) if getattr(crop, "size", 0) else float(np.mean(frame))
                            else:
                                s_norm = float(np.mean(frame)) if frame.size else float(random.gauss(120.0, 30.0))
                        except Exception:
                            s_norm = float(random.gauss(120.0, 30.0))

                        # If a threshold has been applied (Step 2), classify using it.
                        # This keeps dry mode consistent with the sweep logic.
                        bright: bool
                        try:
                            if (tau_on is not None) and (tau_off is not None):
                                # If hysteresis is disabled (tau_on==tau_off), use a simple threshold.
                                if abs(float(tau_on) - float(tau_off)) < 1e-12:
                                    bright = bool(float(s_norm) > float(tau_on))
                                else:
                                    # Simple hysteresis (no external deps in dry mode).
                                    if prev_state is None:
                                        prev_state = bool(float(s_norm) > float(tau_on))
                                    if prev_state:
                                        prev_state = bool(float(s_norm) > float(tau_off))
                                    else:
                                        prev_state = bool(float(s_norm) > float(tau_on))
                                    bright = bool(prev_state)
                            else:
                                bright = bool(bright_label)
                        except Exception:
                            bright = bool(bright_label)
                        send(
                            {
                                "ok": True,
                                "event": "state",
                                "bright": bool(bright),
                                "S_norm": s_norm,
                                "tau_on": float(tau_on) if tau_on is not None else None,
                                "tau_off": float(tau_off) if tau_off is not None else None,
                                "sample": name,
                            }
                        )
                        continue
                    # simple synthetic fallback
                    s_norm = float(random.gauss(150.0, 25.0))
                    if random.random() < 0.5:
                        s_norm = float(random.gauss(50.0, 15.0))
                    s_norm = float(max(0.0, min(255.0, s_norm)))
                    bright = bool(s_norm > 100.0)
                    send({"ok": True, "event": "state", "bright": bright, "S_norm": s_norm, "tau_on": None, "tau_off": None})
                    continue

                if cam is None:
                    raise RuntimeError("Camera worker is not configured")

                # Wait for next frame
                ok, err = cam.wait_for_frame_ready(timeout_s)
                if not ok:
                    send({"ok": False, "event": "timeout", "error": str(err)})
                    continue

                _, frame = cam.GetLastFrame()
                frame_np = np.asarray(frame) if np is not None else frame

                # ROI/threshold might not be ready yet in edge cases.
                S_norm: float | None = None
                bright: bool | None = None
                if (normalize_count is not None) and (roi_t is not None):
                    norm = normalize_count(frame_np, roi_t, bg_roi=bg_roi_t, exposure_s=exposure_s)
                    S_norm = float(norm["S_norm"])
                    if (tau_on is not None) and (tau_off is not None):
                        # If tau_on==tau_off, treat it as a simple threshold (no hysteresis).
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
                    send(
                        {
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
                    )
                else:
                    send(
                        {
                            "ok": True,
                            "event": "state",
                            "bright": bool(bright) if bright is not None else False,
                            "S_norm": S_norm,
                            "tau_on": float(tau_on) if tau_on is not None else None,
                            "tau_off": float(tau_off) if tau_off is not None else None,
                        }
                    )

            except Exception as e:
                send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})

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

        try:
            if _log_file is not None:
                _log_file.close()
        except Exception:
            pass
