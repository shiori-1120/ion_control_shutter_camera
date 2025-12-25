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

    # The legacy camera stack expects `import lib.*` to resolve to src/camera/lib.
    # When running as a module from repo root, we need to prepend src/camera to sys.path.
    import sys
    from pathlib import Path

    camera_dir = Path(__file__).resolve().parent
    if str(camera_dir) not in sys.path:
        sys.path.insert(0, str(camera_dir))

    mode = str(cfg.get("mode") or "dry")  # 'dry' | 'real'

    roi = cfg.get("roi")
    bg_roi = cfg.get("bg_roi")
    exposure_s = float(cfg.get("exposure_s") or 0.001)
    frame_timeout_s = float(cfg.get("frame_timeout_s") or 1.0)
    bootstrap_n = int(cfg.get("bootstrap_n") or 10)

    tau_on = cfg.get("tau_on")
    tau_off = cfg.get("tau_off")

    prev_state: bool | None = None

    cam: Any | None = None

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
                try_load("dark", False)
        except Exception:
            dry_samples = []

    try:
        if mode == "dry":
            send({"ok": True, "event": "ready", "mode": "dry", "dry_samples": len(dry_samples)})

        elif mode == "real":
            # Import the real camera/analysis stack only in real mode.
            import numpy as np

            from .lib.analysis_profiles import generate_rois_from_image
            from .lib.ControlDevice import Control_qCMOScamera
            from .lib.thresholding import bootstrap_threshold_from_stream, classify_hysteresis, normalize_count

            cam = Control_qCMOScamera()
            cam.OpenCamera_GetHandle()
            # Full frame by default (ROI can be applied in software)
            cam.SetParameters(exposure_s)
            cam.StartCapture()

            # Bootstrap ROI + thresholds if missing
            frames: list[np.ndarray] = []
            for _ in range(max(1, bootstrap_n)):
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

            if name != "get_state":
                send({"ok": False, "event": "error", "error": f"unknown cmd: {name}"})
                continue

            timeout_s = float(cmd.get("timeout_s") or frame_timeout_s)

            try:
                if mode == "dry":
                    if dry_samples:
                        import numpy as np  # local import; used only when samples exist

                        arr, bright_label, name = random.choice(dry_samples)
                        try:
                            s_norm = float(np.mean(arr))
                        except Exception:
                            s_norm = float(random.gauss(50_000.0, 10_000.0))
                        send(
                            {
                                "ok": True,
                                "event": "state",
                                "bright": bool(bright_label),
                                "S_norm": s_norm,
                                "tau_on": None,
                                "tau_off": None,
                                "sample": name,
                            }
                        )
                        continue
                    # simple synthetic fallback
                    s_norm = float(random.gauss(50_000.0, 10_000.0))
                    if random.random() < 0.5:
                        s_norm *= 0.3
                    bright = bool(s_norm > 20_000.0)
                    send({"ok": True, "event": "state", "bright": bright, "S_norm": s_norm, "tau_on": None, "tau_off": None})
                    continue

                if cam is None or roi_t is None or tau_on is None or tau_off is None:
                    raise RuntimeError("Camera worker is not configured")

                # Wait for next frame
                ok, err = cam.wait_for_frame_ready(timeout_s)
                if not ok:
                    send({"ok": False, "event": "timeout", "error": str(err)})
                    continue

                _, frame = cam.GetLastFrame()
                frame_np = np.asarray(frame)
                norm = normalize_count(frame_np, roi_t, bg_roi=bg_roi_t, exposure_s=exposure_s)
                S_norm = float(norm["S_norm"])
                bright = classify_hysteresis(S_norm, prev_state_bright=prev_state, tau_on=float(tau_on), tau_off=float(tau_off))
                prev_state = bool(bright)

                send(
                    {
                        "ok": True,
                        "event": "state",
                        "bright": bool(bright),
                        "S_norm": S_norm,
                        "tau_on": float(tau_on),
                        "tau_off": float(tau_off),
                    }
                )

            except Exception as e:
                send({"ok": False, "event": "error", "error": str(e), "traceback": traceback.format_exc(limit=8)})

    except Exception as e:
        send({"ok": False, "event": "fatal", "error": str(e), "traceback": traceback.format_exc(limit=12)})
    finally:
        try:
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
                    cam.CloseUninitCamera()
                except Exception:
                    pass
        except Exception:
            pass
