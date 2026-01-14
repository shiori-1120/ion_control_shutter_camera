from __future__ import annotations

from typing import Any


def init_real_camera(
    *,
    cfg: dict[str, Any],
    roi_t: tuple[int, int, int, int] | None,
    bg_roi_t: tuple[int, int, int, int] | None,
    tau_on: float | None,
    tau_off: float | None,
    subarray_t: tuple[int, int, int, int] | None,
    log,
    log_debug,
) -> dict[str, Any]:
    trigger_cfg = cfg.get("trigger")
    exposure_s = float(cfg.get("exposure_s") or 0.001)
    frame_timeout_s = float(cfg.get("frame_timeout_s") or 1.0)
    bootstrap_n = int(cfg.get("bootstrap_n") or 10)

    log("mode=real -> importing camera stack")
    import numpy as np

    from .lib.analysis_profiles import generate_rois_from_image
    from .lib.ControlDevice import Control_qCMOScamera
    from .lib.image_ops import crop_roi
    from .lib.thresholding import bootstrap_threshold_from_stream, classify_hysteresis, normalize_count

    log("creating Control_qCMOScamera")
    cam = Control_qCMOScamera(trigger_cfg=trigger_cfg, verbose=bool(cfg.get("verbose") or cfg.get("camera_verbose") or False))
    log("OpenCamera_GetHandle")
    cam.OpenCamera_GetHandle()
    log("SetParameters")
    if subarray_t is not None:
        xw, yw, xs, ys = map(int, subarray_t)
        cam.SetParameters(exposure_s, xw, yw, xs, ys)
        log(f"subarray applied: x={xs} y={ys} w={xw} h={yw}")
    else:
        cam.SetParameters(exposure_s)
    log("StartCapture")
    cam.StartCapture()

    if cfg.get("diagnostics_mode", False):
        frames: list[np.ndarray] = []
        for i in range(max(1, bootstrap_n)):
            log_debug(f"diagnostics: wait_for_frame_ready {i + 1}/{max(1, bootstrap_n)}")
            ok, err = cam.wait_for_frame_ready(frame_timeout_s)
            if not ok:
                raise RuntimeError(f"Camera timeout during diagnostics: {err}")
            _, frame = cam.GetLastFrame()
            frames.append(np.asarray(frame))
        ready_msg = {
            "ok": True,
            "event": "ready",
            "mode": "real",
            "diagnostics_mode": True,
            "frames_captured": len(frames),
            "exposure_s": float(exposure_s),
        }
        return {
            "cam": cam,
            "roi_t": roi_t,
            "bg_roi_t": bg_roi_t,
            "tau_on": tau_on,
            "tau_off": tau_off,
            "np": np,
            "normalize_count": normalize_count,
            "classify_hysteresis": classify_hysteresis,
            "ready_msg": ready_msg,
        }

    frames: list[np.ndarray] = []
    for i in range(max(1, bootstrap_n)):
        log_debug(f"bootstrap wait_for_frame_ready {i + 1}/{max(1, bootstrap_n)}")
        ok, err = cam.wait_for_frame_ready(frame_timeout_s)
        if not ok:
            raise RuntimeError(f"Camera timeout during bootstrap: {err}")
        _, frame = cam.GetLastFrame()
        frames.append(np.asarray(frame))

    if roi_t is None:
        rois = generate_rois_from_image(np.asarray(frames[-1]), plot=False)
        if not rois:
            raise RuntimeError("Failed to auto-detect ROI from image")
        best = None
        best_sum = None
        for r in rois:
            if not (isinstance(r, (list, tuple)) and len(r) == 4):
                continue
            xw, yw, xs, ys = map(int, r)
            cropped = crop_roi(frames[-1], (xw, yw, xs, ys))
            s = float(np.sum(cropped))
            if best_sum is None or s > best_sum:
                best_sum = s
                best = (xw, yw, xs, ys)
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

    ready_msg = {
        "ok": True,
        "event": "ready",
        "mode": "real",
        "roi": list(roi_t),
        "bg_roi": (list(bg_roi_t) if bg_roi_t else None),
        "tau_on": float(tau_on),
        "tau_off": float(tau_off),
        "exposure_s": float(exposure_s),
    }
    return {
        "cam": cam,
        "roi_t": roi_t,
        "bg_roi_t": bg_roi_t,
        "tau_on": tau_on,
        "tau_off": tau_off,
        "np": np,
        "normalize_count": normalize_count,
        "classify_hysteresis": classify_hysteresis,
        "ready_msg": ready_msg,
    }
