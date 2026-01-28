from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..gui_support.image_utils import robust_gray_limits
from ..gui_support.camera_capture import acquire_frame_with_ttl
from ..gui_support.worker_messages import format_worker_failure
from ..hardware import CameraQueueDevice, DaqQueueDevice, DaqSequenceCommand
from .roi_bootstrap import run_roi_bootstrap


@dataclass(frozen=True)
class RoiCheckResult:
    roi: list[int] | None


def run_roi_bootstrap_stage(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    nm_397: int,
    camera_trigger: int,
    roi_pulse_s: float,
    roi_idle_s: float,
    max_attempt: int,
    status_cb: Callable[[str], None] | None = None,
    ui_pump: Callable[[], None] | None = None,
) -> bool:
    """Run the ROI bootstrap stage (status update + optional UI pump + bootstrap loop)."""

    if status_cb is not None:
        try:
            status_cb("ROI bootstrap...")
        except Exception:
            pass

    if ui_pump is not None:
        try:
            ui_pump()
        except Exception:
            pass

    return run_roi_bootstrap(
        daq_cmd_q=daq_cmd_q,
        daq_resp_q=daq_resp_q,
        cam_cmd_q=cam_cmd_q,
        cam_resp_q=cam_resp_q,
        nm_397=nm_397,
        camera_trigger=camera_trigger,
        roi_pulse_s=roi_pulse_s,
        roi_idle_s=roi_idle_s,
        max_attempt=max_attempt,
        status_cb=status_cb,
    )


def run_roi_check_stage(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    pulse_seq: list[tuple[int, float]],
    ao_rate_hz: float,
    out_dir: Any,
    cam_log_path: str | None,
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]],
    ui_pump: Callable[[], None] | None,
    status_cb: Callable[[str], None] | None,
    fig: Any,
    canvas: Any,
    max_attempt: int,
    prefer_sample_path: str | None = None,
) -> RoiCheckResult:
    import gc
    import logging
    import time as _time

    logging.info("[ROI_CHECK] === run_roi_check_stage: start ===")
    if status_cb is not None:
        try:
            status_cb("ROI: acquiring frame...")
        except Exception:
            pass
    if ui_pump is not None:
        try:
            ui_pump()
        except Exception:
            pass

    cam_device = CameraQueueDevice(cmd_q=cam_cmd_q)
    daq_device = DaqQueueDevice(cmd_q=daq_cmd_q, resp_q=daq_resp_q)
    frame = None
    cam_resp = None

    # --- カメラリソース/GC/sleepの事前アプローチをログ表示 ---
    try:
        import os as _os
        do_close = _os.environ.get("ION_CONTROL_ROI_PRE_CLOSE", "").strip() == "1"
        if do_close:
            logging.info("[ROI_CHECK] [PRE] Sending camera close command before ROI check.")
            cam_device.close()
            _time.sleep(0.2)
        else:
            logging.info("[ROI_CHECK] [PRE] Skipping camera close (set ION_CONTROL_ROI_PRE_CLOSE=1 to enable).")
        logging.info("[ROI_CHECK] [PRE] Resetting ROI to full-frame before ROI check.")
        cam_device.set_roi(None)
        _time.sleep(0.2)
        logging.info("[ROI_CHECK] [PRE] Forcing GC and sleep to ensure camera resource release.")
        gc.collect()
        _time.sleep(1.0)
        logging.info("[ROI_CHECK] [PRE] GC collected and slept 1s.")
    except Exception as e:
        logging.error(f"[ROI_CHECK] [PRE] Camera pre-initialization failed: {e}")

    # --- worker起動前のカメラopen/closeテスト(任意) ---
    try:
        import os as _os

        if _os.environ.get("ION_CONTROL_CAMERA_PRECHECK", "").strip() == "1":
            from src.camera.lib.ControlDevice import Control_qCMOScamera

            logging.info("[ROI_CHECK] [PRE] Camera open/close test before worker launch (verifying resource release)...")
            cam_test = None
            try:
                cam_test = Control_qCMOScamera(verbose=True)
                ok, info = cam_test.check_connection(retries=1, delay=0.2, try_open=True)
                logging.info(f"[ROI_CHECK] [PRE] Camera open/close test result: ok={ok}, info={info}")
            except Exception as e:
                logging.error(f"[ROI_CHECK] [PRE] Camera open/close test failed: {e}")
            finally:
                if cam_test is not None:
                    try:
                        cam_test.CloseUninitCamera()
                        del cam_test
                    except Exception:
                        pass
                gc.collect()
                _time.sleep(0.5)
        else:
            logging.info("[ROI_CHECK] [PRE] Skipping camera open/close test (set ION_CONTROL_CAMERA_PRECHECK=1 to enable).")
    except Exception as e:
        logging.error(f"[ROI_CHECK] [PRE] Camera open/close test block failed: {e}")

    # --- worker起動前の状況をログ ---
    logging.info(f"[ROI_CHECK] [PRE] Just before ROI worker: cam_device={cam_device}, daq_device={daq_device}")

    # --- Main ROI check (single frame, retry TTL until success) ---
    frame_timeout_s = 1.0
    resp_timeout_s = frame_timeout_s + 2.0

    def _send_get_frame(timeout_s: float, prefer_sample: str | None) -> None:
        cam_device.send_get_frame(float(timeout_s), prefer_sample=(str(prefer_sample) if prefer_sample else None))

    def _run_ttl() -> None:
        daq_device.run_sequence_once(
            DaqSequenceCommand(
                do_sequence=pulse_seq,
                ao_insert_index=-1,
                ao_width_ms=0.0,
            )
        )

    def _wait_resp(timeout_s: float, label: str) -> dict[str, Any]:
        return mpq_get_with_ui(cam_resp_q, float(timeout_s), label)

    def _log(msg: str) -> None:
        logging.info(f"[ROI_CHECK] {msg}")

    resp_pack = acquire_frame_with_ttl(
        send_get_frame=_send_get_frame,
        run_ttl=_run_ttl,
        wait_resp=_wait_resp,
        max_attempt=max_attempt,
        frame_timeout_s=frame_timeout_s,
        resp_timeout_s=resp_timeout_s,
        prefer_sample_path=prefer_sample_path,
        sleep_s=0.05,
        log_cb=_log,
    )
    cam_resp = resp_pack.get("resp") if isinstance(resp_pack.get("resp"), dict) else None
    if resp_pack.get("ok") and cam_resp is not None:
        frame = np.asarray(cam_resp.get("frame"))
        logging.info("[ROI_CHECK] Frame acquired successfully.")

    # --- worker稼働後の状況をログ ---
    logging.info("[ROI_CHECK] === run_roi_check_stage: end ===")
    if frame is None or not (cam_resp and cam_resp.get("ok")):
        logging.error(f"[ROI_CHECK] Camera frame acquisition failed after {max_attempt} attempts.")
        raise RuntimeError(
            format_worker_failure(
                cam_resp or {},
                label="Camera frame failed",
                log_path=cam_log_path,
            )
        )

    roi: list[int] | None = None
    try:
        from src.camera.lib.analysis_profiles import generate_rois_from_image
        from src.camera.lib.image_ops import crop_roi

        rois = generate_rois_from_image(np.asarray(frame), plot=False)
        best: list[int] | None = None
        best_sum: float | None = None
        for r in rois or []:
            if not (isinstance(r, (list, tuple)) and len(r) == 4):
                continue
            xw, yw, xs, ys = map(int, r)
            crop = crop_roi(np.asarray(frame), (xw, yw, xs, ys))
            if crop.size == 0:
                continue
            s = float(np.sum(crop))
            if best_sum is None or s > best_sum:
                best_sum = s
                best = [int(xw), int(yw), int(xs), int(ys)]
        roi = best
    except Exception:
        roi = None

    if roi is None:
        r = cam_resp.get("roi")
        if isinstance(r, (list, tuple)) and len(r) == 4:
            try:
                roi = [int(r[0]), int(r[1]), int(r[2]), int(r[3])]
            except Exception:
                roi = None

    # Save snapshot (best-effort)
    try:
        p = Path(out_dir) / "roi_check.npy"
        np.save(p, frame)
    except Exception:
        pass

    # Plot image only (photon distributions belong to Threshold stage)
    try:
        fig.clear()
        try:
            gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0])
            ax_img = fig.add_subplot(gs[:, 0])
            ax_x = fig.add_subplot(gs[0, 1])
            ax_y = fig.add_subplot(gs[1, 1])

            vmin, vmax = robust_gray_limits(frame)
            ax_img.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            ax_img.set_title("ROI check")
            ax_img.set_axis_off()

            if isinstance(roi, (list, tuple)) and len(roi) == 4:
                try:
                    xw, yw, xs, ys = map(int, roi)
                    from matplotlib.patches import Rectangle

                    ax_img.add_patch(Rectangle((xs, ys), xw, yw, fill=False, edgecolor="tab:red", linewidth=2))
                    # Background ROI: same size, top-left corner.
                    ax_img.add_patch(Rectangle((0, 0), xw, yw, fill=False, edgecolor="tab:blue", linewidth=2))
                except Exception:
                    pass

            # best-effort profile fits
            try:
                from src.camera.lib.analysis_profiles import lorentz_fit_profiles

                results = lorentz_fit_profiles(np.asarray(frame), plot=False) or {}
                horiz = results.get("horizontal") or {}
                vert = results.get("vertical") or {}

                if isinstance(horiz, dict) and horiz.get("profile") is not None:
                    x_prof = np.asarray(horiz.get("profile"), dtype=float)
                    x_axis = (
                        np.asarray(horiz.get("x"), dtype=float)
                        if horiz.get("x") is not None
                        else np.arange(len(x_prof))
                    )
                    ax_x.plot(x_axis, x_prof, color="tab:blue", linewidth=1.0, label="profile")
                    if horiz.get("fitted") is not None:
                        ax_x.plot(
                            x_axis,
                            np.asarray(horiz.get("fitted"), dtype=float),
                            color="tab:orange",
                            linewidth=1.5,
                            label="fit",
                        )
                    fwhms = horiz.get("fwhms")
                    title = "X profile"
                    try:
                        if isinstance(fwhms, (list, tuple)) and fwhms:
                            title += f" (FWHM~{float(np.mean([float(w) for w in fwhms])):.1f}px)"
                    except Exception:
                        pass
                    ax_x.set_title(title)
                    ax_x.grid(True, alpha=0.2)
                    ax_x.tick_params(labelsize=8)
                    try:
                        ax_x.legend(fontsize=7, loc="best")
                    except Exception:
                        pass
                else:
                    ax_x.set_title("X profile (fit failed)")
                    ax_x.set_axis_off()

                if isinstance(vert, dict) and vert.get("profile") is not None:
                    y_prof = np.asarray(vert.get("profile"), dtype=float)
                    y_axis = (
                        np.asarray(vert.get("x"), dtype=float)
                        if vert.get("x") is not None
                        else np.arange(len(y_prof))
                    )
                    ax_y.plot(y_axis, y_prof, color="tab:blue", linewidth=1.0, label="profile")
                    if vert.get("fitted") is not None:
                        ax_y.plot(
                            y_axis,
                            np.asarray(vert.get("fitted"), dtype=float),
                            color="tab:orange",
                            linewidth=1.5,
                            label="fit",
                        )
                    title = "Y profile"
                    try:
                        if vert.get("fwhm") is not None:
                            title += f" (FWHM~{float(vert.get('fwhm')):.1f}px)"
                    except Exception:
                        pass
                    ax_y.set_title(title)
                    ax_y.grid(True, alpha=0.2)
                    ax_y.tick_params(labelsize=8)
                    try:
                        ax_y.legend(fontsize=7, loc="best")
                    except Exception:
                        pass
                else:
                    ax_y.set_title("Y profile (fit failed)")
                    ax_y.set_axis_off()
            except Exception:
                ax_x.set_title("profiles unavailable")
                ax_x.set_axis_off()
                ax_y.set_axis_off()

            fig.tight_layout()
            canvas.draw()
        except Exception:
            fig.clear()
            ax = fig.add_subplot(111)
            vmin, vmax = robust_gray_limits(frame)
            ax.imshow(frame, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_title("ROI check")
            ax.set_axis_off()
            fig.tight_layout()
            canvas.draw()
    except Exception:
        pass

    return RoiCheckResult(roi=roi)
