from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..gui_support.image_utils import robust_gray_limits
from ..gui_support.worker_messages import format_worker_failure
from ..hardware import CameraQueueDevice, DaqQueueDevice, DaqSequenceCommand
from .roi_bootstrap import run_roi_bootstrap


@dataclass(frozen=True)
class RoiCheckResult:
    roi: list[int] | None


@dataclass(frozen=True)
class ThresholdStageResult:
    tau: float
    tau_on: float
    tau_off: float
    agreement: float
    bright_samples_n: int
    dark_samples_n: int
    samples_n: int
    sample_metric: str
    threshold: dict[str, Any]


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
    prefer_sample_path: str | None = None,
) -> RoiCheckResult:
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
    cam_device.send_get_frame(1.0, prefer_sample=(str(prefer_sample_path) if prefer_sample_path else None))
    DaqQueueDevice(cmd_q=daq_cmd_q, resp_q=daq_resp_q).run_sequence_once(
        DaqSequenceCommand(
            do_sequence=pulse_seq,
            ao_insert_index=-1,
            ao_width_ms=0.0,
            ao_rate_hz=float(ao_rate_hz),
            ao_v_high=5.0,
            ao_v_low=0.0,
        )
    )
    cam_resp = mpq_get_with_ui(cam_resp_q, 15, "Camera ROI frame")
    if not cam_resp.get("ok"):
        raise RuntimeError(
            format_worker_failure(
                cam_resp,
                label="Camera frame failed",
                log_path=cam_log_path,
            )
        )

    frame = np.asarray(cam_resp.get("frame"))

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


def run_threshold_stage(
    *,
    daq_cmd_q: Any,
    daq_resp_q: Any,
    cam_cmd_q: Any,
    cam_resp_q: Any,
    do_sequence: list[tuple[int, float]],
    roi: list[int],
    n_target: int,
    max_attempt: int,
    cam_exposure_s: float,
    ao_rate_hz: float,
    mpq_get_with_ui: Callable[[Any, float, str], dict[str, Any]],
    ui_pump: Callable[[], None] | None,
    status_cb: Callable[[str], None] | None,
    fig: Any,
    canvas: Any,
    out_dir: Any | None = None,
) -> ThresholdStageResult:
    # Derive a safe get_frame timeout based on exposure + total shot duration.
    seq_s = 0.0
    try:
        for step in do_sequence or []:
            if isinstance(step, (list, tuple)) and len(step) >= 2:
                seq_s += float(step[1])
    except Exception:
        seq_s = 0.0
    shot_timeout_s = max(1.5, float(seq_s) + float(cam_exposure_s) + 0.8)

    samples: list[float] = []
    profiles: list[np.ndarray] = []
    last_cam_event: str | None = None
    last_cam_error: str | None = None
    cam_timeout_count = 0

    if status_cb is not None:
        try:
            status_cb("Threshold: acquiring frames...")
        except Exception:
            pass
    if ui_pump is not None:
        try:
            ui_pump()
        except Exception:
            pass

    from src.camera.lib.image_ops import crop_roi
    cam_device = CameraQueueDevice(cmd_q=cam_cmd_q)

    for attempt_idx in range(int(max_attempt)):
        if len(samples) >= int(n_target):
            break

        cam_device.send_get_frame(float(shot_timeout_s))
        try:
            DaqQueueDevice(cmd_q=daq_cmd_q, resp_q=daq_resp_q).run_sequence_once(
                DaqSequenceCommand(
                    do_sequence=do_sequence,
                    ao_insert_index=-1,
                    ao_width_ms=0.0,
                    ao_rate_hz=float(ao_rate_hz),
                    ao_v_high=5.0,
                    ao_v_low=0.0,
                )
            )
        except Exception as e:
            raise RuntimeError(f"DAQ error: {e}")
        cam_resp = mpq_get_with_ui(cam_resp_q, 15, "Camera frame")
        if not cam_resp.get("ok"):
            last_cam_event = str(cam_resp.get("event") or "") or None
            last_cam_error = str(cam_resp.get("error") or "") or None
            if cam_resp.get("event") == "timeout":
                cam_timeout_count += 1
            continue

        frame = np.asarray(cam_resp.get("frame"))
        crop = crop_roi(np.asarray(frame), roi)
        if crop.size == 0:
            continue

        try:
            s = float(np.mean(np.asarray(crop, dtype=float)))
            samples.append(s)
            profiles.append(np.asarray(np.sum(np.asarray(crop, dtype=float), axis=0), dtype=float))
        except Exception:
            continue

        if status_cb is not None and (len(samples) % 10 == 0):
            try:
                status_cb(f"Threshold: {len(samples)}/{int(n_target)} frames")
            except Exception:
                pass
            if ui_pump is not None:
                try:
                    ui_pump()
                except Exception:
                    pass

    if len(samples) < max(5, min(10, int(n_target))):
        detail = f"Too few samples: {len(samples)}"
        if len(samples) == 0:
            detail += (
                f" | seq_s~{seq_s:.3f}s exposure_s~{float(cam_exposure_s):.3f}s"
                f" get_frame_timeout_s~{shot_timeout_s:.3f}s"
            )
            if cam_timeout_count:
                detail += f" | camera_timeouts={cam_timeout_count}"
            if last_cam_event:
                detail += f" | last_cam_event={last_cam_event}"
            if last_cam_error:
                detail += f" | last_cam_error={last_cam_error}"
        raise RuntimeError(detail)

    from src.camera.lib.thresholding import quick_threshold_from_samples

    th = quick_threshold_from_samples(list(samples))
    tau = float(th["tau"])
    tau_on = float(tau)
    tau_off = float(tau)

    bright_samples = [float(v) for v in samples if float(v) > tau]
    dark_samples = [float(v) for v in samples if float(v) <= tau]

    bright_profiles = [profiles[i] for i, v in enumerate(samples) if float(v) > tau]
    dark_profiles = [profiles[i] for i, v in enumerate(samples) if float(v) <= tau]

    try:
        from src.camera.lib.thresholding import classify_hysteresis

        prev: bool | None = None
        agree = 0
        total = 0
        for v in samples:
            v_f = float(v)
            simple = bool(v_f > tau)
            hys = bool(classify_hysteresis(v_f, prev_state_bright=prev, tau_on=tau_on, tau_off=tau_off))
            prev = hys
            agree += int(simple == hys)
            total += 1
        acc = (float(agree) / float(total)) if total > 0 else 0.0
    except Exception:
        acc = 0.0

    # Plot (same layout as before)
    try:
        fig.clear()
        ax_ph = fig.add_subplot(211)
        ax_s = fig.add_subplot(212)

        def _concat_profiles(ps: list[np.ndarray]) -> np.ndarray:
            arrs = []
            for p in ps:
                a = np.asarray(p, dtype=float)
                a = a[np.isfinite(a)]
                if a.size:
                    arrs.append(a)
            return np.concatenate(arrs) if arrs else np.asarray([], dtype=float)

        light_counts = _concat_profiles(bright_profiles)
        dark_counts = _concat_profiles(dark_profiles)
        combined = np.concatenate([c for c in (light_counts, dark_counts) if c.size > 0])
        if combined.size == 0:
            raise RuntimeError("No valid photon-count samples")

        try:
            tau_plot = float(tau) * float(roi[1])
        except Exception:
            tau_plot = float(tau)

        start = int(np.floor(float(np.nanmin(combined))))
        end = int(np.ceil(float(np.nanmax(combined))))
        try:
            start = int(min(start, np.floor(float(tau_plot))))
            end = int(max(end, np.ceil(float(tau_plot))))
        except Exception:
            pass
        bin_edges = np.arange(start - 0.5, end + 1.5, 1)

        if light_counts.size > 0:
            mean_light = float(np.mean(light_counts))
            ax_ph.hist(
                light_counts,
                bins=bin_edges,
                density=True,
                alpha=0.6,
                color="tab:orange",
                edgecolor="none",
                label=f"Light (mean={mean_light:.2f})",
            )
            ax_ph.axvline(mean_light, color="tab:orange", linestyle="--")
        if dark_counts.size > 0:
            mean_dark = float(np.mean(dark_counts))
            ax_ph.hist(
                dark_counts,
                bins=bin_edges,
                density=True,
                alpha=0.6,
                color="navy",
                edgecolor="none",
                label=f"Dark (mean={mean_dark:.2f})",
            )
            ax_ph.axvline(mean_dark, color="navy", linestyle="--")

        try:
            ax_ph.axvline(
                float(tau_plot),
                color="tab:red",
                linestyle="-",
                linewidth=2,
                label=f"Threshold (tau*yw={float(tau_plot):.2f})",
            )
        except Exception:
            pass
        ax_ph.set_xlabel("Photon Count (per-column sum; integer bins)")
        ax_ph.set_ylabel("Probability density")
        ax_ph.set_title(f"Photon Distribution (integrated over y-axis) | agree={acc*100:.1f}%")
        ax_ph.legend(loc="upper right")
        ax_ph.grid(True, alpha=0.3)

        try:
            s_all = np.asarray(samples, dtype=float)
            s_all = s_all[np.isfinite(s_all)]
        except Exception:
            s_all = np.asarray([], dtype=float)

        if s_all.size > 0:
            s_bright = np.asarray(bright_samples, dtype=float)
            s_dark = np.asarray(dark_samples, dtype=float)
            try:
                s_min = float(np.nanmin(s_all))
                s_max = float(np.nanmax(s_all))
                s_min = min(s_min, float(tau))
                s_max = max(s_max, float(tau))
                bins_s = max(10, min(80, int(np.sqrt(s_all.size)) * 4))
                edges_s = np.linspace(s_min, s_max, bins_s + 1)
            except Exception:
                edges_s = 50

            if s_bright.size > 0:
                ax_s.hist(
                    s_bright,
                    bins=edges_s,
                    density=True,
                    alpha=0.6,
                    color="tab:orange",
                    edgecolor="none",
                    label=f"roi_mean bright (n={int(s_bright.size)})",
                )
                ax_s.axvline(float(np.mean(s_bright)), color="tab:orange", linestyle="--")
            if s_dark.size > 0:
                ax_s.hist(
                    s_dark,
                    bins=edges_s,
                    density=True,
                    alpha=0.6,
                    color="navy",
                    edgecolor="none",
                    label=f"roi_mean dark (n={int(s_dark.size)})",
                )
                ax_s.axvline(float(np.mean(s_dark)), color="navy", linestyle="--")

            ax_s.axvline(float(tau), color="tab:red", linestyle="-", linewidth=2, label=f"tau={float(tau):.3g}")

        ax_s.set_xlabel("roi_mean (used for tau)")
        ax_s.set_ylabel("Probability density")
        ax_s.set_title("ROI-mean distribution")
        ax_s.legend(loc="upper right")
        ax_s.grid(True, alpha=0.3)

        fig.tight_layout()
        canvas.draw()
    except Exception:
        pass

    if out_dir is not None:
        try:
            import json

            (out_dir / "threshold.json").write_text(
                json.dumps(
                    {
                        "bright_samples_n": len(bright_samples),
                        "dark_samples_n": len(dark_samples),
                        "samples_n": int(len(samples)),
                        "roi": list(roi) if isinstance(roi, (list, tuple)) else None,
                        "sample_metric": "roi_mean",
                        "threshold": th,
                        "agreement": acc,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    return ThresholdStageResult(
        tau=float(tau),
        tau_on=float(tau_on),
        tau_off=float(tau_off),
        agreement=float(acc),
        bright_samples_n=int(len(bright_samples)),
        dark_samples_n=int(len(dark_samples)),
        samples_n=int(len(samples)),
        sample_metric="roi_mean",
        threshold=dict(th),
    )
