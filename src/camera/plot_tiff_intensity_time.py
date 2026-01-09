"""Plot ion blinking vs time from TIFF frames.

Goal:
    For a TIFF frame sequence where ions are aligned horizontally (along X),
    compute per-frame 1D profiles by vertical integration (sum over Y), then
    stack them over time to make a kymograph:

        X-axis: time [s]
        Y-axis: position [px]
        color : integrated intensity [a.u.]

Supported inputs:
  - A folder containing .tif/.tiff frames (sorted by filename)
  - A single TIFF file (single-page or multi-page)

Usage (PowerShell):
    # Folder input, dt inferred from TIFF exposure metadata if present,
    # otherwise assumes 100 ms exposure (dt=0.1s)
    python plot_tiff_intensity_time.py path\to\tiff_folder

    # Explicit FPS (dt = 1/fps)
    python plot_tiff_intensity_time.py path\to\tiff_folder --fps 10

    # Explicit dt [s]
    python plot_tiff_intensity_time.py path\to\tiff_folder --dt 0.1

    # Restrict ROI and x-range (useful to isolate ion chain area)
    python plot_tiff_intensity_time.py path\to\tiff_folder --roi 0:352,0:1008 --x-range 200:800

Outputs:
    - PNG kymograph next to the input.
    - Optionally NPY (profiles) and CSV (time-integrated trace).

Notes:
    - "縦に積算" is implemented as sum over axis=0 (Y).
    - If you want a single trace, use --save-csv (sums over X per frame).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import imageio.v3 as iio
except Exception:  # pragma: no cover
    iio = None  # type: ignore


def _is_tiff_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in (".tif", ".tiff")


def _maybe_parse_seconds_from_value(v) -> float | None:
    if v is None:
        return None

    if isinstance(v, tuple) and len(v) == 2:
        try:
            num = float(v[0])
            den = float(v[1])
            if den == 0:
                return None
            sec = num / den
            return sec if sec > 0 else None
        except Exception:
            return None

    if isinstance(v, (int, float)):
        fv = float(v)
        if fv <= 0:
            return None
        # Heuristic: treat large values as milliseconds.
        return fv / 1000.0 if fv > 10.0 else fv

    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "ignore")
        except Exception:
            return None

    if isinstance(v, str):
        s = v.strip().lower()
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(us|µs|ms|s|sec|secs|second|seconds)?", s)
        if not m:
            return None
        val = float(m.group(1))
        unit = (m.group(2) or "").strip()
        if val <= 0:
            return None
        if unit in ("us", "µs"):
            return val / 1_000_000.0
        if unit == "ms":
            return val / 1000.0
        return val

    return None


def _extract_exposure_seconds_from_description(desc: str) -> float | None:
    if not desc:
        return None

    s = desc.strip()

    # JSON case
    if s.startswith("{") and s.endswith("}"):
        try:
            obj = json.loads(s)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            lowered = {str(k).lower(): v for k, v in obj.items()}
            for key in (
                "exposure_time_s",
                "exposure_s",
                "exposure",
                "exposure_time",
                "exposuretime",
                "integration_time_s",
                "integration_s",
                "integration_time",
                "integrationtime",
            ):
                if key in lowered:
                    sec = _maybe_parse_seconds_from_value(lowered[key])
                    if sec:
                        return sec
            for key in (
                "exposure_time_ms",
                "exposure_ms",
                "integration_time_ms",
                "integration_ms",
            ):
                if key in lowered:
                    try:
                        sec = _maybe_parse_seconds_from_value(float(lowered[key]))
                    except Exception:
                        sec = None
                    if sec:
                        return sec

    # key=value case
    patterns = [
        r"exposure(?:[_\s-]*time)?\s*[:=]\s*([^\s;]+)",
        r"integration(?:[_\s-]*time)?\s*[:=]\s*([^\s;]+)",
        r"shutter(?:[_\s-]*time)?\s*[:=]\s*([^\s;]+)",
    ]
    low = s.lower()
    for pat in patterns:
        m = re.search(pat, low)
        if m:
            sec = _maybe_parse_seconds_from_value(m.group(1))
            if sec:
                return sec
    return None


def infer_exposure_seconds_from_tiff(path: str) -> float | None:
    """Best-effort infer exposure time (seconds) from a TIFF file."""
    # Prefer tifffile when available.
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]

            for tag_name in (
                "ExposureTime",
                "IntegrationTime",
                "ShutterSpeedValue",
            ):
                t = page.tags.get(tag_name)
                if t is not None:
                    sec = _maybe_parse_seconds_from_value(t.value)
                    if sec:
                        return sec

            desc_tag = page.tags.get("ImageDescription")
            if desc_tag is not None:
                desc_val = desc_tag.value
                if isinstance(desc_val, bytes):
                    desc_val = desc_val.decode("utf-8", "ignore")
                if isinstance(desc_val, str):
                    sec = _extract_exposure_seconds_from_description(desc_val)
                    if sec:
                        return sec

            for tag_name in ("Software", "Artist"):
                t = page.tags.get(tag_name)
                if t is not None and isinstance(t.value, (str, bytes)):
                    v = t.value.decode("utf-8", "ignore") if isinstance(t.value, bytes) else t.value
                    sec = _extract_exposure_seconds_from_description(str(v))
                    if sec:
                        return sec
    except Exception:
        pass

    # Fallback to imageio metadata if possible.
    if iio is not None:
        try:
            meta = iio.immeta(path)
            for k, v in meta.items():
                ks = str(k).lower()
                if "exposure" in ks or "integration" in ks or "shutter" in ks:
                    sec = _maybe_parse_seconds_from_value(v)
                    if sec:
                        return sec
            for key in ("description", "ImageDescription", "image_description"):
                if key in meta and isinstance(meta[key], (str, bytes)):
                    desc = meta[key].decode("utf-8", "ignore") if isinstance(meta[key], bytes) else meta[key]
                    sec = _extract_exposure_seconds_from_description(str(desc))
                    if sec:
                        return sec
        except Exception:
            pass

    return None


def infer_dt_seconds_from_first_frame(paths: Sequence[str]) -> float | None:
    if not paths:
        return None
    exp_s = infer_exposure_seconds_from_tiff(paths[0])
    if exp_s and exp_s > 0:
        return float(exp_s)
    return None


def _parse_slice_1d(text: str, max_len: int | None = None) -> slice:
    """Parse 'start:end' where start/end may be empty."""
    if ":" not in text:
        raise ValueError("slice must be like start:end")
    a, b = text.split(":", 1)
    start = int(a) if a.strip() else None
    stop = int(b) if b.strip() else None
    if max_len is not None:
        if start is not None:
            start = int(np.clip(start, 0, max_len))
        if stop is not None:
            stop = int(np.clip(stop, 0, max_len))
    return slice(start, stop)


def _parse_roi(text: str) -> Tuple[slice, slice]:
    """Parse 'y0:y1,x0:x1'."""
    if "," not in text:
        raise ValueError("ROI must be like y0:y1,x0:x1")
    ys, xs = text.split(",", 1)
    return _parse_slice_1d(ys.strip()), _parse_slice_1d(xs.strip())


def _iter_tiff_paths(input_path: str) -> List[str]:
    if os.path.isdir(input_path):
        names = [n for n in os.listdir(input_path) if _is_tiff_path(n)]
        names.sort()
        return [os.path.join(input_path, n) for n in names]
    if os.path.isfile(input_path) and _is_tiff_path(input_path):
        return [input_path]
    raise FileNotFoundError(f"Input not found or not a TIFF/folder: {input_path}")


def _read_tiff_frames(path: str) -> np.ndarray:
    """Read a TIFF file; returns (N,H,W) or (N,H,W,C)."""
    if iio is not None:
        arr = iio.imread(path)
        a = np.asarray(arr)
        # imageio may return (H,W) for single image
        if a.ndim in (2, 3):
            return a[None, ...]
        return a

    # Fallback to tifffile
    try:
        import tifffile  # type: ignore

        a = tifffile.imread(path)
        a = np.asarray(a)
        if a.ndim in (2, 3):
            return a[None, ...]
        return a
    except Exception as e:
        raise RuntimeError(
            "No TIFF reader available. Install imageio+imageio-ffmpeg or tifffile."
        ) from e


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert (H,W) or (H,W,C) to (H,W) float64."""
    a = np.asarray(frame)
    if a.ndim == 2:
        return a.astype(np.float64, copy=False)
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        rgb = a[..., :3].astype(np.float64, copy=False)
        return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    if a.ndim == 3 and a.shape[-1] == 1:
        return a[..., 0].astype(np.float64, copy=False)
    # Unknown layout; best-effort squeeze
    return np.squeeze(a).astype(np.float64, copy=False)


@dataclass
class SeriesResult:
    time_s: np.ndarray
    profiles: np.ndarray  # (N, X)
    intensity: np.ndarray  # (N,) = sum over X of profiles
    used_dt_s: float
    n_frames: int
    x_len: int


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    k = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(x.astype(np.float64, copy=False), k, mode="same")


def _median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = int(window)
    if w % 2 == 0:
        w += 1  # median window should be odd
    x = x.astype(np.float64, copy=False)
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    try:
        sw = np.lib.stride_tricks.sliding_window_view(xp, w)
        return np.median(sw, axis=-1)
    except Exception:
        # Fallback (slower) if sliding_window_view is unavailable
        out = np.empty_like(x)
        for i in range(x.shape[0]):
            out[i] = np.median(xp[i : i + w])
        return out


def smooth_along_axis(arr: np.ndarray, window: int, axis: int, method: str) -> np.ndarray:
    if window <= 1 or method == "none":
        return arr
    if method == "mean":
        fn = lambda v: _moving_average_1d(v, window)
    elif method == "median":
        fn = lambda v: _median_filter_1d(v, window)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    return np.apply_along_axis(fn, axis, arr)


def compute_profile_time_series(
    input_path: str,
    roi: Tuple[slice, slice] | None = None,
    x_range: slice | None = None,
    fps: float | None = None,
    dt_s: float | None = None,
    assume_exposure_ms: float = 100.0,
    smooth_time_frames: int = 3,
    smooth_x_px: int = 1,
    time_filter: str = "mean",
    x_filter: str = "none",
) -> SeriesResult:
    paths = _iter_tiff_paths(input_path)
    if not paths:
        raise FileNotFoundError("No TIFF files found")

    # Determine dt
    used_dt_s: float
    if dt_s is not None:
        used_dt_s = float(dt_s)
    elif fps is not None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        used_dt_s = 1.0 / float(fps)
    else:
        inferred = infer_dt_seconds_from_first_frame(paths)
        if inferred is not None:
            used_dt_s = float(inferred)
        else:
            used_dt_s = float(assume_exposure_ms) / 1000.0

    profiles_list: List[np.ndarray] = []

    # Folder mode: each file is a frame (common in this repo)
    if os.path.isdir(input_path):
        for p in paths:
            frames = _read_tiff_frames(p)
            if frames.shape[0] != 1:
                # If multi-page, treat as multiple frames
                for i in range(frames.shape[0]):
                    profiles_list.append(_frame_to_profile(frames[i], roi=roi, x_range=x_range))
            else:
                profiles_list.append(_frame_to_profile(frames[0], roi=roi, x_range=x_range))
    else:
        # Single TIFF file: possibly multi-page
        frames = _read_tiff_frames(paths[0])
        for i in range(frames.shape[0]):
            profiles_list.append(_frame_to_profile(frames[i], roi=roi, x_range=x_range))

    if not profiles_list:
        raise RuntimeError("No frames could be read")

    # Stack to (N, X)
    profiles = np.stack([np.asarray(p, dtype=np.float64) for p in profiles_list], axis=0)

    # Denoise: smooth along time (axis=0) and/or x (axis=1)
    if smooth_time_frames and smooth_time_frames > 1:
        profiles = smooth_along_axis(profiles, int(smooth_time_frames), axis=0, method=str(time_filter))
    if smooth_x_px and smooth_x_px > 1:
        profiles = smooth_along_axis(profiles, int(smooth_x_px), axis=1, method=str(x_filter))
    n = int(profiles.shape[0])
    x_len = int(profiles.shape[1])
    intensity = np.sum(profiles, axis=1)
    time_s = np.arange(n, dtype=np.float64) * used_dt_s
    return SeriesResult(
        time_s=time_s,
        profiles=profiles,
        intensity=intensity,
        used_dt_s=used_dt_s,
        n_frames=n,
        x_len=x_len,
    )


def _frame_to_profile(frame: np.ndarray, roi: Tuple[slice, slice] | None, x_range: slice | None) -> np.ndarray:
    g = _to_gray(frame)
    if roi is not None:
        ys, xs = roi
        g = g[ys, xs]

    # vertical integration (sum over Y) -> profile along X
    if g.ndim != 2:
        g = np.squeeze(g)
    if g.ndim != 2:
        raise ValueError(f"Unexpected frame shape after processing: {g.shape}")

    profile_x = np.sum(g, axis=0)

    if x_range is not None:
        profile_x = profile_x[x_range]

    return np.asarray(profile_x, dtype=np.float64)


def _default_out_base(input_path: str) -> str:
    if os.path.isdir(input_path):
        base = os.path.basename(os.path.normpath(input_path))
        return os.path.join(os.path.dirname(os.path.abspath(input_path)), base)
    return os.path.splitext(os.path.abspath(input_path))[0]


def save_csv(path: str, time_s: np.ndarray, intensity: np.ndarray) -> None:
    import csv

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "intensity"])
        for t, y in zip(time_s.tolist(), intensity.tolist()):
            w.writerow([t, y])


def save_plot_png(path: str, time_s: np.ndarray, intensity: np.ndarray, title: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to save plots. Install with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    ax.plot(time_s, intensity, lw=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Intensity [a.u.]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_kymograph_png(
    path: str,
    time_s: np.ndarray,
    profiles: np.ndarray,
    title: str,
    vmin_p: float = 1.0,
    vmax_p: float = 99.0,
    dpi: int = 200,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required to save plots. Install with: pip install matplotlib"
        ) from e

    # Robust display scaling
    vmin = float(np.percentile(profiles, vmin_p))
    vmax = float(np.percentile(profiles, vmax_p))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(np.min(profiles)), float(np.max(profiles))
        if vmax <= vmin:
            vmax = vmin + 1.0

    n, x_len = profiles.shape
    t0 = float(time_s[0]) if n > 0 else 0.0
    t1 = float(time_s[-1]) if n > 0 else 0.0

    # Make time horizontal: transpose to (X, N)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=int(dpi))
    im = ax.imshow(
        profiles.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=(t0, t1, 0, x_len),
    )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("X [px]")
    ax.set_title(title)

    # Time ticks every 1 second
    try:
        from matplotlib.ticker import MultipleLocator

        ax.xaxis.set_major_locator(MultipleLocator(1.0))
    except Exception:
        pass

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Integrated intensity [a.u.]")
    fig.tight_layout()
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Make a kymograph from TIFF frames by vertical integration (ion blinking vs time).")
    p.add_argument("input", help="Folder of TIFF frames or a TIFF file")
    p.add_argument("--fps", type=float, default=None, help="Explicit FPS for time axis")
    p.add_argument("--dt", type=float, default=None, help="Explicit dt [s] for time axis (overrides --fps)")
    p.add_argument(
        "--assume-exposure-ms",
        type=float,
        default=100.0,
        help="If dt cannot be inferred and --dt/--fps are not given, assume this exposure (default: 100ms)",
    )
    p.add_argument(
        "--smooth-time",
        type=int,
        default=3,
        help="Moving-average window (frames) along time for denoising (default: 3; set 1 to disable)",
    )
    p.add_argument(
        "--time-filter",
        choices=["mean", "median", "none"],
        default="mean",
        help="Time denoise method: mean (blurs a bit), median (edge-preserving), none",
    )
    p.add_argument(
        "--smooth-x",
        type=int,
        default=1,
        help="Moving-average window (pixels) along X for denoising (default: 1=off)",
    )
    p.add_argument(
        "--x-filter",
        choices=["mean", "median", "none"],
        default="none",
        help="X denoise method (default: none)",
    )
    p.add_argument("--dpi", type=int, default=200, help="Output PNG DPI (default: 200)")
    p.add_argument("--roi", type=str, default=None, help="ROI as y0:y1,x0:x1 (e.g. 0:352,0:1008)")
    p.add_argument("--x-range", type=str, default=None, help="X range to sum after vertical integration, e.g. 200:800")
    p.add_argument("--out", type=str, default=None, help="Output base path (without extension)")
    p.add_argument("--save-csv", action="store_true", help="Also save CSV alongside PNG")
    p.add_argument("--save-npy", action="store_true", help="Also save profiles as .npy (shape: N x X)")
    p.add_argument("--vmin-p", type=float, default=1.0, help="Percentile for display vmin (default: 1)")
    p.add_argument("--vmax-p", type=float, default=99.0, help="Percentile for display vmax (default: 99)")
    args = p.parse_args(argv)

    roi = _parse_roi(args.roi) if args.roi else None
    x_range = _parse_slice_1d(args.x_range) if args.x_range else None

    try:
        res = compute_profile_time_series(
            args.input,
            roi=roi,
            x_range=x_range,
            fps=args.fps,
            dt_s=args.dt,
            assume_exposure_ms=args.assume_exposure_ms,
            smooth_time_frames=int(args.smooth_time),
            smooth_x_px=int(args.smooth_x),
            time_filter=str(args.time_filter),
            x_filter=str(args.x_filter),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    out_base = args.out if args.out else _default_out_base(args.input) + "_kymograph"
    png_path = out_base + ".png"
    csv_path = out_base + "_sum.csv"
    npy_path = out_base + ".npy"

    title = f"Kymograph (N={res.n_frames}, dt={res.used_dt_s:.6g}s)"

    try:
        save_kymograph_png(
            png_path,
            res.time_s,
            res.profiles,
            title=title,
            vmin_p=float(args.vmin_p),
            vmax_p=float(args.vmax_p),
            dpi=int(args.dpi),
        )
        print(f"Saved plot -> {png_path}")
    except Exception as e:
        print(f"Plot save failed: {e}", file=sys.stderr)
        print("Hint: install matplotlib (pip install matplotlib)", file=sys.stderr)

    if args.save_csv:
        try:
            save_csv(csv_path, res.time_s, res.intensity)
            print(f"Saved CSV  -> {csv_path}")
        except Exception as e:
            print(f"CSV save failed: {e}", file=sys.stderr)

    if args.save_npy:
        try:
            np.save(npy_path, res.profiles)
            print(f"Saved NPY  -> {npy_path}  [shape={res.profiles.shape}]")
        except Exception as e:
            print(f"NPY save failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
