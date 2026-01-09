"""
Convert a folder of TIFF frames either to an MP4 video or to NPY files.

Default: MP4. You can choose with --mode {mp4,npy}.

Usage (PowerShell):
    # MP4 (default)
    python visualize_tif.py path\to\tiff_frames_folder
    # NPY conversion
    python visualize_tif.py path\to\tiff_frames_folder --mode npy
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import re
from typing import List, Tuple

import numpy as np

try:
    import imageio.v3 as iio
except Exception as e:  # pragma: no cover
    iio = None  # type: ignore

# Optional import for NPY conversion mode
try:
    from .tif_to_npy import convert_folder_tif_to_npy  # when run as a module
except Exception:
    try:
        # when run as a script in this folder
        from tif_to_npy import convert_folder_tif_to_npy
    except Exception:
        convert_folder_tif_to_npy = None  # type: ignore

# Defaults (no CLI flags)
DEFAULT_FPS: int = 20
DEFAULT_PMIN: float = 1.0
DEFAULT_PMAX: float = 99.0
DEFAULT_GLOBAL_SCALE: bool = False  # per-frame scaling by default


def _maybe_parse_seconds_from_value(v) -> float | None:
    """Best-effort conversion to seconds.

    Accepts:
      - float/int -> seconds (if <= 10 assumed seconds; if > 10 assume ms)
      - (num, den) rational tuple -> seconds
      - strings like "10ms", "0.01 s", "10000us"
    """
    if v is None:
        return None

    # Rational (numerator, denominator)
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
        # Heuristic: many cameras store exposure in ms; if it's large assume ms.
        # e.g. 10, 20, 50 could be ms; but could also be seconds.
        # We'll treat >10 as ms to avoid 20s exposures mapping to 0.05fps.
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
        # default seconds
        return val

    return None


def _extract_exposure_seconds_from_description(desc: str) -> float | None:
    """Try to parse exposure/integration time from ImageDescription-like text."""
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
            # common keys (case-insensitive)
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
                    sec = _maybe_parse_seconds_from_value(float(lowered[key]))
                    if sec:
                        # _maybe_parse_seconds_from_value treats >10 as ms already
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
    # Try tifffile first (better access to tags)
    try:
        import tifffile  # type: ignore

        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]

            # Direct tag names (depends on writer)
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

            # ImageDescription often contains camera settings
            desc_tag = page.tags.get("ImageDescription")
            if desc_tag is not None:
                desc_val = desc_tag.value
                if isinstance(desc_val, bytes):
                    desc_val = desc_val.decode("utf-8", "ignore")
                if isinstance(desc_val, str):
                    sec = _extract_exposure_seconds_from_description(desc_val)
                    if sec:
                        return sec

            # Some writers store metadata in Software tag etc.
            for tag_name in ("Software", "Artist"):
                t = page.tags.get(tag_name)
                if t is not None and isinstance(t.value, (str, bytes)):
                    v = t.value.decode("utf-8", "ignore") if isinstance(t.value, bytes) else t.value
                    sec = _extract_exposure_seconds_from_description(str(v))
                    if sec:
                        return sec
    except Exception:
        pass

    # Fallback to imageio metadata if available
    if iio is not None:
        try:
            meta = iio.immeta(path)
            for k, v in meta.items():
                ks = str(k).lower()
                if "exposure" in ks or "integration" in ks or "shutter" in ks:
                    sec = _maybe_parse_seconds_from_value(v)
                    if sec:
                        return sec
            # some plugins provide a description field
            for key in ("description", "ImageDescription", "image_description"):
                if key in meta and isinstance(meta[key], (str, bytes)):
                    desc = meta[key].decode("utf-8", "ignore") if isinstance(meta[key], bytes) else meta[key]
                    sec = _extract_exposure_seconds_from_description(str(desc))
                    if sec:
                        return sec
        except Exception:
            pass
    return None


def infer_fps_from_first_frame(folder: str) -> Tuple[float | None, float | None, str | None]:
    """Return (fps, exposure_seconds, tiff_path) inferred from the first TIFF in folder."""
    names = [n for n in os.listdir(folder) if _is_tiff_path(n)]
    if not names:
        return (None, None, None)
    names.sort()
    first_path = os.path.join(folder, names[0])
    exp_s = infer_exposure_seconds_from_tiff(first_path)
    if exp_s is None or exp_s <= 0:
        return (None, None, first_path)
    fps = 1.0 / exp_s
    # Guardrails
    if not np.isfinite(fps) or fps <= 0:
        return (None, exp_s, first_path)
    fps = float(np.clip(fps, 0.1, 240.0))
    return (fps, exp_s, first_path)


def _is_tiff_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in (".tif", ".tiff")


def _load_tiff_folder(folder: str) -> np.ndarray:
    if iio is None:
        raise RuntimeError(
            "imageio is required: pip install imageio imageio-ffmpeg")
    names = [n for n in os.listdir(folder) if _is_tiff_path(n)]
    if not names:
        raise FileNotFoundError("No .tif/.tiff files found in folder")
    names.sort()
    frames: List[np.ndarray] = []
    for n in names:
        p = os.path.join(folder, n)
        img = iio.imread(p)
        frames.append(np.asarray(img))
    # Stack along first axis
    # Broadcast grayscale to shape (N,H,W)
    arr = np.stack(frames, axis=0)
    return arr


def _to_uint8(frames: np.ndarray, pmin: float, pmax: float, global_scale: bool) -> np.ndarray:
    """Scale frames to uint8 via percentile clipping.
    frames: (N,H,W) or (N,H,W,C) numeric array.
    """
    f = np.asarray(frames)
    if f.dtype == np.uint8:
        return f
    # If color given, compute percentiles on luminance approximation
    if f.ndim == 4 and f.shape[-1] in (3, 4):
        gray = np.dot(f[..., :3], np.array([0.2126, 0.7152, 0.0722]))
    else:
        gray = f if f.ndim == 3 else f[..., 0]

    if global_scale:
        vmin = np.percentile(gray, pmin)
        vmax = np.percentile(gray, pmax)
        vmin = float(vmin)
        vmax = float(max(vmin + 1e-9, vmax))
        g = np.clip((f - vmin) / (vmax - vmin), 0, 1)
        out = (g * 255.0 + 0.5).astype(np.uint8)
    else:
        # Per-frame scaling
        out_list: List[np.ndarray] = []
        for i in range(f.shape[0]):
            gi = gray[i]
            vmin = float(np.percentile(gi, pmin))
            vmax = float(np.percentile(gi, pmax))
            vmax = float(max(vmin + 1e-9, vmax))
            g = np.clip((f[i] - vmin) / (vmax - vmin), 0, 1)
            out_list.append((g * 255.0 + 0.5).astype(np.uint8))
        out = np.stack(out_list, axis=0)
    return out


def _ensure_rgb(frames: np.ndarray) -> np.ndarray:
    # Convert (N,H,W) -> (N,H,W,3)
    if frames.ndim == 3:
        return np.repeat(frames[..., None], 3, axis=-1)
    if frames.ndim == 4 and frames.shape[-1] == 1:
        return np.repeat(frames, 3, axis=-1)
    if frames.ndim == 4 and frames.shape[-1] == 4:
        return frames[..., :3]
    return frames


def _pad_to_mod(frames: np.ndarray, mod: int = 16) -> np.ndarray:
    """Pad spatial dims (H,W) to be divisible by `mod` with zeros (black borders).
    Keeps dtype and channels; returns original array if already divisible.
    """
    if frames.ndim not in (3, 4):
        return frames
    N = frames.shape[0]
    H = frames.shape[1]
    W = frames.shape[2]
    Hn = ((H + mod - 1) // mod) * mod
    Wn = ((W + mod - 1) // mod) * mod
    if Hn == H and Wn == W:
        return frames
    if frames.ndim == 3:
        out = np.zeros((N, Hn, Wn), dtype=frames.dtype)
        out[:, :H, :W] = frames
        return out
    else:
        C = frames.shape[3]
        out = np.zeros((N, Hn, Wn, C), dtype=frames.dtype)
        out[:, :H, :W, :] = frames
        return out


def save_mp4(frames: np.ndarray, out_path: str, fps: float) -> None:
    if iio is None:
        raise RuntimeError(
            "imageio is required: pip install imageio imageio-ffmpeg")
    # 強制的に .mp4 にする
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".mp4":
        out_path = root + ".mp4"
    # 失敗時のフォールバックは行わない
    iio.imwrite(out_path, frames, fps=float(fps), codec="libx264")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a folder of TIFF frames to MP4 or NPY.")
    parser.add_argument(
        "input", help="Path to a folder containing .tif/.tiff frames")
    parser.add_argument(
        "--mode", choices=["mp4", "npy"], default="mp4", help="Conversion mode (default: mp4)")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "FPS for MP4 output. If omitted, tries to infer from TIFF exposure time metadata; "
            f"falls back to DEFAULT_FPS={DEFAULT_FPS}."
        ),
    )
    args = parser.parse_args(argv)

    inp = args.input
    if not os.path.exists(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    if not os.path.isdir(inp):
        print("Input must be a folder containing .tif/.tiff frames.", file=sys.stderr)
        return 2

    # NPY conversion mode
    if args.mode == "npy":
        if convert_folder_tif_to_npy is None:
            print(
                "NPY conversion is unavailable: tif_to_npy module not found.", file=sys.stderr)
            return 2
        try:
            convert_folder_tif_to_npy(inp)
            return 0
        except Exception as e:
            print(f"Error during NPY conversion: {e}", file=sys.stderr)
            return 1

    # MP4 mode (default)
    # Decide FPS
    fps = args.fps
    inferred_exp_s: float | None = None
    if fps is None:
        inferred_fps, inferred_exp_s, inferred_path = infer_fps_from_first_frame(inp)
        if inferred_fps is not None:
            fps = inferred_fps
        else:
            fps = float(DEFAULT_FPS)

    frames = _load_tiff_folder(inp)
    base = os.path.basename(os.path.normpath(inp))

    frames = _to_uint8(frames, pmin=DEFAULT_PMIN,
                       pmax=DEFAULT_PMAX, global_scale=DEFAULT_GLOBAL_SCALE)
    frames = _ensure_rgb(frames)
    # Avoid macro_block_size resize by padding to multiples of 16
    frames = _pad_to_mod(frames, mod=16)

    out = os.path.join(os.path.dirname(os.path.abspath(inp)), f"{base}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_mp4(frames, out, fps=fps)
    # out 変数は拡張子が .mp4 に強制されている可能性があるので再構築
    out_final = os.path.splitext(out)[0] + ".mp4"
    h, w = frames.shape[1], frames.shape[2]
    exp_str = "" if inferred_exp_s is None else f", exposure={inferred_exp_s:.6g}s"
    print(
        f"Saved MP4 -> {out_final}  [frames={frames.shape[0]}, size={w}x{h}, fps={fps:.6g}{exp_str}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
