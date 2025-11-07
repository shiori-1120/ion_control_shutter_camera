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


def save_mp4(frames: np.ndarray, out_path: str, fps: int) -> None:
    if iio is None:
        raise RuntimeError(
            "imageio is required: pip install imageio imageio-ffmpeg")
    # 強制的に .mp4 にする
    root, ext = os.path.splitext(out_path)
    if ext.lower() != ".mp4":
        out_path = root + ".mp4"
    # 失敗時のフォールバックは行わない
    iio.imwrite(out_path, frames, fps=fps, codec="libx264")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a folder of TIFF frames to MP4 or NPY.")
    parser.add_argument(
        "input", help="Path to a folder containing .tif/.tiff frames")
    parser.add_argument(
        "--mode", choices=["mp4", "npy"], default="mp4", help="Conversion mode (default: mp4)")
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
    frames = _load_tiff_folder(inp)
    base = os.path.basename(os.path.normpath(inp))

    frames = _to_uint8(frames, pmin=DEFAULT_PMIN,
                       pmax=DEFAULT_PMAX, global_scale=DEFAULT_GLOBAL_SCALE)
    frames = _ensure_rgb(frames)
    # Avoid macro_block_size resize by padding to multiples of 16
    frames = _pad_to_mod(frames, mod=16)

    out = os.path.join(os.path.dirname(os.path.abspath(inp)), f"{base}.mp4")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    save_mp4(frames, out, fps=max(1, int(DEFAULT_FPS)))
    # out 変数は拡張子が .mp4 に強制されている可能性があるので再構築
    out_final = os.path.splitext(out)[0] + ".mp4"
    h, w = frames.shape[1], frames.shape[2]
    print(
        f"Saved MP4 -> {out_final}  [frames={frames.shape[0]}, size={w}x{h}, fps={DEFAULT_FPS}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
