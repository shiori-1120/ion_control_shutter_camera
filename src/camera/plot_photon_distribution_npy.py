"""
plot_photon_distribution_npy.py

Purpose:
  Build distributions from a folder of .npy frames.
  - metric=pixel: pixel-wise photon-count distribution (plot_photon_distribution)
  - metric=S: ROI-integrated S_norm distribution (normalize_count)

Usage:
  python src\\camera\\plot_photon_distribution_npy.py "C:\\path\\to\\sequence_capture\\YYYYMMDD_HHMMSS"
  python src\\camera\\plot_photon_distribution_npy.py "C:\\path\\to\\raw-data" --metric S

Output:
  Saves a PNG into a "plots" subfolder under the input folder by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

try:
    from src.camera.lib.plotting import plot_photon_distribution
    from src.camera.lib.thresholding import normalize_count
    from src.camera.lib.analysis_profiles import generate_rois_from_image
except ImportError:
    from lib.plotting import plot_photon_distribution
    from lib.thresholding import normalize_count
    from lib.analysis_profiles import generate_rois_from_image


def list_npy_files(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() == ".npy"]


def resolve_input_dir(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Input path not found: {p}")
    if p.is_file():
        raise ValueError(f"Input must be a folder, not a file: {p}")
    if p.name.lower() == "raw-data":
        return p
    candidate = p / "raw-data"
    if candidate.is_dir():
        return candidate
    if any(child.suffix.lower() == ".npy" for child in p.iterdir()):
        return p
    raise FileNotFoundError(f"No .npy files found under: {p}")


def load_frames(npy_paths: List[Path]) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    for p in npy_paths:
        try:
            arr = np.load(p, allow_pickle=False)
        except Exception:
            continue
        frames.append(arr)
    if not frames:
        raise RuntimeError("No .npy frames could be loaded.")
    return frames


def _pick_best_roi(frame: np.ndarray) -> List[int] | None:
    rois = generate_rois_from_image(np.asarray(frame), plot=False)
    if not rois:
        return None
    best = None
    best_sum = None
    for r in rois:
        if not (isinstance(r, (list, tuple)) and len(r) == 4):
            continue
        xw, yw, xs, ys = map(int, r)
        cropped = np.asarray(frame)[ys:ys + yw, xs:xs + xw]
        s = float(np.sum(cropped))
        if best_sum is None or s > best_sum:
            best_sum = s
            best = [xw, yw, xs, ys]
    return best


def _parse_roi(text: str | None) -> List[int] | None:
    if not text:
        return None
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if len(parts) != 4:
        raise ValueError("roi must be 4 ints: xw yw xs ys")
    return [int(float(p)) for p in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot photon distribution from .npy frames.")
    parser.add_argument("folder", help="Folder containing .npy frames")
    parser.add_argument("--metric", default="pixel", choices=["pixel", "S"], help="Distribution type")
    parser.add_argument("--roi", default=None, help="ROI as 'xw yw xs ys' (for metric=S)")
    parser.add_argument("--bg-roi", default=None, help="BG ROI as 'xw yw xs ys' (optional)")
    parser.add_argument("--exposure-s", default=None, help="Exposure in seconds (for metric=S)")
    parser.add_argument("--out-dir", default=None, help="Output folder for plot (default: <input>\\plots)")
    parser.add_argument("--out-name", default=None, help="Output filename (default: <folder>_photon_dist.png)")
    parser.add_argument("--bins", default=None, help="Override histogram bins (int)")
    args = parser.parse_args()

    input_dir = resolve_input_dir(args.folder)
    npy_paths = list_npy_files(input_dir)
    if not npy_paths:
        raise RuntimeError(f"No .npy files found in {input_dir}")

    frames = load_frames(npy_paths)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        session_root = input_dir.parent if input_dir.name.lower() == "raw-data" else input_dir
        out_dir = session_root / "plots"
    if args.metric == "pixel":
        out_name = args.out_name or f"{input_dir.name}_photon_dist.png"
        plot_photon_distribution(
            light_images=frames,
            dark_images=None,
            save_dir=str(out_dir),
            save_name=out_name,
        )
        print(f"Saved plot -> {out_dir / out_name}")
        return 0

    # metric == "S": ROI-integrated, exposure-normalized
    roi = _parse_roi(args.roi)
    bg_roi = _parse_roi(args.bg_roi)
    if roi is None:
        roi = _pick_best_roi(frames[-1])
    if roi is None:
        raise RuntimeError("ROI not specified and auto-detection failed.")
    exposure_s = float(args.exposure_s) if args.exposure_s is not None else None
    if exposure_s is None or exposure_s <= 0:
        # Try default from device_registry.json if present (80 ms -> 0.08 s)
        exposure_s = 0.08

    samples: List[float] = []
    for f in frames:
        info = normalize_count(np.asarray(f), tuple(roi), bg_roi=tuple(bg_roi) if bg_roi else None, exposure_s=exposure_s)
        samples.append(float(info["S_norm"]))

    import matplotlib.pyplot as plt

    out_name = args.out_name or f"{input_dir.name}_S_dist.png"
    plt.figure(figsize=(10, 5))
    if args.bins is not None:
        bins = int(float(args.bins))
    else:
        bins = max(50, int(np.sqrt(len(samples))) * 8)
    s_min = float(np.min(samples))
    s_max = float(np.max(samples))
    pad = max(1e-9, 0.05 * (s_max - s_min) if s_max > s_min else 1.0)
    plt.hist(samples, bins=bins, range=(s_min - pad, s_max + pad), density=False, alpha=0.7, color="tab:blue")
    plt.xlim(s_min - pad, s_max + pad)
    plt.xlabel("S_norm")
    plt.ylabel("Count")
    plt.title("S_norm distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / out_name, dpi=150)
    print(f"Saved plot -> {out_dir / out_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
