#!/usr/bin/env python3
# Run:
#   python src\camera\tif_to_npy.py "C:\\path\\to\\folder_with_tif"
#   python src\camera\tif_to_npy.py "C:\\path\\to\\folder_with_tif" -o "C:\\path\\to\\output"
"""
Convert all .tif/.tiff files in a folder to .npy files.

Behavior:
- Input: a folder containing .tif/.tiff files (single- or multi-page)
- Output folder:
    - Default: If the folder name is "tif": save into sibling "raw-data" (session style)
        Otherwise: save into a child folder "npy" under the input folder
    - You can override the output directory with `-o/--out-dir <path>`
- Naming:
  - Single-page: <basename>.npy
  - Multi-page:  <basename>_0001.npy, <basename>_0002.npy, ...
- Existing files are skipped (no overwrite)

Dependencies: numpy, imageio (for reading tiff)
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Optional

import numpy as np

try:
    import imageio.v3 as iio
except Exception:
    iio = None  # type: ignore


def _is_tiff_path(p: str) -> bool:
    ext = os.path.splitext(p)[1].lower()
    return ext in (".tif", ".tiff")


def _load_tiff_any(path: str) -> np.ndarray:
    if iio is None:
        raise RuntimeError("imageio is required: pip install imageio tifffile")
    arr = iio.imread(path, index=None)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    return arr


def _decide_out_dir(folder: str, out_override: Optional[str] = None) -> str:
    if out_override:
        return os.path.abspath(out_override)
    parent = os.path.dirname(os.path.abspath(folder))
    base = os.path.basename(os.path.normpath(folder))
    if base.lower() == "tif":
        return os.path.join(parent, "raw-data")
    return os.path.join(folder, "npy")


def convert_folder_tif_to_npy(folder: str, out_dir: Optional[str] = None) -> int:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    names = [n for n in sorted(os.listdir(folder)) if _is_tiff_path(n)]
    if not names:
        print(f"No .tif/.tiff files found in {folder}")
        return 0

    out_dir = _decide_out_dir(folder, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for name in names:
        tif_path = os.path.join(folder, name)
        base = os.path.splitext(name)[0]
        try:
            arr = _load_tiff_any(tif_path)
            # arr: (N,H,W) or (N,H,W,C)
            if arr.ndim not in (3, 4):
                print(f"[skip] {name}: unsupported ndim={arr.ndim}")
                continue
            num_pages = arr.shape[0]
            if num_pages == 1:
                out_path = os.path.join(out_dir, f"{base}.npy")
                if os.path.exists(out_path):
                    print(f"[skip] exists: {out_path}")
                else:
                    np.save(out_path, arr[0])
                    saved += 1
            else:
                for i in range(num_pages):
                    out_path = os.path.join(out_dir, f"{base}_{i+1:04d}.npy")
                    if os.path.exists(out_path):
                        print(f"[skip] exists: {out_path}")
                        continue
                    np.save(out_path, arr[i])
                    saved += 1
        except Exception as e:
            print(f"[skip] {name}: {e}")

    print(f"Saved {saved} NPY files -> {out_dir}")
    return saved


def main(argv: List[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        description="Convert .tif/.tiff files in a folder to .npy files."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing .tif/.tiff files",
    )
    parser.add_argument(
        "-o", "--out-dir",
        dest="out_dir",
        default=None,
        help="Output directory to save .npy files (overrides default policy)",
    )

    args = parser.parse_args(argv)

    try:
        convert_folder_tif_to_npy(args.folder, args.out_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
