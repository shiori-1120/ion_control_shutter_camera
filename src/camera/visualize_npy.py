#!/usr/bin/env python3
# Run: python src\camera\visualize_npy.py "C:\\path\\to\\output\\YYYYMMDD_HHMMSS\\raw-data"
"""
visualize_npy.py

目的: raw-data フォルダにある .npy フレームを一括で PNG 画像化し、同セッションの plots フォルダへ保存する。
スタイルは initial_preparation.py の show_npy_2d に合わせ、軸ラベル・1/99パーセンタイルでの表示範囲を使用。

使い方（PowerShell）:
    python src\camera\visualize_npy.py C:\path\to\...\output\YYYYMMDD_HHMMSS\raw-data

依存: numpy, matplotlib
"""

# npyに複数の画像が入っている？？

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from typing import List
# initial_preparation のプロット関数を絶対importで使用
try:
    from src.camera.initial_preparation import show_npy_2d
except ImportError:
    # 直接src/cameraから実行した場合のフォールバック
    from initial_preparation import show_npy_2d


def load_array(path: str):
    arr = np.load(path)
    # If data has a channel-first shape like (1, H, W) or (3, H, W), convert to HxW or HxWxC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def list_npy_files(raw_dir: str) -> List[str]:
    return [os.path.join(raw_dir, fn) for fn in sorted(os.listdir(raw_dir)) if fn.lower().endswith('.npy')]


def _resolve_raw_dir(raw_dir: str) -> str:
    p = Path(raw_dir).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(f"raw-data directory not found: {raw_dir}")
    if p.name.lower() == "raw-data":
        return str(p)
    candidate = p / "raw-data"
    if candidate.is_dir():
        return str(candidate)
    if any(child.suffix.lower() == ".npy" for child in p.iterdir()):
        return str(p)
    raise FileNotFoundError(f"raw-data directory not found: {raw_dir}")


def save_all_npy_as_png(raw_dir: str) -> int:
    raw_dir = _resolve_raw_dir(raw_dir)

    # plots フォルダの決定（raw-data の親がセッションルート）
    session_root = os.path.dirname(os.path.abspath(raw_dir))
    plots_dir = os.path.join(session_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    files = list_npy_files(raw_dir)
    if not files:
        print(f"No .npy files found in {raw_dir}")
        return 0

    saved = 0
    for path in files:
        try:
            img = load_array(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            save_name = f"{stem}.png"
            title = stem

            # initial_preparation の関数で保存（軸ラベル・1/99パーセンタイル）
            fig, ax = show_npy_2d(
                img, title=title, save_dir=plots_dir, save_name=save_name)
            plt.close(fig)
            saved += 1
        except Exception as e:
            print(f"[skip] {path}: {e}")

    print(f"Saved {saved} PNGs -> {plots_dir}")
    return saved


def main() -> int:
    # 引数は raw-data ディレクトリ（.npy が並んでいるフォルダ）
    if len(sys.argv) != 2:
        print("Usage: python src\\camera\\visualize_npy.py <path-to-session-or-raw-data>")
        print("Example: python src\\camera\\visualize_npy.py data\\output\\spectrum\\YYYYMMDD_HHMMSS")
        return 2
    raw_dir = sys.argv[1]
    try:
        save_all_npy_as_png(raw_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
