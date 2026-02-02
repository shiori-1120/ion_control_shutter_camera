#!/usr/bin/env python3
# Run: python src\camera\visualize_npy.py "C:\\path\\to\\output\\YYYYMMDD_HHMMSS\\raw-data"
r"""
visualize_npy.py

目的: raw-data フォルダにある .npy フレームを一括で PNG 画像化し、同セッションの plots フォルダへ保存する。
スタイル:
  - Matplotlibを使用した2Dヒートマップ表示
  - 1/99パーセンタイルでのコントラスト調整
  - 軸単位: um (PIX_TO_UM = 0.2)
  - カラーバー: 画像と同じ縦幅で表示
  - スケールバー: 画像右下に表示

使い方（PowerShell）:
    python -m src.camera.visualize_npy "C:\path\to\raw-data"
"""

import os
import sys
import math
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# --- 設定値 ---
PIX_TO_UM = 0.2  # 1ピクセルあたりのマイクロメートル
# -------------

def load_array(path: str) -> np.ndarray:
    """npyファイルを読み込み、(H, W) の2次元配列として返す"""
    arr = np.load(path)
    
    # 配列の形状チェックと整形
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        arr = np.mean(arr, axis=2)
        
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

def plot_and_save(img: np.ndarray, title: str, save_path: str):
    """
    Matplotlibを使って画像を描画・保存する。
    - 軸は um 単位
    - カラーバーの高さを画像に合わせる
    - スケールバーを表示する
    """
    h, w = img.shape
    
    # 物理的なサイズ (um)
    width_um = w * PIX_TO_UM
    height_um = h * PIX_TO_UM
    
    # コントラスト調整 (1% - 99%)
    vmin = np.percentile(img, 1.0)
    vmax = np.percentile(img, 99.0)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # extent=[left, right, bottom, top] (画像座標系:上が0)
    im = ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax,
                   extent=[0, width_um, height_um, 0])
    
    ax.set_title(title)
    ax.set_xlabel('x [um]')
    ax.set_ylabel('y [um]')
    
    # --- 1. カラーバーを画像と同じ高さにする ---
    # 既存のaxの右側に、新しいax(cax)を作成して割り当てる
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(im, cax=cax, label='Intensity')
    
    # --- 2. スケールバーを追加する ---
    # 画像の幅の約1/5程度の長さを計算し、キリの良い数字にする
    target_len_um = width_um / 5.0
    exponent = math.floor(math.log10(target_len_um))
    fraction = target_len_um / (10 ** exponent)
    
    if fraction >= 5:
        bar_len_um = 5 * (10 ** exponent)
    elif fraction >= 2:
        bar_len_um = 2 * (10 ** exponent)
    else:
        bar_len_um = 1 * (10 ** exponent)
        
    scalebar = AnchoredSizeBar(ax.transData,
                               bar_len_um, 
                               f'{int(bar_len_um)} um', 
                               'lower right', 
                               pad=0.5,
                               color='white',
                               frameon=False,
                               size_vertical=height_um * 0.01, # バーの太さ(画像の高さの1%)
                               fontproperties=fm.FontProperties(size=10))
    
    ax.add_artist(scalebar)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def save_all_npy_as_png(raw_dir: str) -> int:
    raw_dir = _resolve_raw_dir(raw_dir)

    # plots フォルダの決定
    session_root = os.path.dirname(os.path.abspath(raw_dir))
    plots_dir = os.path.join(session_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    files = list_npy_files(raw_dir)
    if not files:
        print(f"No .npy files found in {raw_dir}")
        return 0

    print(f"Processing {len(files)} files...")
    print(f"Output directory: {plots_dir}")
    print(f"Scale: {PIX_TO_UM} um/pixel")

    saved = 0
    for path in files:
        try:
            img = load_array(path)
            
            if img.ndim == 3: 
                img = img[0]

            stem = os.path.splitext(os.path.basename(path))[0]
            save_name = f"{stem}.png"
            save_path = os.path.join(plots_dir, save_name)
            
            plot_and_save(img, title=stem, save_path=save_path)
            
            saved += 1
        except Exception as e:
            print(f"[skip] {path}: {e}")

    print(f"Saved {saved} PNGs.")
    return saved


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python src\\camera\\visualize_npy.py <path-to-raw-data>")
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