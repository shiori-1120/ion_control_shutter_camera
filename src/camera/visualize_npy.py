#!/usr/bin/env python3
"""
visualize_npy.py

簡単な CLI: .npy で保存された画像配列を読み込み、表示・保存・ヒストグラム出力を行う。

Usage examples:
  python visualize_npy.py path/to/img-1.npy
  python visualize_npy.py --cmap viridis --percentiles 1 99 img-1.npy
  python visualize_npy.py --save out.png img-1.npy

依存: numpy, matplotlib
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def load_array(path: str):
    arr = np.load(path)
    # If data has a channel-first shape like (1, H, W) or (3, H, W), convert to HxW or HxWxC
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def show_image(arr, cmap='gray', vmin=None, vmax=None, origin='lower'):
    plt.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    plt.colorbar()


def plot_histogram(arr, ax=None, bins=256):
    if ax is None:
        fig, ax = plt.subplots()
    flat = arr.ravel()
    ax.hist(flat, bins=bins, color='gray')
    ax.set_xlabel('pixel value')
    ax.set_ylabel('count')
    return ax


def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize .npy image files (simple)')
    p.add_argument('file', help='.npy file or directory containing .npy files')
    p.add_argument('--cmap', default='gray',
                   help='Matplotlib colormap (default: gray)')
    p.add_argument(
        '--origin', choices=['lower', 'upper'], default='lower', help='imshow origin')
    return p.parse_args()


def main():
    args = parse_args()

    target = args.file
    files = []
    if os.path.isdir(target):
        # list .npy files
        for fn in sorted(os.listdir(target)):
            if fn.lower().endswith('.npy'):
                files.append(os.path.join(target, fn))
        if not files:
            print('No .npy files found in', target)
            sys.exit(1)
    else:
        files = [target]

    for path in files:
        if not os.path.exists(path):
            print('Not found:', path)
            continue
        arr = load_array(path)
        # Only display the image
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            ax.imshow(arr, origin=args.origin)
        else:
            # compute default vmin/vmax from 1..99 percentiles for visibility
            vmin = np.percentile(arr, 1)
            vmax = np.percentile(arr, 99)
            show_image(arr, cmap=args.cmap, vmin=vmin,
                       vmax=vmax, origin=args.origin)
        ax.set_title(os.path.basename(path))

        plt.tight_layout()
        # Only show the figure interactively
        plt.show()


if __name__ == '__main__':
    main()
