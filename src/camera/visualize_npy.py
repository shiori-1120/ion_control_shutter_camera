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
    p = argparse.ArgumentParser(description='Visualize .npy image files')
    p.add_argument('file', help='.npy file or directory containing .npy files')
    p.add_argument('--cmap', default='gray', help='Matplotlib colormap (default: gray)')
    p.add_argument('--vmin', type=float, default=None, help='vmin for imshow')
    p.add_argument('--vmax', type=float, default=None, help='vmax for imshow')
    p.add_argument('--percentiles', nargs=2, type=float, metavar=('PLOW', 'PHIGH'),
                   help='use percentiles to set vmin/vmax (e.g. --percentiles 1 99)')
    p.add_argument('--save', help='save displayed image to this path (png)')
    p.add_argument('--hist', action='store_true', help='also show histogram')
    p.add_argument('--check-nonzero', action='store_true', help='check and report non-zero pixel counts')
    p.add_argument('--threshold', type=float, default=0.0, help='threshold for counting pixels (default 0)')
    p.add_argument('--check-only', action='store_true', help='only run nonzero check and skip display/save')
    p.add_argument('--origin', choices=['lower', 'upper'], default='lower', help='imshow origin')
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
        # 非ゼロ/閾値チェック
        if args.check_nonzero:
            total = arr.size
            if args.threshold == 0.0:
                nonzero = np.count_nonzero(arr)
            else:
                nonzero = np.sum(arr > args.threshold)
            pct = 100.0 * nonzero / total if total > 0 else 0.0
            print(f"{os.path.basename(path)}: nonzero={nonzero}/{total} ({pct:.3f}%) threshold={args.threshold}")
            if args.check_only:
                continue
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # color image
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(arr, origin=args.origin)
            ax.set_title(os.path.basename(path))
            if args.hist:
                fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
                plot_histogram(arr, ax2)
        else:
            # grayscale
            # compute vmin/vmax from percentiles if requested
            vmin = args.vmin
            vmax = args.vmax
            if args.percentiles is not None:
                p0, p1 = args.percentiles
                vmin = np.percentile(arr, p0)
                vmax = np.percentile(arr, p1)
            elif vmin is None or vmax is None:
                # default: 1..99 percentiles
                vmin = np.percentile(arr, 1) if vmin is None else vmin
                vmax = np.percentile(arr, 99) if vmax is None else vmax

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            show_image(arr, cmap=args.cmap, vmin=vmin, vmax=vmax, origin=args.origin)
            ax.set_title(os.path.basename(path))
            if args.hist:
                fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3))
                plot_histogram(arr, ax2)

        plt.tight_layout()
        if args.save:
            out = args.save
            # if directory given, save with same basename but .png
            if os.path.isdir(out):
                out = os.path.join(out, os.path.basename(path).rsplit('.', 1)[0] + '.png')
            plt.savefig(out, dpi=200)
            print('Saved image to', out)
        plt.show()


if __name__ == '__main__':
    main()
