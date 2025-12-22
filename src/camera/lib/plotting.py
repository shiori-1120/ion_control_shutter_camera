import os
import numpy as np
import matplotlib.pyplot as plt


def plot_profile(data, xs=None, fitted_curve=None, peaks=None,
                 centers_fwhm=None, title='', axis_name='Pixel',
                 save_dir: str | None = None, save_name: str | None = None):
    plt.figure(figsize=(10, 4))
    if xs is None:
        xs = np.arange(len(data))
    plt.plot(xs, data, '.-', label='Data', alpha=0.7)
    if peaks is not None and len(peaks) > 0:
        plt.plot(xs[peaks], np.asarray(data)[peaks], 'x', ms=10, mew=2, label='Detected Peaks')
    if fitted_curve is not None:
        x_fit, y_fit = fitted_curve
        plt.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Curve')
    if centers_fwhm is not None:
        for i, (center, fwhm) in enumerate(centers_fwhm):
            label_c = f'Center {i+1}' if i == 0 else None
            label_f = f'FWHM {i+1}' if i == 0 else None
            half_fwhm = fwhm / 2.0
            plt.axvline(center, color='g', linestyle='--', label=label_c)
            plt.axvspan(center - half_fwhm, center + half_fwhm, color='g', alpha=0.2, label=label_f)
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel('Intensity (Sum)')
    plt.legend()
    plt.grid(True)
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=150)


def plot_filter_comparison(data_raw, data_filtered, xs=None,
                           title: str = '', axis_name: str = 'Pixel',
                           save_dir: str | None = None, save_name: str | None = None):
    plt.figure(figsize=(10, 4))
    if xs is None:
        xs = np.arange(len(data_filtered))
    plt.plot(xs, data_raw, ':', color='gray', lw=1.5, label='Raw')
    plt.plot(xs, data_filtered, '-', color='tab:blue', lw=2, label='Filtered (moving average)')
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel('Intensity (Sum)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=150)


def show_npy_2d(img: np.ndarray, origin: str = 'lower', figsize=(6, 6),
                title: str | None = None, save_dir: str | None = None, save_name: str | None = None):
    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    vmin = p1
    vmax = p99
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(img, cmap='gray', origin=origin, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    X = int(img.shape[1])
    Y = int(img.shape[0])
    ax.set_xlabel('X (pixel index)')
    ax.set_ylabel('Y (pixel index)')
    step_x = max(1, X // 8)
    step_y = max(1, Y // 8)
    xticks = list(range(0, X, step_x))
    yticks = list(range(0, Y, step_y))
    if (X - 1) not in xticks:
        xticks.append(X - 1)
    if (Y - 1) not in yticks:
        yticks.append(Y - 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    plt.tight_layout()
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, save_name), dpi=150)
    return fig, ax


def plot_photon_distribution(light_images: list | None = None,
                             dark_images: list | None = None,
                             save_dir: str | None = None,
                             save_name: str | None = None):
    light_images = light_images or []
    dark_images = dark_images or []
    def _aggregate_counts(images):
        counts = []
        for img in images:
            arr = np.asarray(img, dtype=float)
            if arr.ndim < 2:
                raise ValueError("Each image must be at least 2D for photon count integration.")
            counts.append(arr.sum(axis=0))
        if counts:
            return np.concatenate(counts)
        return np.array([], dtype=float)
    light_counts = _aggregate_counts(light_images)
    dark_counts = _aggregate_counts(dark_images)
    combined = np.concatenate([c for c in (light_counts, dark_counts) if c.size > 0])
    if combined.size == 0:
        raise ValueError("Provided images did not yield valid photon counts.")
    start = int(np.floor(float(np.nanmin(combined))))
    end = int(np.ceil(float(np.nanmax(combined))))
    bin_edges = np.arange(start - 0.5, end + 1.5, 1)
    plt.figure(figsize=(10, 5))
    if light_counts.size > 0:
        mean_light = float(np.mean(light_counts))
        plt.hist(light_counts, bins=bin_edges, density=True, alpha=0.6, color='tab:orange', edgecolor='black', label=f'Light (mean={mean_light:.2f})')
        plt.axvline(mean_light, color='tab:orange', linestyle='--')
    if dark_counts.size > 0:
        mean_dark = float(np.mean(dark_counts))
        plt.hist(dark_counts, bins=bin_edges, density=True, alpha=0.6, color='navy', edgecolor='black', label=f'Dark (mean={mean_dark:.2f})')
        plt.axvline(mean_dark, color='navy', linestyle='--')
    plt.xlabel('Photon Count (integer bins)')
    plt.ylabel('Probability density')
    plt.title('Photon Distribution (integrated over y-axis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=150)


def plot_filter_effects(img: np.ndarray,
                        save_dir: str | None = None,
                        prefix: str = "filter_effect",
                        window_y: int = 21,
                        window_x: int = 21):
    """
    画像から垂直/水平プロファイルの生データと移動平均フィルタ後を可視化して保存する。
    save_dir が与えられた場合、`${prefix}_vertical.png` と `${prefix}_horizontal.png` を保存。
    """
    def _moving_average_1d(profile: np.ndarray, window: int) -> np.ndarray:
        prof = np.asarray(profile, dtype=float)
        w = int(window)
        if w <= 1:
            return prof.astype(float, copy=False)
        if w % 2 == 0:
            w += 1
        pad_left = w // 2
        pad_right = w - 1 - pad_left
        prof_pad = np.pad(prof, (pad_left, pad_right), mode='edge')
        kernel = np.ones(w, dtype=float) / float(w)
        return np.convolve(prof_pad, kernel, mode='valid')

    # 垂直
    y_raw = np.asarray(img).sum(axis=1)
    y_f = _moving_average_1d(y_raw, window_y)
    y_x = np.arange(len(y_raw))
    plot_filter_comparison(
        data_raw=y_raw,
        data_filtered=y_f,
        xs=y_x,
        title='Vertical Profile: Raw vs Filtered',
        axis_name='Y Pixel',
        save_dir=save_dir,
        save_name=(f"{prefix}_vertical.png" if save_dir else None)
    )

    # 水平
    x_raw = np.asarray(img).sum(axis=0)
    x_f = _moving_average_1d(x_raw, window_x)
    x_x = np.arange(len(x_raw))
    plot_filter_comparison(
        data_raw=x_raw,
        data_filtered=x_f,
        xs=x_x,
        title='Horizontal Profile: Raw vs Filtered',
        axis_name='X Pixel',
        save_dir=save_dir,
        save_name=(f"{prefix}_horizontal.png" if save_dir else None)
    )
