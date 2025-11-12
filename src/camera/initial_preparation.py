# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

import numpy as np
import time
import os
import datetime
import re
from typing import Optional, Dict
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
try:
    from lib.ControlDevice import Control_CONTEC, Control_qCMOScamera
except Exception:
    # Camera control module may not be available in this environment.
    Control_CONTEC = None
    Control_qCMOScamera = None


EXPOSE_TIME = 0.050
WAIT_MARGIN_SEC = 0.02
IDLE_TIMEOUT_SEC = 10.0
ROUGH_ROI = [600, 100, 2976, 984]

# ROI フィッティング前の平滑化（ベースライン除去は行わず、移動平均のみ）
# 窓幅は奇数推奨。データによって 5〜51 程度で調整してください。
MOVING_AVG_WINDOW_Y = 21  # 垂直プロファイル用（行方向）
MOVING_AVG_WINDOW_X = 21  # 水平プロファイル用（列方向）

# 画面端のピークをノイズとして無視するためのマージン設定
# 配列長の一定割合か、ピクセル固定値の大きい方を採用
EDGE_IGNORE_RATIO = 0.02   # 全長の2%
EDGE_IGNORE_MIN_PIX = 10   # 最低でも10px

# 実験のログ出力用のtxt
# 露光時間、ROI情報、撮影枚数などを記録する
# npy以外のプロットに必要なデータを格納する（フィッティングなど）
# プロットは時間がかかるから後からでもできるようにする
# テキストよりも最適なフォーマットがある？


def log_experiment_details(log_path: str, expose_time: float, rois: list, n_frames: int):
    pass


def build_session_dirs(timestamp: str, base_parent: Optional[str] = None) -> Dict[str, str]:
    if base_parent is None:
        base_parent = os.path.join(os.path.dirname(__file__), "output")

    root = os.path.join(base_parent, timestamp)
    raw = os.path.join(root, "raw-data")
    plots = os.path.join(root, "plots")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return {"root": root, "raw": raw, "plots": plots}


def list_session_frame_paths(raw_dir: str, timestamp: str) -> list:
    """raw-data フォルダ内の .npy をすべて読み込むためのパス一覧を返す。

    仕様変更: ファイル名のパターンやタイムスタンプには依存せず、
    ディレクトリ内の拡張子 .npy のファイルをすべて対象にします（名前順）。
    引数 timestamp は互換性のため残しますが、フィルタには使いません。
    """
    names = [n for n in sorted(os.listdir(raw_dir))
             if n.lower().endswith('.npy')]
    if not names:
        raise RuntimeError(f"No .npy files present in '{raw_dir}'.")
    return [os.path.join(raw_dir, n) for n in names]


def load_session_frames(raw_dir: str, timestamp: str) -> list:
    """list_session_frame_paths で抽出した .npy を読み込み、壊れたファイルはスキップして ndarray のリストを返す。"""
    paths = list_session_frame_paths(raw_dir, timestamp)
    frames: list[np.ndarray] = []
    bad: list[str] = []
    for p in paths:
        try:
            # allow_pickle=False で安全側。壊れた/空ファイルは例外や size==0 で弾く
            arr = np.load(p, allow_pickle=False)
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                print(f"[skip] empty or invalid array: {os.path.basename(p)}")
                bad.append(p)
                continue
            frames.append(arr)
        except Exception as e:
            print(f"[skip] failed to load {os.path.basename(p)}: {e}")
            bad.append(p)

    if not frames:
        raise RuntimeError(
            "No usable .npy frames after loading. "
            f"Tried {len(paths)} files in '{raw_dir}', skipped {len(bad)} bad files."
        )
    if bad:
        print(
            f"[warn] skipped {len(bad)} corrupted/empty files under '{raw_dir}'.")
    return frames


def get_n_frames_from_buffer(
    expose_time: float = 0.100,
    roi: Optional[list] = None,
    session_root: Optional[str] = None,
    start_index: Optional[int] = None,
) -> tuple[int, int]:

    # 出力先
    if session_root is None:
        session_root = os.path.join(os.path.dirname(
            __file__), "output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_path = os.path.join(session_root, "raw-data")
    os.makedirs(output_path, exist_ok=True)
    wait_timeout_sec = max(float(expose_time) + WAIT_MARGIN_SEC, 0.05)

    # Windows の非ブロッキングキー入力（Enter で終了）
    try:
        import msvcrt  # type: ignore

        def _enter_pressed():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                return ch in (b"\r", b"\n")
            return False
    except Exception:
        def _enter_pressed():
            return False

    qCMOS = Control_qCMOScamera()
    qCMOS.OpenCamera_GetHandle()

    try:
        qCMOS.SetParameters(expose_time, roi[0], roi[1], roi[2], roi[3])
        qCMOS.StartCapture()

        # インデックス決定（スキャンは行わない）
        idx = int(start_index) if (
            start_index is not None and int(start_index) >= 1) else 1
        saved = 0
        last_saved = time.time()

        while True:
            # Enter で即終了
            if _enter_pressed():
                print("[capture] Enter pressed. stopping...")
                break

            # idle timeout を超えないよう、待機時間を動的に調整
            remaining_idle = max(0.001, float(
                IDLE_TIMEOUT_SEC) - (time.time() - last_saved))
            dynamic_timeout = max(0.001, min(wait_timeout_sec, remaining_idle))

            # フレーム準備完了を待つ
            ok, _ = qCMOS.wait_for_frame_ready(dynamic_timeout)
            if not ok:
                # idle timeout 判定
                if (time.time() - last_saved) >= float(IDLE_TIMEOUT_SEC):
                    print(f"[capture] idle {IDLE_TIMEOUT_SEC}s. stopping...")
                    break
                continue

            data = qCMOS.GetLastFrame()
            img = data[1]
            if img.size == 0 or not np.any(img):
                continue

            filename = f"{idx:04d}.npy"
            np.save(os.path.join(output_path, filename), img)
            idx += 1
            saved += 1
            last_saved = time.time()

        print(f"[capture] saved {saved} frames to {output_path}")
        return saved, idx
    except KeyboardInterrupt:
        if 'saved' in locals() and 'idx' in locals():
            return saved, idx
        return 0, (start_index if start_index is not None else 1)
    finally:
        qCMOS.StopCapture()
        qCMOS.ReleaseBuf()
        qCMOS.CloseUninitCamera()


# 新しい閾値評価関数を追加


# TODO: 引数をndarrayに変更
def apply_roi_npy(npy_path: str, roi: list):
    img = np.load(npy_path)
    x_width, y_width, x_start, y_start = map(int, roi)
    img_cropped = img[y_start:y_start+y_width, x_start:x_start+x_width]
    return img_cropped


# 汎用的な1Dプロット関数。
def plot_profile(data, xs=None, fitted_curve=None, peaks=None,
                 centers_fwhm=None, title='', axis_name='Pixel',
                 save_dir: Optional[str] = None, save_name: Optional[str] = None):
    plt.figure(figsize=(10, 4))

    if xs is None:
        xs = np.arange(len(data))

    plt.plot(xs, data, '.-', label='Data', alpha=0.7)

    # ピーク位置を 'x' でプロット
    if peaks is not None and len(peaks) > 0:
        plt.plot(xs[peaks], data[peaks], 'x', ms=10,
                 mew=2, label='Detected Peaks')

    # フィット曲線をプロット
    if fitted_curve is not None:
        x_fit, y_fit = fitted_curve
        plt.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Curve')

    # 中心とFWHM (半値全幅) をプロット
    if centers_fwhm is not None:
        for i, (center, fwhm) in enumerate(centers_fwhm):
            label_c = f'Center {i+1}' if i == 0 else None
            label_f = f'FWHM {i+1}' if i == 0 else None
            half_fwhm = fwhm / 2.0
            plt.axvline(center, color='g', linestyle='--', label=label_c)
            plt.axvspan(center - half_fwhm, center + half_fwhm,
                        color='g', alpha=0.2, label=label_f)

    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel('Intensity (Sum)')
    plt.legend()
    plt.grid(True)
    # save if requested
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=150)
    # plt.show()


def plot_filter_comparison(data_raw, data_filtered, xs=None,
                           title: str = '', axis_name: str = 'Pixel',
                           save_dir: Optional[str] = None, save_name: Optional[str] = None):
    """フィルタ前後の1Dプロファイルを重ねて可視化する簡易プロット。"""
    plt.figure(figsize=(10, 4))
    if xs is None:
        xs = np.arange(len(data_filtered))
    plt.plot(xs, data_raw, ':', color='gray', lw=1.5, label='Raw')
    plt.plot(xs, data_filtered, '-', color='tab:blue',
             lw=2, label='Filtered (moving average)')
    plt.title(title)
    plt.xlabel(axis_name)
    plt.ylabel('Intensity (Sum)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_dir is not None and save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, save_name), dpi=150)
    # plt.show()


# 1D の多峰ローレンツ和（最後の引数はオフセット）
def FUNC(x, *params):
    num_func = int((len(params) - 1) / 3)
    y_sum = np.zeros_like(x, dtype=np.float64)
    for i in range(num_func):
        amp = params[3*i]
        ctr = params[3*i+1]
        wid = params[3*i+2]
        y_sum += amp * (wid**2) / ((x - ctr)**2 + wid**2)
    y_sum += params[-1]
    return y_sum


def lorentz(x, A, x0, wid, offset):
    return A * (wid**2) / ((x - x0)**2 + wid**2) + offset


def _moving_average_1d(profile: np.ndarray, window: int) -> np.ndarray:
    """1D 移動平均フィルタ。エッジは端値でパディングして長さを保ちます。
    window<=1 の場合は元データを float にして返します。
    """
    prof = np.asarray(profile, dtype=float)
    w = int(window)
    if w <= 1:
        return prof.astype(float, copy=False)
    # 奇数にそろえる（偶数の場合は+1）
    if w % 2 == 0:
        w += 1
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    prof_pad = np.pad(prof, (pad_left, pad_right), mode='edge')
    kernel = np.ones(w, dtype=float) / float(w)
    smoothed = np.convolve(prof_pad, kernel, mode='valid')
    return smoothed


def _edge_margin(n: int) -> int:
    """配列長 n に対する端マージン（ピーク無視）を返す。"""
    return max(int(round(n * float(EDGE_IGNORE_RATIO))), int(EDGE_IGNORE_MIN_PIX))


# 2D画像から垂直プロファイルを抽出し、ローレンツフィッティングを実行する。
def fit_vertical_profile(img):
    # 元のプロファイル
    y_profile_raw = img.sum(axis=1)
    # 移動平均フィルタ（平坦化はしない）
    y_profile = _moving_average_1d(y_profile_raw, MOVING_AVG_WINDOW_Y)
    y_x = np.arange(len(y_profile))

    # ピーク検出: フィルタ済みプロファイルをそのまま使用（端のピークは除外）
    y_peaks, _ = find_peaks(y_profile, distance=5)
    if y_peaks.size > 0:
        m = _edge_margin(len(y_profile))
        y_peaks = y_peaks[(y_peaks >= m) & (y_peaks <= len(y_profile) - 1 - m)]

    # 初期値
    y_offset0 = float(np.median(y_profile))
    y_A0 = float(y_profile.max() - y_offset0)
    # 端の極大を除外して初期中心を選ぶ
    m = _edge_margin(len(y_profile))
    if len(y_profile) > 2*m:
        inner = y_profile[m:len(y_profile)-m]
        y_peak_idx = int(m + np.argmax(inner))
    else:
        y_peak_idx = int(np.argmax(y_profile))
    p0y = [max(0.0, y_A0), float(y_peak_idx), 5.0, y_offset0]
    bounds_y = ([0.0, max(0, y_peak_idx-10), 0.5, 0.0],
                [np.inf, min(len(y_profile)-1, y_peak_idx+10), 200.0, np.inf])

    try:
        popt_y, _ = curve_fit(lorentz, y_x, y_profile,
                              p0=p0y, bounds=bounds_y, maxfev=10000)
        A_fit, y_ctr, y_wid, y_off = popt_y
        y_fitted = lorentz(y_x, *popt_y)
        y_fwhm = 2.0 * abs(float(y_wid))

        return {
            'profile': y_profile,
            'profile_raw': y_profile_raw,
            'x': y_x,
            'peaks': y_peaks,
            'fitted': y_fitted,
            'params': popt_y,
            'center': y_ctr,
            'fwhm': y_fwhm
        }
    except Exception as e:
        print(f"Vertical lorentz fit failed: {e}")
        return None


#  2D画像から水平プロファイルを抽出し、多峰ローレンツフィッティングを実行する。
def fit_horizontal_profile(img):
    # 元のプロファイル
    x_profile_raw = img.sum(axis=0)
    # 移動平均フィルタ（平坦化はしない）
    x_profile = _moving_average_1d(x_profile_raw, MOVING_AVG_WINDOW_X)
    x_x = np.arange(len(x_profile))

    # ピーク検出
    hth = (x_profile.max() + x_profile.min()) / 2.0
    x_peaks, props = find_peaks(x_profile, height=hth, distance=20)
    # 端のピークを除外
    if x_peaks.size > 0:
        m = _edge_margin(len(x_profile))
        mask = (x_peaks >= m) & (x_peaks <= len(x_profile) - 1 - m)
        x_peaks = x_peaks[mask]

    if len(x_peaks) == 0:
        print("Horizontal profile: No peaks found.")
        return None

    # 初期値と境界の構築
    guess = []
    lower = []
    upper = []
    median_x = float(np.median(x_profile))
    for p in x_peaks:
        amp0 = float(max(0.0, x_profile[p] - median_x))
        guess.extend([amp0, float(p), 5.0])
        lower.extend([0.0, max(0, p-10), 0.5])
        upper.extend([np.inf, min(len(x_profile)-1, p+10), 200.0])
    guess.append(median_x)  # offset
    lower.append(0.0)
    upper.append(np.inf)

    try:
        popt_h, _ = curve_fit(FUNC, x_x, x_profile, p0=guess,
                              bounds=(np.array(lower), np.array(upper)),
                              maxfev=20000)
        x_fitted = FUNC(x_x, *popt_h)

        # 複数ピークの中心とFWHMを抽出
        num_funcs_h = int((len(popt_h) - 1) / 3)
        centers = []
        fwhms = []
        for i in range(num_funcs_h):
            ctr = float(popt_h[3*i+1])
            wid = float(popt_h[3*i+2])
            fwhm = 2.0 * abs(wid)
            centers.append(ctr)
            fwhms.append(fwhm)

        return {
            'profile': x_profile,
            'profile_raw': x_profile_raw,
            'x': x_x,
            'peaks': x_peaks,
            'fitted': x_fitted,
            'params': popt_h,
            'centers': centers,
            'fwhms': fwhms
        }
    except Exception as e:
        print(f"Horizontal multi-lorentz fit failed: {e}")
        return None


def lorentz_fit_profiles(img, plot=False) -> dict:
    results = {'vertical': None, 'horizontal': None}

    # 垂直プロファイルのローレンツフィット
    vertical_lorentz_fit = fit_vertical_profile(img)
    results['vertical'] = vertical_lorentz_fit
    if plot and vertical_lorentz_fit is not None:
        # フィルタ効果の可視化（縦）
        try:
            plot_filter_comparison(
                data_raw=vertical_lorentz_fit.get(
                    'profile_raw', vertical_lorentz_fit['profile']),
                data_filtered=vertical_lorentz_fit['profile'],
                xs=vertical_lorentz_fit['x'],
                title='Vertical Profile: Raw vs Filtered',
                axis_name='Y Pixel'
            )
        except Exception as _:
            pass
        plot_profile(
            data=vertical_lorentz_fit['profile'],
            xs=vertical_lorentz_fit['x'],
            fitted_curve=(
                vertical_lorentz_fit['x'], vertical_lorentz_fit['fitted']),
            peaks=vertical_lorentz_fit['peaks'],
            centers_fwhm=[(vertical_lorentz_fit['center'],
                           vertical_lorentz_fit['fwhm'])],
            title='Vertical Profile with Lorentz Fit',
            axis_name='Y Pixel'
        )

    # 水平プロファイルのローレンツフィット
    start = time.time()
    horizontal_lorentz_fit = fit_horizontal_profile(img)
    results['horizontal'] = horizontal_lorentz_fit
    if plot and horizontal_lorentz_fit is not None:
        # フィルタ効果の可視化（横）
        try:
            plot_filter_comparison(
                data_raw=horizontal_lorentz_fit.get(
                    'profile_raw', horizontal_lorentz_fit['profile']),
                data_filtered=horizontal_lorentz_fit['profile'],
                xs=horizontal_lorentz_fit['x'],
                title='Horizontal Profile: Raw vs Filtered',
                axis_name='X Pixel'
            )
        except Exception as _:
            pass
        plot_profile(
            data=horizontal_lorentz_fit['profile'],
            xs=horizontal_lorentz_fit['x'],
            fitted_curve=(
                horizontal_lorentz_fit['x'], horizontal_lorentz_fit['fitted']),
            peaks=horizontal_lorentz_fit['peaks'],
            centers_fwhm=list(
                zip(horizontal_lorentz_fit['centers'], horizontal_lorentz_fit['fwhms'])),
            title='Horizontal Profile with Multi-Lorentz Fit',
            axis_name='X Pixel'
        )

    end = time.time()
    print("Horizontal fit time:", end - start)

    return results


def plot_filter_effects(img: np.ndarray,
                        save_dir: Optional[str] = None,
                        prefix: str = "filter_effect"):
    """画像から垂直/水平プロファイルの生データと移動平均フィルタ後を可視化して保存する。
    save_dir が与えられた場合、`${prefix}_vertical.png` と `${prefix}_horizontal.png` を保存する。
    """
    # 垂直
    y_raw = np.asarray(img).sum(axis=1)
    y_f = _moving_average_1d(y_raw, MOVING_AVG_WINDOW_Y)
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
    x_f = _moving_average_1d(x_raw, MOVING_AVG_WINDOW_X)
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


def generate_rois_from_analyze_results(results: dict, img_shape) -> list:
    """
    - ROI の線幅は (縦FWHM と 横FWHM平均) の平均値を採用（上下左右とも同じピクセル幅）
    - 各ROIは [x-width, y-width, x-start, y-start] 形式（DCAM subarrayの順序に合わせる）
    - 画像外にはみ出さないよう開始座標をクリップ
    """
    vert = results.get('vertical') or {}
    horiz = results.get('horizontal') or {}

    if not vert or vert.get('center') is None or vert.get('fwhm') is None:
        raise ValueError('Vertical fit result is missing center/fwhm')
    if not horiz or not horiz.get('centers') or not horiz.get('fwhms'):
        raise ValueError('Horizontal fit result is missing centers/fwhms')

    y_center = float(vert['center'])
    v_fwhm = float(vert['fwhm'])
    centers_x = [float(c) for c in horiz['centers']]
    fwhms_x = [float(w) for w in horiz['fwhms']]

    avg_linewidth = float((v_fwhm + np.mean(fwhms_x)) / 2.0)
    width_px = max(1, int(round(avg_linewidth)))
    x_width = width_px
    y_width = width_px

    X = int(img_shape[1])  # width (x)
    Y = int(img_shape[0])  # height (y)

    rois = []
    for x_center in centers_x:
        x_start = int(round(x_center - x_width / 2.0))
        y_start = int(round(y_center - y_width / 2.0))

        # 画像内に収める（最低限のクリップ）
        x_start = max(0, min(x_start, X - x_width))
        y_start = max(0, min(y_start, Y - y_width))

        rois.append([x_width, y_width, x_start, y_start])

    return rois


def generate_rois_from_image(img: np.ndarray, plot: bool = False) -> list:
    results = lorentz_fit_profiles(img, plot)
    rois = generate_rois_from_analyze_results(results, img.shape)
    return rois


def show_npy_2d(img: np.ndarray,
                origin: str = 'lower',
                figsize=(6, 6),
                title: str = None,
                save_dir: Optional[str] = None,
                save_name: Optional[str] = None):

    p1 = np.percentile(img, 1)
    p99 = np.percentile(img, 99)
    vmin = p1
    vmax = p99
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(img, cmap='gray', origin=origin, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax)
    if title:
        ax.set_title(title)
    # 軸とインデックス（ピクセル）を表示
    X = int(img.shape[1])  # width (x)
    Y = int(img.shape[0])  # height (y)
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
        save_path = os.path.join(save_dir, save_name)
        fig.savefig(save_path, dpi=150)
    # plt.show()
    return fig, ax


def plot_photon_distribution(light_images: list | None = None,
                             dark_images: list | None = None,
                             save_dir: Optional[str] = None,
                             save_name: Optional[str] = None):
    light_images = light_images or []
    dark_images = dark_images or []

    def _aggregate_counts(images):
        counts = []
        for img in images:
            arr = np.asarray(img, dtype=float)
            if arr.ndim < 2:
                raise ValueError(
                    "Each image must be at least 2D for photon count integration.")
            counts.append(arr.sum(axis=0))
        if counts:
            return np.concatenate(counts)
        return np.array([], dtype=float)

    light_counts = _aggregate_counts(light_images)
    dark_counts = _aggregate_counts(dark_images)

    combined = np.concatenate(
        [c for c in (light_counts, dark_counts) if c.size > 0])
    if combined.size == 0:
        raise ValueError("Provided images did not yield valid photon counts.")

    pc_min = float(np.nanmin(combined))
    pc_max = float(np.nanmax(combined))
    start = int(np.floor(pc_min))
    end = int(np.ceil(pc_max))
    bin_edges = np.arange(start - 0.5, end + 1.5, 1)

    plt.figure(figsize=(10, 5))

    if light_counts.size > 0:
        mean_light = float(np.mean(light_counts))
        plt.hist(light_counts, bins=bin_edges, density=True,
                 alpha=0.6, color='tab:orange', edgecolor='black',
                 label=f'Light (mean={mean_light:.2f})')
        plt.axvline(mean_light, color='tab:orange', linestyle='--')

    if dark_counts.size > 0:
        mean_dark = float(np.mean(dark_counts))
        plt.hist(dark_counts, bins=bin_edges, density=True,
                 alpha=0.6, color='navy', edgecolor='black',
                 label=f'Dark (mean={mean_dark:.2f})')
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
    plt.show()


def determine_ion_state(img: np.ndarray, threshold: float) -> bool:
    # y軸方向に積分して1次元の光子数分布をl取得
    photon_counts = img.sum(axis=0)

    # 閾値以下のデータ点の数を取得
    dark_count = np.sum(photon_counts <= threshold)

    # 全データ点に対する、閾値以下のデータ点の割合を計算
    dark_ratio = dark_count / len(photon_counts)

    # 暗状態のデータが半分以上なら暗状態(False)、そうでなければ明状態(True)
    return dark_ratio <= 0.5


def estimate_threshold_otsu_from_frames(frames: list[np.ndarray], nbins: int = 256) -> float:
    """各フレームの総光子数（画素和）の2クラス分離を仮定し、Otsu法で閾値を推定。
    戻り値はスカラー閾値（画素和）。
    """
    if not frames:
        raise ValueError("No frames provided for threshold estimation.")
    sums = np.array([np.asarray(f, dtype=float).sum()
                    for f in frames], dtype=float)
    if np.allclose(sums.min(), sums.max()):
        return float(sums.mean())
    hist, edges = np.histogram(sums, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total <= 1:
        return float(np.median(sums))
    weight1 = np.cumsum(hist)
    weight2 = total - weight1
    sum_total = np.sum(hist * centers)
    sum1 = np.cumsum(hist * centers)
    # avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mean1 = sum1 / np.maximum(weight1, 1e-12)
        mean2 = (sum_total - sum1) / np.maximum(weight2, 1e-12)
        var_between = weight1 * weight2 * (mean1 - mean2) ** 2
    var_between[weight1 == 0] = -1
    var_between[weight2 == 0] = -1
    idx = int(np.argmax(var_between))
    if var_between[idx] <= 0:
        return float(np.median(sums))
    return float(centers[idx])


def split_images_by_threshold(frames: list[np.ndarray], threshold: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """フレームの画素和で二分。threshold より大きい→Light、小さい→Dark を返す。"""
    light, dark = [], []
    for f in frames:
        val = float(np.asarray(f, dtype=float).sum())
        (light if val > threshold else dark).append(f)
    return light, dark


def extract_rois_from_image(img: np.ndarray, rois: list) -> list:

    img2 = np.asarray(img)

    Y, X = int(img2.shape[0]), int(img2.shape[1])

    crops = []
    for roi in rois:
        try:
            x_width, y_width, x_start, y_start = map(int, roi)
        except Exception:
            continue

        # マイナスや過大をクリップ
        x_width = max(1, min(x_width, X))
        y_width = max(1, min(y_width, Y))
        x_start = max(0, min(x_start, X - x_width))
        y_start = max(0, min(y_start, Y - y_width))

        y_end = y_start + y_width
        x_end = x_start + x_width

        if y_start >= y_end or x_start >= x_end:
            continue

        crop = img2[y_start:y_end, x_start:x_end]
        crops.append(crop)

    return crops


def verify_ion_count_consistency(img: np.ndarray, ion_positions) -> bool:
    try:
        if img is None:
            return False
        expected_count = len(ion_positions or [])
        hfit = fit_horizontal_profile(img)
        if hfit is None:
            return False
        detected_count = len(hfit.get('centers', []) or [])
        return detected_count == expected_count
    except Exception:
        return False


# integrate_photon_countsをイオンの個数だけ繰り返し、さらに周波数ごとに繰り返して、周波数と励起成功確率の2D配列を作成する関数
def create_frequency_excitation_probability_matrix(spectrum_data, ion_counts):
    pass  # 実装はここに記述してください


# 周波数と励起成功確率の2Dndarrayをプロットする関数。ローレンチアンフィットも行う。中心周波数と線幅も求める。
def plot_frequency_excitation_probability(frequencies_excite_probability):
    data = np.asarray(frequencies_excite_probability, dtype=float)

    freqs = data[:, 0]
    prob_matrix = data[:, 1:]

    num_sets = prob_matrix.shape[1]
    colors = plt.cm.viridis(np.linspace(0, 1, num_sets))

    plt.figure(figsize=(10, 6))
    fit_results = []

    for idx in range(num_sets):
        y = prob_matrix[:, idx]

        # 初期推定値を構築する
        offset0 = float(np.min(y))
        amp0 = float(np.max(y) - offset0)
        peak_idx = int(np.argmax(y))
        center0 = float(freqs[peak_idx])
        width0 = max((freqs.max() - freqs.min()) / 10.0, 1e-6)
        p0 = [amp0, center0, width0, offset0]

        # 境界を設定しフィットを実行
        bounds = ([0.0, freqs.min(), 1e-9, 0.0],
                  [1.0, freqs.max(), (freqs.max() - freqs.min()), 1.0])
        try:
            popt, _ = curve_fit(lorentz, freqs, y, p0=p0,
                                bounds=bounds, maxfev=20000)
            fitted = lorentz(freqs, *popt)
            center = float(popt[1])
            width = float(popt[2])
            fwhm = 2.0 * abs(width)
            fit_results.append(
                {'index': idx, 'params': popt, 'center': center, 'fwhm': fwhm})
        except Exception as exc:
            fitted = None
            fit_results.append({'index': idx, 'params': None,
                               'center': None, 'fwhm': None, 'error': str(exc)})

        label = f"Dataset {idx+1}"
        plt.plot(freqs, y, '.-', color=colors[idx], label=label)
        if fitted is not None:
            plt.plot(freqs, fitted, '-', color=colors[idx], alpha=0.6,
                     label=f"Fit {idx+1} (center={center:.3f}, FWHM={fwhm:.3f})")

    plt.xlabel('Frequency')
    plt.ylabel('Excitation Probability')
    plt.title('Excitation Probability vs Frequency with Lorentz Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fit_results


def main():
    # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 固定のセッションIDを使う場合は文字列で指定してください
    ts = "20251102_104353"
    s = build_session_dirs(ts)

    # 同一セッション内での保存ファイル連番をメモリで管理
    next_idx = 1

    # ガウシアンフィッティングして大きなROI決定してもいいかも

    # 明状態の取得
    # saved, next_idx = get_n_frames_from_buffer(
    #     expose_time=EXPOSE_TIME,
    #     roi=ROUGH_ROI,
    #     session_root=s["root"],
    #     start_index=next_idx,
    # )
    saved = np.load(
        "C:\\Users\\karishio\\Desktop\\single_ion_control\\src\\camera\\output\\20251102_104353\\raw-data\\2025_1102_105441data_000040.npy")
    # トリミング範囲決定（フィット可視化 + フィルタ効果の可視化も実施）
    rois = generate_rois_from_image(saved, plot=True)
    # フィルタ前後の比較図を保存
    try:
        plot_filter_effects(saved, save_dir=s["plots"], prefix=f"{ts}_filter")
    except Exception as _:
        pass
    print(f"[ROI] determined {len(rois)} ROIs: {rois}")
    # 暗状態の取得
    # saved_2, next_idx = get_n_frames_from_buffer(
    #     expose_time=EXPOSE_TIME,
    #     roi=ROUGH_ROI,
    #     session_root=s["root"],
    #     start_index=next_idx,
    # )

    # 今回セッションの全フレームを関数で読み込む
    frames = load_session_frames(s["raw"], ts)
    print(f"[load] loaded {len(frames)} frames from session '{ts}'")
    # 全体画像の可視化（軸つき）
    show_npy_2d(frames[0], title="Full frame",
                save_dir=s["plots"], save_name=f"{ts}_full_frame.png")
    print
    all_crops = []
    for f in frames:
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)
    print(f"[extract] extracted {len(all_crops)} ROIs from all frames")

    # 閾値を推定し、Light/Dark に分けてプロット
    print(f"[threshold] estimating threshold...")
    images_for_hist = all_crops if len(all_crops) > 0 else frames
    th = estimate_threshold_otsu_from_frames(images_for_hist)
    print(f"[threshold] estimated by Otsu: {th:.3f} (sum over ROI/frame)")
    light_imgs, dark_imgs = split_images_by_threshold(images_for_hist, th)
    print(
        f"[threshold] split -> light={len(light_imgs)}, dark={len(dark_imgs)}")

    plot_photon_distribution(light_images=light_imgs,
                             dark_images=dark_imgs,
                             save_dir=s["plots"],
                             save_name=f"{ts}_photon_dist_split.png")

# 光子数分布をプロットする関数（色々をまとめた関数）


if __name__ == "__main__":
    main()
