# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

import numpy as np
import time
import os
import datetime
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
    """raw-data から今回セッションのフレーム .npy のパス一覧を返す。

    対応するファイル名パターン:
    - 旧式: "{timestamp}_0001.npy" のようにタイムスタンプ接頭辞 + 4桁連番
    - 新式: "0001.npy" のように 4桁連番のみ（セッションフォルダで識別）
    """
    paths: list[str] = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith('.npy'):
            continue

        stem = os.path.splitext(name)[0]
        ok = False
        # 旧式: ts_0001.npy 形式
        if timestamp and name.startswith(f"{timestamp}_"):
            seq = stem.split("_")[-1]
            if len(seq) == 4 and seq.isdigit():
                ok = True
        else:
            # 新式: 0001.npy 形式（セッションフォルダで識別）
            if len(stem) == 4 and stem.isdigit():
                ok = True

        if ok:
            paths.append(os.path.join(raw_dir, name))

    if not paths:
        raise RuntimeError("No raw frames found for this session.")
    return paths


def load_session_frames(raw_dir: str, timestamp: str) -> list:
    """list_session_frame_paths で抽出した .npy を読み込んで ndarray のリストを返す。"""
    paths = list_session_frame_paths(raw_dir, timestamp)
    return [np.load(p) for p in paths]


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


# 2D画像から垂直プロファイルを抽出し、ローレンツフィッティングを実行する。
def fit_vertical_profile(img):
    y_profile = img.sum(axis=1)
    y_x = np.arange(len(y_profile))

    # ピーク検出 (平滑化してから)
    y_smooth = gaussian_filter1d(y_profile, sigma=1.5)
    y_peaks, _ = find_peaks(y_smooth, distance=5)

    # 初期値
    y_offset0 = float(np.median(y_profile))
    y_A0 = float(y_profile.max() - y_offset0)
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
    x_profile = img.sum(axis=0)
    x_x = np.arange(len(x_profile))

    # ピーク検出
    hth = (x_profile.max() + x_profile.min()) / 2.0
    x_peaks, _ = find_peaks(x_profile, height=hth, distance=20)

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


############################
# Ion state thresholding utils (NumPy-only)
############################

def _crop_roi_np(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Crop with ROI=(x_width,y_width,x_start,y_start)."""
    xw, yw, xs, ys = map(int, roi)
    ys = max(0, ys); xs = max(0, xs)
    y_end = max(ys, min(img.shape[0], ys + yw))
    x_end = max(xs, min(img.shape[1], xs + xw))
    if ys >= y_end or xs >= x_end:
        return np.zeros((0, 0), dtype=img.dtype)
    return img[ys:y_end, xs:x_end]


def normalize_count(
    img: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    bg_roi: tuple[int, int, int, int] | None = None,
    exposure_s: float = 1.0,
) -> dict:
    """
        ROI合計を「1秒あたり」に正規化して返す（ROI画素数では割らない）。
        S = (sum(ROI) - mean(bg_roi)*Npx) / exposure_s
        背景差分は bg_roi（背景ROI）を指定した場合に、bg の平均×Npx を差し引きます。
    Returns:
      {
                "S_norm": float,     # 正規化スカラー（1秒あたり）
        "S_raw": float,      # 背景差し後の生合計（ADU）
        "Npx": int, "bg_mean": float
      }
    """
    roi_img = _crop_roi_np(np.asarray(img), roi)
    Npx = int(roi_img.size)
    if Npx == 0:
        raise ValueError("ROI has zero pixels")

    if bg_roi is not None:
        bg_img = _crop_roi_np(np.asarray(img), bg_roi)
        bg_mean = float(np.mean(bg_img)) if bg_img.size > 0 else 0.0
    else:
        bg_mean = 0.0

    roi_sum = float(np.sum(roi_img))
    S_raw = roi_sum - bg_mean * Npx

    denom = float(exposure_s)
    S_norm = S_raw / (denom if denom > 0 else 1.0)

    return {"S_norm": float(S_norm), "S_raw": float(S_raw), "Npx": Npx, "bg_mean": float(bg_mean)}


def otsu_from_array(arr: np.ndarray, nbins: int = 64) -> float:
    """
    軽量Otsu（NumPyのみ）。arrは1D。
    ヒストグラム→クラス間分散最大のbin下端を返す。
    """
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0

    hist, edges = np.histogram(x, bins=int(max(2, nbins)))
    centers = (edges[:-1] + edges[1:]) / 2.0

    w = hist.astype(float)
    w_total = np.sum(w)
    if w_total <= 0:
        return float(edges[0])

    p = w / w_total
    mu_k = np.cumsum(p * centers)
    omega = np.cumsum(p)
    mu_t = mu_k[-1]

    denom = omega * (1.0 - omega)
    denom[denom <= 0] = np.nan
    var_b = (mu_t * omega - mu_k) ** 2 / denom
    var_b[0] = np.nan
    var_b[-1] = np.nan

    k = int(np.nanargmax(var_b))
    return float(edges[k])


def quick_threshold_from_samples(
    samples: list[float],
    *,
    provisional_tau: float | None = None,
    nbins: int = 64,
    k_sigma: float = 6.0,
) -> dict:
    """
    ROI正規化カウントの小サンプル（例:10個）から閾値を作る。
    判別ロジック:
      - Otsuでτを試算し、p_low=Pr(S<τ), p_high=1-p_low を計算。
      - 両方>=0.1 かつ τが分布内(q05<τ<q95)にあれば 'BIMODAL' として採用。
      - ほぼ全て下側なら 'DARK_ONLY'、ほぼ全て上側なら 'BRIGHT_ONLY'。
      - 中途半端は 'NOT_SURE' とし、暗側扱いに倒す。
    片側系のτは保守的に：
      DARK_ONLY/NOT_SURE: τ = median + k_sigma*sqrt(max(median,1))
      BRIGHT_ONLY       : τ = median - k_sigma*sqrt(max(median,1))
    provisional_tau が与えられたら、それを上下±1のヒステリシス中心として
    片側ケースの安全側に寄せる（上記より優先）。
    Returns:
      {"tau": float, "tau_on": float, "tau_off": float,
       "mode": str,  # "BIMODAL" | "DARK_ONLY" | "BRIGHT_ONLY" | "NOT_SURE"
       "q05": float, "q50": float, "q95": float, "p_low": float, "p_high": float}
    """
    x = np.asarray(samples, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("samples is empty")

    q05, q50, q95 = np.percentile(x, [5, 50, 95])
    tau_otsu = otsu_from_array(x, nbins=int(max(2, nbins)))

    p_low = float(np.mean(x < tau_otsu))
    p_high = 1.0 - p_low

    mode: str
    tau = float(tau_otsu)

    if (p_low >= 0.1 and p_high >= 0.1) and (q05 < tau_otsu < q95):
        mode = "BIMODAL"
    else:
        if p_low >= 0.9:
            mode = "DARK_ONLY"
        elif p_high >= 0.9:
            mode = "BRIGHT_ONLY"
        else:
            mode = "NOT_SURE"

        if provisional_tau is not None:
            tau = float(provisional_tau)
        else:
            sigma_like = np.sqrt(max(q50, 1.0))
            if mode in ("DARK_ONLY", "NOT_SURE"):
                tau = float(q50 + k_sigma * sigma_like)
            else:
                tau = float(q50 - k_sigma * sigma_like)

    tau_on = float(tau + 1.0)
    tau_off = float(tau - 1.0)

    return {
        "tau": float(tau),
        "tau_on": tau_on,
        "tau_off": tau_off,
        "mode": mode,
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "p_low": float(p_low),
        "p_high": float(p_high),
    }


def classify_hysteresis(
    S: float, *,
    prev_state_bright: bool | None,
    tau_on: float,
    tau_off: float,
) -> bool:
    """
    ヒステリシスのみで明暗を判定（デバウンスなし）。
    prev_state_bright が None のときは S> (tau_on+tau_off)/2 で初期化。
    Returns: True=bright, False=dark
    """
    if prev_state_bright is None:
        mid = 0.5 * (float(tau_on) + float(tau_off))
        return bool(S > mid)

    if prev_state_bright:
        return not (S < float(tau_off))
    else:
        return bool(S > float(tau_on))


def bootstrap_threshold_from_stream(
    imgs: list[np.ndarray],
    roi: tuple[int, int, int, int],
    *,
    bg_roi: tuple[int, int, int, int] | None = None,
    exposure_s_list: list[float] | None = None,
    provisional_tau: float | None = None,
    sample_n: int = 10,
) -> dict:
    """
    測定開始時に先頭から sample_n 枚だけ使って閾値を決める。
    各フレームを normalize_count で正規化→ quick_threshold_from_samples。
    exposure_s_list が None なら全て同一露光とみなす。
    Returns: quick_threshold_from_samples と同じdict。
    """
    if sample_n <= 0:
        raise ValueError("sample_n must be positive")

    n = min(sample_n, len(imgs))
    if n == 0:
        raise ValueError("no images provided")

    if exposure_s_list is None:
        exposure_s_list = [1.0] * n
    else:
        if len(exposure_s_list) < n:
            last = float(exposure_s_list[-1]) if exposure_s_list else 1.0
            exposure_s_list = list(exposure_s_list) + [last] * (n - len(exposure_s_list))

    samples: list[float] = []
    for i in range(n):
        info = normalize_count(
            imgs[i], roi,
            bg_roi=bg_roi,
            exposure_s=float(exposure_s_list[i]),
        )
        samples.append(float(info["S_norm"]))

    return quick_threshold_from_samples(samples, provisional_tau=provisional_tau)


def _self_test():
    """
    ダミーデータで正規化・閾値推定・ヒステリシス判定の自己検証を行う。
    - 露光時間のみで正規化するため、露光違いでも S_norm が揃うことを確認。
    - 明暗が混在するサンプルから Otsu で閾値を推定し、ヒステリシスで判定。
    出力は print のみ（プロット無し）。
    """
    rng = np.random.default_rng(42)

    # 画像サイズとROI設定（固定）
    H, W = 80, 160
    roi = (40, 16, 60, 32)  # (x_width, y_width, x_start, y_start)
    bg_roi = (40, 16, 10, 10)

    # 露光を交互に変える（0.05s と 0.1s）
    N = 40
    exposures = np.array([0.05 if (i % 2 == 0) else 0.10 for i in range(N)], dtype=float)

    # 明/暗フラグ（前半は暗多め、後半は明多め）
    is_bright = np.array([(i % 4 in (2, 3)) for i in range(N)], dtype=bool)

    # 背景平均とノイズ（ADU）
    bg_mean_true = 100.0
    read_noise_sigma = 3.0

    # 明状態の信号（ROI合計の1秒あたりターゲット）
    # 正規化は exposure のみなので、平均的な S_norm ≈ signal_per_sec * Npx になる点に注意
    xw, yw, xs, ys = roi
    Npx = xw * yw
    signal_per_sec = 8.0  # 1pxでなくROI合計/秒の強さではなく、後で* Npx 相当の分を織り込む

    imgs = []
    for i in range(N):
        img = rng.normal(loc=bg_mean_true, scale=read_noise_sigma, size=(H, W)).astype(np.float64)
        if is_bright[i]:
            # ROI内に信号を加算：合計が exposure * signal_per_sec * Npx 付近になるように
            signal = signal_per_sec * exposures[i]
            img[ys:ys+yw, xs:xs+xw] += signal
        imgs.append(img)

    # 先頭10枚で初期閾値推定
    info_list = [normalize_count(imgs[i], roi, bg_roi=bg_roi, exposure_s=float(exposures[i])) for i in range(N)]
    samples = [d["S_norm"] for d in info_list[:10]]
    q = quick_threshold_from_samples(samples, provisional_tau=None)

    # 全フレームにヒステリシス判定
    state = None
    results = []
    for i in range(N):
        S = float(info_list[i]["S_norm"])
        state = classify_hysteresis(S, prev_state_bright=state, tau_on=q["tau_on"], tau_off=q["tau_off"])
        results.append(bool(state))

    # 検証サマリ
    s_dark = [info_list[i]["S_norm"] for i in range(N) if not is_bright[i]]
    s_bright = [info_list[i]["S_norm"] for i in range(N) if is_bright[i]]
    print("[SELFTEST] samples(first 10) ->", samples)
    print("[SELFTEST] tau/tau_on/tau_off:", {k: q[k] for k in ("tau", "tau_on", "tau_off", "mode")})
    print("[SELFTEST] mean S_norm dark/bright:", np.mean(s_dark), np.mean(s_bright))
    acc = float(np.mean(results == is_bright))
    print(f"[SELFTEST] hysteresis accuracy vs ground-truth: {acc*100:.1f}% (rough check)")

def main():
    # 環境変数 SELFTEST=1 のときはダミーデータで自己検証を実行
    if os.environ.get("SELFTEST", "0") == "1":
        _self_test()
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    s = build_session_dirs(ts)
    
    # 同一セッション内での保存ファイル連番をメモリで管理
    next_idx = 1

    # ガウシアンフィッティングして大きなROI決定してもいいかも

    # 明状態の取得
    saved, next_idx = get_n_frames_from_buffer(
        expose_time=EXPOSE_TIME,
        roi=ROUGH_ROI,
        session_root=s["root"],
        start_index=next_idx,
    )

    # トリミング範囲決定
    rois = generate_rois_from_image(saved, plot=False)

    # 暗状態の取得
    saved_2, next_idx = get_n_frames_from_buffer(
        expose_time=EXPOSE_TIME,
        roi=ROUGH_ROI,
        session_root=s["root"],
        start_index=next_idx,
    )
    
    # 今回セッションの全フレームを関数で読み込む
    frames = load_session_frames(s["raw"], ts)

    # 全体画像の可視化（軸つき）
    show_npy_2d(frames[0], title="Full frame",
                save_dir=s["plots"], save_name=f"{ts}_full_frame.png")

    # 決定したトリミング範囲でと
    all_crops = []
    for f in frames:
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)

    for i, crop in enumerate(all_crops):
        np.save(os.path.join(s["raw"], f"{ts}_roi{i:02d}.npy"), crop)

    # 閾値を決める関数
    images_for_hist = all_crops if len(all_crops) > 0 else [frames[0]]
    # 閾値とデータを入れると分別しながらプロットしてくれる
    # 閾値があるときは分別してプロットして、light, darkにリストを渡してそれぞれプロットしてもらう（どちらかがない場合もある）
    plot_photon_distribution(light_images=images_for_hist,
                             save_dir=s["plots"],
                             save_name=f"{ts}_photon_dist_light.png")

# 光子数分布をプロットする関数（色々をまとめた関数）


if __name__ == "__main__":
    main()
