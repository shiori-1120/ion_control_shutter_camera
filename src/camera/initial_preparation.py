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


expose_time = 0.050
WAIT_MARGIN_SEC = 0.02


def build_session_dirs(timestamp: str, base_parent: Optional[str] = None) -> Dict[str, str]:
    """Create and return common session directories for a given timestamp.

    Returns dict with keys: 'root', 'raw', 'plots'.
    """
    if base_parent is None:
        base_parent = os.path.join(os.path.dirname(__file__), "output")

    root = os.path.join(base_parent, timestamp)
    raw = os.path.join(root, "raw-data")
    plots = os.path.join(root, "plots")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return {"root": root, "raw": raw, "plots": plots}

# ROIは一つだけ受け取る
def get_n_frames_from_buffer(n_frames,
                             expose_time: float = 0.100,
                             rois=None,
                             timestamp: Optional[str] = None,
                             session_root: Optional[str] = None,
                             capture_duration_sec: float = 10.0):
    # rois = [h-width, v-width, h-start, v-start]

    # store raw frames under common raw-data folder
    output_path = os.path.join(session_root, "raw-data")
    os.makedirs(output_path, exist_ok=True)

    # ROI が未指定のときはフルフレームで取得する
    if rois and len(rois) > 0:
        # Use first ROI for capture
        h_width, v_width, h_start, v_start = map(int, rois[0])
    else:
        h_width = v_width = h_start = v_start = None
    wait_timeout_sec = max(float(expose_time) + WAIT_MARGIN_SEC, 0.05)

    qCMOS = Control_qCMOScamera()
    qCMOS.OpenCamera_GetHandle()
    try:
        qCMOS.SetParameters(expose_time, h_width, v_width, h_start, v_start)
        qCMOS.StartCapture()

        idx = 1
        loop_ts = timestamp
        saved = 0
        # 仕様変更: 呼び出し時点から capture_duration_sec の間だけ取り込み続ける
        window_start = time.time()
        window_end = window_start + float(capture_duration_sec)

        # 仕様: 関数開始時点から一定時間だけ取り込みループを回す
        while True:
            now = time.time()
            if now >= window_end:
                break
            remaining = window_end - now
            # 残り時間を超えないように待機タイムアウトを調整
            dynamic_timeout = max(0.001, min(wait_timeout_sec, remaining))

            ok, _ = qCMOS.wait_for_frame_ready(dynamic_timeout)
            if not ok:
                # タイムアウト: 残り時間があれば次ループへ
                continue
            data = qCMOS.GetLastFrame()
            img = data[1]
            if img.size == 0 or not np.any(img):
                # 空フレームは捨てる
                continue
            filename = f"{loop_ts}_{idx:04d}.npy"
            np.save(os.path.join(output_path, filename), img)
            idx += 1
            saved += 1
    except KeyboardInterrupt:
        pass
    finally:
        qCMOS.StopCapture()
        qCMOS.ReleaseBuf()
        qCMOS.CloseUninitCamera()

# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする
# cameraの接続確認はoneshotの関数内で撮影で行う
# 新しい閾値評価関数を追加

# TODO: 引数をndarrayにしたほうがいいかも


def apply_roi_npy(npy_path: str, roi: list):
    img = np.load(npy_path)
    h_width, v_width, h_start, v_start = map(int, roi)
    img_cropped = img[v_start:v_start+v_width, h_start:h_start+h_width]
    return img_cropped

# いらないかも


def capture_roi_image(exposure_time: float, roi: list, wait_margin: float = 0.01) -> np.ndarray:
    """Capture a single frame from the camera using DCAM subarray ROI."""
    if Control_qCMOScamera is None:
        raise RuntimeError(
            "Camera control module is unavailable in this environment.")

    roi_int = list(map(int, roi))
    cam = Control_qCMOScamera()
    cam.OpenCamera_GetHandle()
    try:
        frame = cam.capture_roi_frame(exposure_time, roi_int, wait_margin)
        return frame.astype(np.float64)
    finally:
        cam.ReleaseBuf()
        cam.CloseUninitCamera()

# 汎用的な1Dプロット関数。


def plot_profile(data, xs=None, fitted_curve=None, peaks=None,
                 centers_fwhm=None, title='', axis_name='Pixel',
                 save_dir: Optional[str] = None, save_name: Optional[str] = None):
    plt.figure(figsize=(10, 4))

    # X軸が指定されていなければ作成
    if xs is None:
        xs = np.arange(len(data))

    # 元データをプロット (マーカー付きの線)
    plt.plot(xs, data, '.-', label='Data', alpha=0.7)

    # ピーク位置を 'x' でプロット
    if peaks is not None and len(peaks) > 0:
        # xs[peaks] で正しいX座標を指定
        plt.plot(xs[peaks], data[peaks], 'x', ms=10,
                 mew=2, label='Detected Peaks')

    # フィット曲線をプロット
    if fitted_curve is not None:
        x_fit, y_fit = fitted_curve
        plt.plot(x_fit, y_fit, 'r-', lw=2, label='Fitted Curve')

    # 中心とFWHM (半値全幅) をプロット
    if centers_fwhm is not None:
        # ラベルが重複しないように工夫
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

# 垂直プロファイルのフィッティング結果をプロットする。


def plot_vertical_profile(fit_result):
    if fit_result is None:
        return

    y_profile = fit_result['profile']
    y_x = fit_result['x']
    y_fitted = fit_result['fitted']
    y_peaks = fit_result['peaks']
    y_ctr = fit_result['center']
    y_fwhm = fit_result['fwhm']

    plot_profile(y_profile, xs=y_x, axis_name='Y Pixel',
                 fitted_curve=(y_x, y_fitted),
                 centers_fwhm=[(y_ctr, y_fwhm)],
                 peaks=y_peaks,
                 title='Vertical profile (sum over x)')


def plot_horizontal_profile(fit_result):
    # 水平プロファイルのフィッティング結果をプロットする。
    if fit_result is None:
        return

    x_profile = fit_result['profile']
    x_x = fit_result['x']
    x_fitted = fit_result['fitted']
    x_peaks = fit_result['peaks']
    centers = fit_result['centers']
    fwhms = fit_result['fwhms']

    # 複数ピークの場合、center と fwhm をペアリング
    centers_fwhm_h = [(center, fwhm) for center, fwhm in zip(centers, fwhms)]

    plot_profile(x_profile, xs=x_x, axis_name='X Pixel',
                 fitted_curve=(x_x, x_fitted),
                 centers_fwhm=centers_fwhm_h,
                 peaks=x_peaks,
                 title='Horizontal profile (sum over y)')


# TODO: 変数名が分かりにくい
def analyze_ion_profiles(img, plot=False):
    # 2D画像から垂直・水平プロファイルを抽出し、ローレンツフィッティングを実行する。
    results = {'vertical': None, 'horizontal': None}

    # 垂直プロファイルのフィッティング
    vertical_fit = fit_vertical_profile(img)
    results['vertical'] = vertical_fit

    if plot and vertical_fit is not None:
        plot_vertical_profile(vertical_fit)

    # 水平プロファイルのフィッティング
    start = time.time()
    horizontal_fit = fit_horizontal_profile(img)
    results['horizontal'] = horizontal_fit
    end = time.time()
    print("Horizontal fit time:", end - start)

    if plot and horizontal_fit is not None:
        plot_horizontal_profile(horizontal_fit)

    return results


def generate_rois_from_analyze_results(results: dict, img_shape) -> list:
    """
    analyze_ion_profiles(img) の結果に基づき、複数ROIを生成して返す。

    仕様:
    - 縦（single）：center_y, fwhm_y を使用
    - 横（multi）：centers_x[], fwhms_x[] を使用し、横FWHMの平均を算出
    - ROI の線幅は (縦FWHM と 横FWHM平均) の平均値を採用（上下左右とも同じピクセル幅）
    - 各ROIは [h-width, v-width, h-start, v-start] 形式（DCAM subarrayの順序に合わせる）
    - 画像外にはみ出さないよう開始座標をクリップ

    Args:
        results (dict): analyze_ion_profiles(img) の戻り値 { 'vertical': {...}, 'horizontal': {...} }
        img_shape (tuple): 画像配列の shape (V, H) または (height, width)

    Returns:
        list[list[int]]: ROI のリスト。各要素は [h-width, v-width, h-start, v-start]
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

    if len(fwhms_x) == 0:
        raise ValueError('No horizontal FWHMs found')

    avg_linewidth = float((v_fwhm + np.mean(fwhms_x)) / 2.0)
    # 線幅をそのまま総幅（ピクセル）として用いる
    width_px = max(1, int(round(avg_linewidth)))
    h_width = width_px
    v_width = width_px

    H = int(img_shape[1])  # width (x)
    V = int(img_shape[0])  # height (y)

    rois = []
    for x_center in centers_x:
        h_start = int(round(x_center - h_width / 2.0))
        v_start = int(round(y_center - v_width / 2.0))

        # 画像内に収める（最低限のクリップ）
        h_start = max(0, min(h_start, H - h_width))
        v_start = max(0, min(v_start, V - v_width))

        rois.append([h_width, v_width, h_start, v_start])

    return rois


def generate_rois_from_image(img: np.ndarray, plot: bool = False) -> list:
    """
    便利関数: 画像から直接プロファイル解析を行い、ROI リストを返す。

    Args:
        img (np.ndarray): 2D 画像。
        plot (bool): 解析時にプロファイルの可視化を行う場合 True。

    Returns:
        list[list[int]]: ROI のリスト [h-width, v-width, h-start, v-start]
    """
    results = analyze_ion_profiles(img, plot=plot)
    return generate_rois_from_analyze_results(results, img.shape)


def show_npy_2d(img: np.ndarray,
                origin: str = 'lower',
                figsize=(6, 6),
                title: str = None,
                save_dir: Optional[str] = None,
                save_name: Optional[str] = None):

    vmin = None
    vmax = None

    if vmin is None or vmax is None:
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        if vmin is None:
            vmin = p1
        if vmax is None:
            vmax = p99
        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(img, cmap='gray', origin=origin, vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)
        if title:
            ax.set_title(title)
        # 軸とインデックス（ピクセル）を表示
        H = int(img.shape[1])  # width (x)
        V = int(img.shape[0])  # height (y)
        ax.set_xlabel('X (pixel index)')
        ax.set_ylabel('Y (pixel index)')
        # ほどよい間隔で目盛りを配置
        step_x = max(1, H // 8)
        step_y = max(1, V // 8)
        xticks = list(range(0, H, step_x))
        yticks = list(range(0, V, step_y))
        if (H - 1) not in xticks:
            xticks.append(H - 1)
        if (V - 1) not in yticks:
            yticks.append(V - 1)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

    plt.tight_layout()
    # save if requested
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
    """
    y軸方向に積分した光子数ヒストグラムを明状態・暗状態で比較表示する。

    Args:
        light_images (list[np.ndarray] | None): 明状態画像のリスト。
        dark_images (list[np.ndarray] | None): 暗状態画像のリスト。
    """
    light_images = light_images or []
    dark_images = dark_images or []

    if not light_images and not dark_images:
        raise ValueError(
            "At least one of light_images or dark_images must contain data.")

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
    """
    イオンの明状態・暗状態を判別する関数。
    閾値以下の光子数の割合が全体の半分を超える場合は暗状態(False)、
    そうでない場合は明状態(True)と判定。
    """
    # y軸方向に積分して1次元の光子数分布を取得
    photon_counts = img.sum(axis=0)

    # 閾値以下のデータ点の数を取得
    dark_count = np.sum(photon_counts <= threshold)

    # 全データ点に対する、閾値以下のデータ点の割合を計算
    dark_ratio = dark_count / len(photon_counts)

    # 暗状態のデータが半分以上なら暗状態(False)、そうでなければ明状態(True)
    return dark_ratio <= 0.5

# roiのリストと2dndarrayの配列を受け取る。
# それぞれのROIに対して、対応する画像データを抽出する。
# 返り値はndarrayのリスト


def extract_rois_from_image(img: np.ndarray, rois: list) -> list:
    """
    2D画像と ROI リストから、それぞれの領域を切り出して返す。

    Args:
        img (np.ndarray): 2D 画像 (V,H)
        rois (list): [[h-width, v-width, h-start, v-start], ...]

    Returns:
        list[np.ndarray]: 切り出した小画像のリスト。無効な ROI はスキップ。
    """
    if img is None:
        raise ValueError("img is None")
    if rois is None:
        return []

    img2 = np.asarray(img)
    if img2.ndim < 2:
        raise ValueError("img must be a 2D array")

    V, H = int(img2.shape[0]), int(img2.shape[1])

    crops = []
    for roi in rois:
        try:
            h_width, v_width, h_start, v_start = map(int, roi)
        except Exception:
            # ROI 形式が不正ならスキップ
            continue

        # マイナスや過大をクリップ
        h_width = max(1, min(h_width, H))
        v_width = max(1, min(v_width, V))
        h_start = max(0, min(h_start, H - h_width))
        v_start = max(0, min(v_start, V - v_width))

        v_end = v_start + v_width
        h_end = h_start + h_width

        # 最終防御: 形が崩れていたらスキップ
        if v_start >= v_end or h_start >= h_end:
            continue

        crop = img2[v_start:v_end, h_start:h_end]
        crops.append(crop)

    return crops

# イオンの個数が変化していないことを確認する関数


def verify_ion_count_consistency(img: np.ndarray, ion_positions) -> bool:
    """
    イオンの個数が変化していないか（= 期待個数と検出個数が同じか）だけを True/False で返す。

    Args:
        img (np.ndarray): 2D 画像。
        ion_positions (list[float|int]): 期待するイオンの x 位置（長さが期待個数）。

    Returns:
        bool: 個数が一致していれば True、合わなければ False。
              フィット失敗やピーク検出ゼロのときも False。
    """
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
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    s = build_session_dirs(ts)

    get_n_frames_from_buffer(
        1, expose_time, [[600, 100, 2976, 984]], ts, s["root"])

    # ② raw-data ディレクトリから今回セッションの全フレームをリストで読み込む
    raw_dir = s["raw"]
    all_paths = []
    for name in sorted(os.listdir(raw_dir)):
        if not name.endswith('.npy'):
            continue
        if not name.startswith(f"{ts}_"):
            continue
        stem = os.path.splitext(name)[0]
        seq = stem.split("_")[-1]
        if len(seq) == 4 and seq.isdigit():
            all_paths.append(os.path.join(raw_dir, name))

    if not all_paths:
        raise RuntimeError("No raw frames found for this session.")

    frames = [np.load(p) for p in all_paths]
    first_path = all_paths[0]

    show_npy_2d(frames[0], title="not ROI",
                save_dir=s["plots"], save_name=f"{ts}_not_roi.png")

    # # ② ダーク領域トリミング
    # dark_roi = [15, 15, 300, 100]
    # dark = apply_roi_npy(first_path, dark_roi)

    # # ③ 保存
    # np.save(os.path.join(s["raw"], f"{ts}_dark.npy"), dark)
    # show_npy_2d(dark, title="Dark ROI",
    #             save_dir=s["plots"], save_name=f"{ts}_dark_roi.png")

    # ④ 自動トリミング（全フレーム）→ 小さい画像を1つのリストに集約
    all_crops = []
    for f in frames:
        rois = generate_rois_from_image(f, plot=False)
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)

    # ⑤ ROI保存とフィット図（画像ごとに分けず、通し番号で保存）
    for i, crop in enumerate(all_crops):
        np.save(os.path.join(s["raw"], f"{ts}_roi{i:02d}.npy"), crop)
        show_npy_2d(
            crop, title=f"ROI {i}", save_dir=s["plots"], save_name=f"{ts}_roi{i:02d}.png")
        v = fit_vertical_profile(crop)
        if v:
            plot_profile(v['profile'], xs=v['x'], fitted_curve=(v['x'], v['fitted']),
                         peaks=v['peaks'], centers_fwhm=[
                             (v['center'], v['fwhm'])],
                         title=f"ROI {i} vertical", axis_name='Y Pixel',
                         save_dir=s["plots"], save_name=f"{ts}_roi{i:02d}_vertical_fit.png")
        h = fit_horizontal_profile(crop)
        if h:
            plot_profile(h['profile'], xs=h['x'], fitted_curve=(h['x'], h['fitted']),
                         peaks=h['peaks'], centers_fwhm=list(
                             zip(h['centers'], h['fwhms'])),
                         title=f"ROI {i} horizontal", axis_name='X Pixel',
                         save_dir=s["plots"], save_name=f"{ts}_roi{i:02d}_horizontal_fit.png")

    # ⑥ 光子数分布（明）
    plot_photon_distribution(light_images=[
                             frames[0]], save_dir=s["plots"], save_name=f"{ts}_photon_dist_light.png")


if __name__ == "__main__":
    main()
