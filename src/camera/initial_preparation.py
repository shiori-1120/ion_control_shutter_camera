import numpy as np
import time
import os
import datetime
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


expose_time = 0.100


def get_n_frames_from_buffer(n_frames, expose_time=0.100, rois=None):
    # rois = [h-width, v-width, h-start, v-start]
    now = datetime.datetime.now()
    timestamp_dir = now.strftime("%Y%m%d_%H%M")
    base_output = os.path.join(os.path.dirname(__file__), "output")
    output_path = os.path.join(base_output, timestamp_dir)
    os.makedirs(output_path, exist_ok=True)
    try:
        qCMOS = Control_qCMOScamera()
        qCMOS.OpenCamera_GetHandle()
        for roi in rois:
            qCMOS.SetParameters(expose_time, roi[0], roi[1], roi[2], roi[3])
            qCMOS.StartCapture()
            count = 1
            if count <= n_frames:
                time.sleep(expose_time)
                time.sleep(0.1)
                data = qCMOS.GetLastFrame()
                time.sleep(0.006)
                img = data[1].astype(np.float64)
                filename = datetime.datetime.now().strftime(
                    f"{output_path}_{count}.npy")
                out_file = os.path.join(output_path, filename)
                np.save(out_file, img)
                # TODO: 保存が成功したら count をインクリメントする
                count += 1

    finally:
        qCMOS.StopCapture()
        qCMOS.ReleaseBuf()
        qCMOS.CloseUninitCamera()

# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする
# cameraの接続確認はoneshotの関数内で撮影で行う
# 新しい閾値評価関数を追加


def apply_roi_npy(npy_path: str, roi: list):
    img = np.load(npy_path)
    h_width, v_width, h_start, v_start = map(int, roi)
    img_cropped = img[v_start:v_start+v_width, h_start:h_start+h_width]
    return img_cropped


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
                 centers_fwhm=None, title='', axis_name='Pixel'):
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
    plt.show()

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


def show_npy_2d(img: np.ndarray, origin: str = 'lower', figsize=(6, 6), title: str = None):

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
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_photon_distribution(img):
    """
    y軸方向に積分した光子数分布をプロットする。
    縦軸は全イベント数で正規化したヒストグラムとして表示。

    Args:
        img (np.ndarray): 2D画像配列。
    """
    # y軸方向に積分して光子数のヒストグラムを作成
    photon_counts = img.sum(axis=0)

    # 光子数の範囲から適切なビン数を決定
    bins = int(np.sqrt(len(photon_counts)))

    # ヒストグラムを作成し、全要素数で正規化
    plt.figure(figsize=(10, 5))
    plt.hist(photon_counts, bins=bins, density=False,
             alpha=0.7, edgecolor='black')
    plt.axvline(photon_counts.mean(), color='r', linestyle='--',
                label=f'Mean: {photon_counts.mean():.2f}')

    plt.xlabel('Photon Count')
    plt.ylabel('Frequency (normalized)')
    plt.title('Photon Distribution (integrated over y-axis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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


# イオンの個数が変化していないことを確認する関数
def verify_ion_count_consistency(ndarray, ion_positions):
    pass  # 実装はここに記述してください


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
    roi = [400, 50, 400, 160]  # [h-width, v-width, h-start, v-start]
    exposure = 0.100
    frame = capture_roi_image(exposure, roi)

    start_time = time.time()
    results = analyze_ion_profiles(frame.astype(np.float64))
    elapsed = time.time() - start_time
    print("Profile analysis time:", elapsed)
    print(results)

    # Optional visualization utilities
    # show_npy_2d(frame)
    # plot_photon_distribution(frame)


if __name__ == "__main__":
    main()
