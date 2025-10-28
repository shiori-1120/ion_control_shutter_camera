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


expose_time = 0.050
WAIT_MARGIN_SEC = 0.02

# 実験のログ出力用のtxt
# 露光時間、ROI情報、撮影枚数などを記録する
# npy以外のプロットに必要なデータを格納する（フィッティングなど）
# プロットは時間がかかるから後からでもできるようにする
# テキストよりも最適なフォーマットがある？
def log_experiment_details(log_path: str, expose_time: float, rois: list, n_frames: int):
    pass


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

# h, v はx, yに変更
# ROIは一つだけ受け取る
# フォルダ名を受け取る
# 引数をなるべく減らしたい
# なくせそうな引数
# n_frames(トリガで指定), 
# timestamp（タイムスタンプはルートのみ、ファイル名には番号つけるだけ）
# capture_duration_sec トリガが終わってからx秒後に停止、またはエンターで停止で対応
def get_n_frames_from_buffer(expose_time: float = 0.100, roi: Optional[list] = None, session_root: Optional[str] = None):

    idle_timeout_sec = 10.0

    # 出力先
    if session_root is None:
        session_root = os.path.join(os.path.dirname(__file__), "output", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
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

        idx = 1
        saved = 0
        last_saved = time.time()

        while True:
            # Enter で即終了
            if _enter_pressed():
                print("[capture] Enter pressed. stopping...")
                break

            # idle_timeout を超えたら終了（少なくとも一枚保存済みに限定しない）
            if (time.time() - last_saved) >= float(idle_timeout_sec):
                print(f"[capture] idle {idle_timeout_sec}s. stopping...")
                break

            # 待機時間は idle_timeout を超えないよう小刻みに待つ
            remaining_idle = max(0.05, float(idle_timeout_sec) - (time.time() - last_saved))
            dynamic_timeout = max(0.001, min(wait_timeout_sec, remaining_idle))

            ok, _ = qCMOS.wait_for_frame_ready(dynamic_timeout)
            if not ok:
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
    except KeyboardInterrupt:
        pass
    finally:
        qCMOS.StopCapture()
        qCMOS.ReleaseBuf()
        qCMOS.CloseUninitCamera()



# 新しい閾値評価関数を追加


# TODO: 引数をndarrayに変更
def apply_roi_npy(npy_path: str, roi: list):
    img = np.load(npy_path)
    h_width, v_width, h_start, v_start = map(int, roi)
    img_cropped = img[v_start:v_start+v_width, h_start:h_start+h_width]
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




# TODO: 変数名が分かりにくい。ローレンツフィッティングしていることを明示的に。
def analyze_ion_profiles(img):
    results = {'vertical': None, 'horizontal': None}

    # 垂直プロファイルのフィッティング
    vertical_fit = fit_vertical_profile(img)
    results['vertical'] = vertical_fit

    # 水平プロファイルのフィッティング
    start = time.time()
    horizontal_fit = fit_horizontal_profile(img)
    results['horizontal'] = horizontal_fit
    end = time.time()
    print("Horizontal fit time:", end - start)

    return results

# analyze results -> ローレンツフィッティング
def generate_rois_from_analyze_results(results: dict, img_shape) -> list:
    """
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

    avg_linewidth = float((v_fwhm + np.mean(fwhms_x)) / 2.0)
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

# plotオプション付き
# but プロットはnnanalyze_ion_profilesではやらずに追加する
def generate_rois_from_image(img: np.ndarray, plot: bool = False) -> list:
    results = analyze_ion_profiles(img, plot=plot)
    if v_full:
        plot_profile(v_full['profile'], xs=v_full['x'],
                     fitted_curve=(v_full['x'], v_full['fitted']),
                     peaks=v_full['peaks'],
                     centers_fwhm=[(v_full['center'], v_full['fwhm'])],
                     title="Full image vertical", axis_name='Y Pixel',
                     save_dir=s["plots"], save_name=f"{ts}_full_vertical_fit.png")
    if h_full:
        plot_profile(h_full['profile'], xs=h_full['x'],
                     fitted_curve=(h_full['x'], h_full['fitted']),
                     peaks=h_full['peaks'],
                     centers_fwhm=list(zip(h_full['centers'], h_full['fwhms'])),
                     title="Full image horizontal (multi-peak)", axis_name='X Pixel',
                     save_dir=s["plots"], save_name=f"{ts}_full_horizontal_fit.png")
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
    H = int(img.shape[1])  # width (x)
    V = int(img.shape[0])  # height (y)
    ax.set_xlabel('X (pixel index)')
    ax.set_ylabel('Y (pixel index)')
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
    """
    イオンの明状態・暗状態を判別する関数。
    閾値以下の光子数の割合が全体の半分を超える場合は暗状態(False)、
    そうでない場合は明状態(True)と判定。
    """
    # y軸方向に積分して1次元の光子数分布をl取得
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

    img2 = np.asarray(img)

    V, H = int(img2.shape[0]), int(img2.shape[1])

    crops = []
    for roi in rois:
        try:
            h_width, v_width, h_start, v_start = map(int, roi)
        except Exception:
            continue

        # マイナスや過大をクリップ
        h_width = max(1, min(h_width, H))
        v_width = max(1, min(v_width, V))
        h_start = max(0, min(h_start, H - h_width))
        v_start = max(0, min(v_start, V - v_width))

        v_end = v_start + v_width
        h_end = h_start + h_width

        if v_start >= v_end or h_start >= h_end:
            continue

        crop = img2[v_start:v_end, h_start:h_end]
        crops.append(crop)

    return crops


# イオンの個数が変化していないことを確認する関数
def verify_ion_count_consistency(img: np.ndarray, ion_positions) -> bool:
    """
    イオンの個数が変化していないか（= 期待個数と検出個数が同じか）だけを True/False で返す。
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

    # ガウシアンフィッティングして大きなROI決定してもいいかも
    
    
    # 明状態を取得する
    # 一枚分のデータだけ返すようにしてもいいかも
    # 明状態の取得（ROIは1つ想定。未指定ならフルフレーム）
    get_n_frames_from_buffer(
        expose_time=expose_time,
        roi=[600, 100, 2976, 984],
        session_root=s["root"],
        idle_timeout_sec=3.0,
    )
    
    # 暗状態の取得
    # 明状態と暗状態を分けなくてもいいかも。画像をたくさん撮って、それを分布して、閾値を決めて、それから暗状態と明状態を分けてもいいかも

    # 関数化する
    # "raw", "light", "dark"をenumで管理してそのディレクトリのndarrayをリストで受け取れるようにする
    # ② raw-data ディレクトリから今回セッションの全フレームをリストで読み込む
    # get_n_frames_from_bufferで１枚だけreturnを受けて、フォルダ全体を取り出すのは後にする（カメラを使う時間をなるべく短くしたい）
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

    # 全体画像の可視化（軸つき）
    show_npy_2d(frames[0], title="Full frame",
                save_dir=s["plots"], save_name=f"{ts}_full_frame.png")

    # lightの1枚の画像に対してのみフィットプロットを行う
    v_full = fit_vertical_profile(frames[0])
    if v_full:
        plot_profile(v_full['profile'], xs=v_full['x'],
                     fitted_curve=(v_full['x'], v_full['fitted']),
                     peaks=v_full['peaks'],
                     centers_fwhm=[(v_full['center'], v_full['fwhm'])],
                     title="Full image vertical", axis_name='Y Pixel',
                     save_dir=s["plots"], save_name=f"{ts}_full_vertical_fit.png")

    h_full = fit_horizontal_profile(frames[0])
    if h_full:
        plot_profile(h_full['profile'], xs=h_full['x'],
                     fitted_curve=(h_full['x'], h_full['fitted']),
                     peaks=h_full['peaks'],
                     centers_fwhm=list(zip(h_full['centers'], h_full['fwhms'])),
                     title="Full image horizontal (multi-peak)", axis_name='X Pixel',
                     save_dir=s["plots"], save_name=f"{ts}_full_horizontal_fit.png")


    # roiの自動生成

    # ④ 自動トリミング（全フレーム）→ 小さい画像を1つのリストに集約
    # 暗状態と明状態で独立したループにする
    # 同時に保存もする
    all_crops = []
    for f in frames:
        rois = generate_rois_from_image(f, plot=False)
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)

    # 上のループで保存するからここは削除
    # ⑤ ROIは保存のみ（個別フィットのプロット・保存はしない）
    for i, crop in enumerate(all_crops):
        np.save(os.path.join(s["raw"], f"{ts}_roi{i:02d}.npy"), crop)

    # ⑥ 光子数分布（明）: トリミング（ROI）した範囲を対象に集計
    # 暗状態も入れる
    images_for_hist = all_crops if len(all_crops) > 0 else [frames[0]]
    plot_photon_distribution(light_images=images_for_hist,
                             save_dir=s["plots"],
                             save_name=f"{ts}_photon_dist_light.png")


if __name__ == "__main__":
    main()
