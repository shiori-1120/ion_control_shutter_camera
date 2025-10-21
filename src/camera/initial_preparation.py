import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.patches import Rectangle
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
                filename = datetime.datetime.now().strftime(f"{output_path}_{count}.npy")
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
    """
    指定された ROI を用いて npy 画像をトリミングする。
    roi = [h-width, v-width, h-start, v-start]
    戻り値: トリミングされた 2D numpy 配列
    """
    img = np.load(npy_path)
    h_width, v_width, h_start, v_start = map(int, roi)
    img_cropped = img[v_start:v_start+v_width, h_start:h_start+h_width]
    return img_cropped


def plot_profile(profile, xs=None, axis_name='x', fitted_curve=None, centers_fwhm=None, peaks=None, title=None, figsize=(6,3)):
    """
    汎用プロファイルプロット
    - profile: 1D array of summed intensities
    - xs: x coordinates (optional)
    - axis_name: 'x' or 'y' for labeling
    - fitted_curve: tuple (xs, ys) to overplot fitted curve
    - centers_fwhm: list of (center, fwhm) tuples to draw center and FWHM lines
    - peaks: list of peak indices (optional)
    """
    try:
        if xs is None:
            xs = np.arange(len(profile))
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(xs, profile, color='C0', lw=1, label='profile')
        ax.set_xlabel(f'{axis_name} (pixel)')
        ax.set_ylabel('summed intensity')
        if title:
            ax.set_title(title)
        ax.grid(True, linestyle=':', alpha=0.6)

        if fitted_curve is not None:
            fx, fy = fitted_curve
            ax.plot(fx, fy, color='red', lw=1, label='fit')

        if centers_fwhm:
            for c, fwhm in centers_fwhm:
                left = c - fwhm / 2.0
                right = c + fwhm / 2.0
                ax.axvline(c, color='magenta', linestyle='--', linewidth=0.8)
                ax.axvline(left, color='orange', linestyle=':', linewidth=0.6)
                ax.axvline(right, color='orange', linestyle=':', linewidth=0.6)

        if peaks is not None:
            ax.plot(xs[peaks], profile[peaks], 'x', color='k', markersize=6, label='peaks')

        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        return fig, ax
    except Exception:
        # プロット環境がない場合は何もしない
        return None, None


def roi_settings_nofit(img2: np.ndarray):
    """
    ローレンツフィットを行わない ROI 検出（旧実装）。
    img2 は 2D numpy 配列（H, W）を想定し、
    戻り値は [[h_width, v_width, h_start, v_start], ...]
    """
    H, W = img2.shape

    def smooth1d(a, window=5):
        if window <= 1:
            return a
        pad = window // 2
        a_pad = np.pad(a, pad, mode='reflect')
        kernel = np.ones(window, dtype=float) / window
        res = np.convolve(a_pad, kernel, mode='valid')
        return res

    def detect_peaks_1d(a):
        peaks = []
        for i in range(1, len(a)-1):
            if a[i] >= a[i-1] and a[i] >= a[i+1]:
                peaks.append(i)
        return peaks

    def estimate_fwhm_simple(a, idx):
        peak = float(a[idx])
        win = 8
        left_bg = a[max(0, idx-win):idx]
        right_bg = a[idx+1:min(len(a), idx+1+win)]
        bg_candidates = []
        if left_bg.size > 0:
            bg_candidates.append(np.median(left_bg))
        if right_bg.size > 0:
            bg_candidates.append(np.median(right_bg))
        bg = float(np.median(bg_candidates)) if bg_candidates else 0.0
        amp = max(peak - bg, 0.0)
        if amp <= 0:
            return 1.0
        half = bg + amp / 2.0
        i_left = idx
        while i_left > 0 and a[i_left] > half:
            i_left -= 1
        i_right = idx
        while i_right < len(a)-1 and a[i_right] > half:
            i_right += 1

        def interp(i0, i1):
            v0 = a[i0]
            v1 = a[i1]
            if v1 == v0:
                return float(i0)
            return i0 + (half - v0) / (v1 - v0) * (i1 - i0)
        left_x = interp(max(0, i_left), min(len(a)-1, i_left+1))
        right_x = interp(max(0, i_right-1), min(len(a)-1, i_right))
        fwhm = max(1.0, right_x - left_x)
        return fwhm

    # 垂直プロフィール
    y_profile = img2.sum(axis=1)
    y_smooth = smooth1d(y_profile, window=7)
    y_peaks = detect_peaks_1d(y_smooth)
    if len(y_peaks) == 0:
        return []
    y_vals = np.array([y_smooth[p] for p in y_peaks])
    y_peak_idx = int(y_peaks[np.argmax(y_vals)])
    y_fwhm = estimate_fwhm_simple(y_smooth, y_peak_idx)
    v_half = int(np.ceil(y_fwhm / 2.0))
    v_width = max(2, v_half * 2)
    v_start = max(0, y_peak_idx - v_half)
    v_start = min(v_start, H - v_width)

    # 水平プロフィール
    x_profile = img2.sum(axis=0)
    x_smooth = smooth1d(x_profile, window=7)
    x_peaks = detect_peaks_1d(x_smooth)
    if len(x_peaks) == 0:
        return []
    rois = []
    for px in x_peaks:
        px = int(px)
        x_fwhm = estimate_fwhm_simple(x_smooth, px)
        h_half = int(np.ceil(2.0 * x_fwhm))
        h_width = max(2, h_half * 2)
        h_start = max(0, px - h_half)
        h_start = min(h_start, W - h_width)
        rois.append([int(h_width), int(v_width), int(h_start), int(v_start)])
    return rois


def roi_settings(img):
   
    H, W = img.shape

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

    # 垂直方向: 画像を x 方向に積分して y プロファイルを得る
    y_profile = img.sum(axis=1)
    # fit vertical profile and plot using generic plot_profile
    y_x = np.arange(len(y_profile))
    y_y = y_profile
    y_offset0 = float(np.median(y_y))
    y_A0 = float(y_y.max() - y_offset0)
    y_peak_idx = int(np.argmax(y_y))
    # initial guess and bounds for A, ctr, wid, offset
    p0y = [max(0.0, y_A0), float(y_peak_idx), 5.0, y_offset0]
    bounds_y = ([0.0, max(0, y_peak_idx-10), 0.5, 0.0], [np.inf, min(len(y_y)-1, y_peak_idx+10), 200.0, np.inf])

    def lorentz(x, A, x0, wid, offset):
        return A * (wid**2) / ((x - x0)**2 + wid**2) + offset

    try:
        popt_y, pcov_y = curve_fit(lorentz, y_x, y_y, p0=p0y, bounds=bounds_y, maxfev=10000)
        A_fit, y_ctr, y_wid, y_off = popt_y
        y_fitted = lorentz(y_x, *popt_y)
        y_fwhm = 2.0 * abs(float(y_wid))
        plot_profile(y_y, xs=y_x, axis_name='y', fitted_curve=(y_x, y_fitted), centers_fwhm=[(y_ctr, y_fwhm)], peaks=y_peaks, title='Vertical profile (sum over x)')
    except Exception:
        print("roi_settings: vertical lorentz fit failed")
        plot_profile(y_y, xs=y_x, axis_name='y', peaks=y_peaks, title='Vertical profile (fit failed)')
        return False

    # ピーク位置から線幅(FWHM)の半分ずつ上下に取る（合計でほぼ FWHM 相当）
    v_half = int(np.ceil(y_fwhm / 2.0))
    v_width = max(2, v_half * 2)
    v_start = max(0, int(round(y_peak_idx - v_half)))
    v_start = min(v_start, H - v_width)

    # 水平方向: y を積分して x プロファイルを得る
    x_profile = img.sum(axis=0)
    # plot horizontal profile using generic plot_profile; attempt fit if peaks found
    x = np.arange(len(x_profile))
    hth = (x_profile.max() + x_profile.min()) / 2.0
    peaks, _ = find_peaks(x_profile, height=hth, distance=20)

    if len(peaks) > 0:
        guess = []
        lower = []
        upper = []
        median_x = float(np.median(x_profile))
        for p in peaks:
            amp0 = float(max(0.0, x_profile[p] - median_x))
            guess.extend([amp0, float(p), 5.0])
            lower.extend([0.0, max(0, p-10), 0.5])
            upper.extend([np.inf, min(len(x_profile)-1, p+10), 200.0])
        guess.append(median_x)
        lower.append(0.0)
        upper.append(np.inf)

        try:
            popt_h, pcov_h = curve_fit(FUNC, x, x_profile, p0=guess, bounds=(np.array(lower), np.array(upper)), maxfev=20000)
            fitted = FUNC(x, *popt_h)
            num_funcs_h = int((len(popt_h) - 1) / 3)
            centers_fwhm = []
            for i in range(num_funcs_h):
                ctr = float(popt_h[3*i+1])
                wid = float(popt_h[3*i+2])
                fwhm = 2.0 * abs(wid)
                centers_fwhm.append((ctr, fwhm))
            plot_profile(x_profile, xs=x, axis_name='x', fitted_curve=(x, fitted), centers_fwhm=centers_fwhm, peaks=peaks, title='Horizontal profile (sum over y)')
        except Exception:
            print('horizontal fit failed')
            plot_profile(x_profile, xs=x, axis_name='x', peaks=peaks, title='Horizontal profile (fit failed)')
    else:
        plot_profile(x_profile, xs=x, axis_name='x', peaks=peaks, title='Horizontal profile (no peaks)')
    x = np.arange(len(x_profile))
    # ピーク検出（高さは中央値ベースの閾値）
    hth = (x_profile.max() + x_profile.min()) / 2.0
    peaks, props = find_peaks(x_profile, height=hth, distance=20)
    if len(peaks) == 0:
        return []

    # フィッティング用の初期値と bounds を構築
    guess = []
    lower = []
    upper = []
    median_x = float(np.median(x_profile))
    for p in peaks:
        amp0 = float(max(0.0, x_profile[p] - median_x))
        guess.extend([amp0, float(p), 5.0])
        lower.extend([0.0, max(0, p-10), 0.5])
        upper.extend([np.inf, min(len(x_profile)-1, p+10), 200.0])
    # オフセット初期値
    guess.append(median_x)
    lower.append(0.0)
    upper.append(np.inf)

    try:
        popt, pcov = curve_fit(FUNC, x, x_profile, p0=guess, bounds=(
            np.array(lower), np.array(upper)), maxfev=20000)
    except Exception:
        print("roi_settings: horizontal multi-lorentz fit failed")
        return False

    # popt を解析して各ピークの中心と幅を取得し、
    # 複数イオンが並んでいる場合は左端の中心 - margin から右端の中心 + margin
    # までを単一の水平範囲として返す（ユーザ指示）
    num_funcs = int((len(popt) - 1) / 3)
    ctrs = []
    halves = []
    for i in range(num_funcs):
        ctr = float(popt[3*i+1])
        wid = float(popt[3*i+2])
        fwhm = 2.0 * abs(wid)
        # 各ピークについて左右マージンは ceil(FWHM/2)
        h_half = int(np.ceil(fwhm / 2.0))
        ctrs.append(ctr)
        halves.append(h_half)

    # 左右の範囲を決定
    left = min([c - h for c, h in zip(ctrs, halves)])
    right = max([c + h for c, h in zip(ctrs, halves)])
    # 整数座標に変換して画像境界にクランプ
    h_start = max(0, int(np.floor(left)))
    h_end = min(W, int(np.ceil(right)))
    h_width = max(2, h_end - h_start)
    print(h_width)


    roi = [int(h_width), int(v_width), int(h_start), int(v_start)]

    return roi


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


def main():
    npy_path = "C:/Users/karishio/Desktop/single_ion_control/src/camera/input_test/npy/202504231631_217000.npy"
    img = np.load(npy_path)
    # roi = roi_settings(img)
    roi = [400, 50, 400, 160]
    cropped_img = apply_roi_npy(npy_path, roi)
    # print(roi)
    show_npy_2d(cropped_img)

if __name__ == "__main__":
    main()
