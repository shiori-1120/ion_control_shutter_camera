import time
import numpy as np
from typing import Optional
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# Optional simple moving average to stabilize profiles
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
    smoothed = np.convolve(prof_pad, kernel, mode='valid')
    return smoothed


def lorentz(x, A, x0, wid, offset):
    return A * (wid**2) / ((x - x0)**2 + wid**2) + offset


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


def fit_vertical_profile(img: np.ndarray, *, avg_window: int = 21):
    y_profile = _moving_average_1d(np.asarray(img).sum(axis=1), avg_window)
    y_x = np.arange(len(y_profile))
    y_peaks, _ = find_peaks(y_profile, distance=5)
    y_offset0 = float(np.median(y_profile))
    y_A0 = float(y_profile.max() - y_offset0)
    y_peak_idx = int(np.argmax(y_profile))
    p0y = [max(0.0, y_A0), float(y_peak_idx), 5.0, y_offset0]
    bounds_y = ([0.0, max(0, y_peak_idx-10), 0.5, 0.0],
                [np.inf, min(len(y_profile)-1, y_peak_idx+10), 200.0, np.inf])
    try:
        popt_y, _ = curve_fit(lorentz, y_x, y_profile, p0=p0y, bounds=bounds_y, maxfev=10000)
        y_fitted = lorentz(y_x, *popt_y)
        y_fwhm = 2.0 * abs(float(popt_y[2]))
        return {
            'profile': y_profile,
            'x': y_x,
            'peaks': y_peaks,
            'fitted': y_fitted,
            'params': popt_y,
            'center': float(popt_y[1]),
            'fwhm': y_fwhm,
        }
    except Exception:
        return None


def fit_horizontal_profile(img: np.ndarray, *, avg_window: int = 21):
    x_profile = _moving_average_1d(np.asarray(img).sum(axis=0), avg_window)
    x_x = np.arange(len(x_profile))
    hth = (x_profile.max() + x_profile.min()) / 2.0
    x_peaks, _ = find_peaks(x_profile, height=hth, distance=20)
    if len(x_peaks) == 0:
        return None
    guess = []
    lower = []
    upper = []
    median_x = float(np.median(x_profile))
    for p in x_peaks:
        amp0 = float(max(0.0, x_profile[p] - median_x))
        guess.extend([amp0, float(p), 5.0])
        lower.extend([0.0, max(0, p-10), 0.5])
        upper.extend([np.inf, min(len(x_profile)-1, p+10), 200.0])
    guess.append(median_x)
    lower.append(0.0)
    upper.append(np.inf)
    try:
        popt_h, _ = curve_fit(FUNC, x_x, x_profile, p0=guess,
                              bounds=(np.array(lower), np.array(upper)), maxfev=20000)
        x_fitted = FUNC(x_x, *popt_h)
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
            'fwhms': fwhms,
        }
    except Exception:
        return None


def lorentz_fit_profiles(img: np.ndarray, plot: bool = False) -> dict:
    results = {'vertical': None, 'horizontal': None}
    results['vertical'] = fit_vertical_profile(img)
    start = time.time()
    results['horizontal'] = fit_horizontal_profile(img)
    _ = time.time() - start
    return results


def generate_rois_from_analyze_results(results: dict, img_shape) -> list:
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
    X = int(img_shape[1])
    Y = int(img_shape[0])
    rois = []
    for x_center in centers_x:
        x_start = int(round(x_center - x_width / 2.0))
        y_start = int(round(y_center - y_width / 2.0))
        x_start = max(0, min(x_start, X - x_width))
        y_start = max(0, min(y_start, Y - y_width))
        rois.append([x_width, y_width, x_start, y_start])
    return rois


def generate_rois_from_image(img: np.ndarray, plot: bool = False) -> list:
    results = lorentz_fit_profiles(img, plot)
    return generate_rois_from_analyze_results(results, img.shape)
