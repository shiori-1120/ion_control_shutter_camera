import time
import numpy as np
from typing import Optional
import os
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


_EDGE_MARGIN_ENV = "ION_ROI_EDGE_MARGIN_PX"
_SINGLE_ION_ENV = "ION_SINGLE_ION"
_PROFILE_AVG_WINDOW_ENV = "ION_PROFILE_AVG_WINDOW"
DEFAULT_PROFILE_AVG_WINDOW = 21


def _get_profile_avg_window(default: int) -> int:
    """Return moving-average window for profile smoothing.

    Environment variable ION_PROFILE_AVG_WINDOW can override the default.
    To avoid surprising callers that explicitly pass a non-default window,
    we only apply the env override when the provided default equals the
    module default.
    """
    w = int(default)
    # If caller intentionally passed a different window, respect it.
    if w != int(DEFAULT_PROFILE_AVG_WINDOW):
        return max(1, w)
    try:
        v = (os.environ.get(_PROFILE_AVG_WINDOW_ENV, "") or "").strip()
        if not v:
            return max(1, w)
        n = int(v)
        return max(1, n)
    except Exception:
        return max(1, w)


def _is_single_ion_mode() -> bool:
    try:
        v = (os.environ.get(_SINGLE_ION_ENV, "0") or "0").strip().lower()
        return v in ("1", "true", "yes", "y", "on")
    except Exception:
        return False


def _get_edge_margin_px() -> int:
    """Reuse existing ROI edge margin env var for profile fitting.

    Auto ROI detection is sensitive to bright edges / readout artifacts.
    If ION_ROI_EDGE_MARGIN_PX is set, we ignore that many pixels at the
    image boundary when fitting profiles.
    """
    try:
        v = os.environ.get(_EDGE_MARGIN_ENV, "0")
        n = int(v)
        return max(0, n)
    except Exception:
        return 0


def _crop_inner(img: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Crop away a margin from all sides; returns (img_cropped, x0, y0)."""
    a = np.asarray(img)
    Y, X = int(a.shape[0]), int(a.shape[1])
    m = _get_edge_margin_px()
    if m <= 0:
        return a, 0, 0
    # Ensure at least a 1px region remains.
    m = int(min(m, (min(Y, X) - 1) // 2))
    if m <= 0:
        return a, 0, 0
    return a[m:Y - m, m:X - m], m, m


def _box_filter_2d(img: np.ndarray, k: int) -> np.ndarray:
    """Simple local-sum/mean filter (odd kernel). Returns float array."""
    a = np.asarray(img, dtype=float)
    kk = int(k)
    if kk <= 1:
        return a
    if kk % 2 == 0:
        kk += 1
    # Prefer SciPy if available (already a dependency of this module).
    try:
        from scipy.ndimage import uniform_filter  # type: ignore

        # uniform_filter gives local mean; scale doesn't matter for argmax.
        return uniform_filter(a, size=kk, mode="nearest")
    except Exception:
        # NumPy fallback: separable box filter via convolution.
        kernel = np.ones((kk, kk), dtype=float) / float(kk * kk)
        # naive convolution (small k); use padding to keep shape
        pad = kk // 2
        ap = np.pad(a, ((pad, pad), (pad, pad)), mode="edge")
        out = np.zeros_like(a, dtype=float)
        for dy in range(kk):
            for dx in range(kk):
                out += ap[dy:dy + a.shape[0], dx:dx + a.shape[1]] * kernel[dy, dx]
        return out


def _generate_single_ion_roi(img: np.ndarray) -> list:
    """Return a single ROI centered at the brightest local region.

    This is more robust than global argmax (hot pixels) and more robust than
    Lorentz fitting when profiles are dominated by background/edge artifacts.
    """
    img2, x0, y0 = _crop_inner(img)
    a = np.asarray(img2, dtype=float)
    if a.size == 0:
        return []

    # Background suppress: subtract a robust central value.
    base = float(np.median(a[np.isfinite(a)])) if np.any(np.isfinite(a)) else 0.0
    a = a - base

    # Local-average to suppress salt-and-pepper noise.
    k = 9
    try:
        k = int(os.environ.get("ION_ROI_LOCAL_SUM_K", "9"))
    except Exception:
        k = 9
    k = max(3, min(51, int(k)))
    sm = _box_filter_2d(a, k)

    # Find brightest location in the inner-cropped coordinates.
    iy, ix = np.unravel_index(int(np.nanargmax(sm)), sm.shape)

    # ROI box size (square) – keep conservative default; user can override.
    try:
        box = int(os.environ.get("ION_ROI_BOX_PX", "200"))
    except Exception:
        box = 200
    box = max(10, int(box))

    Y, X = int(np.asarray(img).shape[0]), int(np.asarray(img).shape[1])
    x_center = int(ix + x0)
    y_center = int(iy + y0)
    xw = min(box, X)
    yw = min(box, Y)
    xs = int(round(x_center - xw / 2.0))
    ys = int(round(y_center - yw / 2.0))
    xs = max(0, min(xs, X - xw))
    ys = max(0, min(ys, Y - yw))
    return [[int(xw), int(yw), int(xs), int(ys)]]


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


def fit_vertical_profile(img: np.ndarray, *, avg_window: int = DEFAULT_PROFILE_AVG_WINDOW):
    img2, x0, y0 = _crop_inner(img)
    avg_window = _get_profile_avg_window(avg_window)
    y_profile = _moving_average_1d(np.asarray(img2).sum(axis=1), avg_window)
    y_x = np.arange(len(y_profile))
    # Use a simple height threshold to avoid selecting noise-only peaks.
    hth = (float(y_profile.max()) + float(y_profile.min())) / 2.0
    y_peaks, _ = find_peaks(y_profile, height=hth, distance=5)
    y_offset0 = float(np.median(y_profile))
    y_A0 = float(y_profile.max() - y_offset0)
    if len(y_peaks) > 0:
        y_peak_idx = int(y_peaks[int(np.argmax(y_profile[y_peaks]))])
    else:
        # Fallback: pick global max within cropped image.
        y_peak_idx = int(np.argmax(y_profile))
    p0y = [max(0.0, y_A0), float(y_peak_idx), 5.0, y_offset0]
    bounds_y = ([0.0, max(0, y_peak_idx-10), 0.5, 0.0],
                [np.inf, min(len(y_profile)-1, y_peak_idx+10), 200.0, np.inf])
    try:
        popt_y, _ = curve_fit(lorentz, y_x, y_profile, p0=p0y, bounds=bounds_y, maxfev=10000)
        y_fitted = lorentz(y_x, *popt_y)
        y_fwhm = 2.0 * abs(float(popt_y[2]))
        y_center = float(popt_y[1]) + float(y0)
        return {
            'profile': y_profile,
            'x': (y_x + y0),
            'peaks': (y_peaks + y0),
            'fitted': y_fitted,
            'params': popt_y,
            'center': y_center,
            'fwhm': y_fwhm,
        }
    except Exception:
        return None


def fit_horizontal_profile(img: np.ndarray, *, avg_window: int = DEFAULT_PROFILE_AVG_WINDOW):
    img2, x0, y0 = _crop_inner(img)
    avg_window = _get_profile_avg_window(avg_window)
    x_profile = _moving_average_1d(np.asarray(img2).sum(axis=0), avg_window)
    x_x = np.arange(len(x_profile))
    hth = (x_profile.max() + x_profile.min()) / 2.0
    x_peaks, _ = find_peaks(x_profile, height=hth, distance=20)
    if _is_single_ion_mode():
        # Single-ion: fit ONE Lorentz to the horizontal profile.
        x_offset0 = float(np.median(x_profile))
        x_A0 = float(x_profile.max() - x_offset0)
        if len(x_peaks) > 0:
            x_peak_idx = int(x_peaks[int(np.argmax(x_profile[x_peaks]))])
        else:
            x_peak_idx = int(np.argmax(x_profile))

        p0x = [max(0.0, x_A0), float(x_peak_idx), 5.0, x_offset0]
        bounds_x = ([0.0, max(0, x_peak_idx - 10), 0.5, 0.0],
                    [np.inf, min(len(x_profile) - 1, x_peak_idx + 10), 200.0, np.inf])
        try:
            popt_x, _ = curve_fit(lorentz, x_x, x_profile, p0=p0x, bounds=bounds_x, maxfev=10000)
            x_fitted = lorentz(x_x, *popt_x)
            x_center = float(popt_x[1]) + float(x0)
            x_fwhm = 2.0 * abs(float(popt_x[2]))
            return {
                'profile': x_profile,
                'x': (x_x + x0),
                'peaks': (x_peaks + x0),
                'fitted': x_fitted,
                'params': popt_x,
                'centers': [x_center],
                'fwhms': [x_fwhm],
            }
        except Exception:
            return None

    # Multi-ion (legacy): fit sum of Lorentzians.
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
            ctr = float(popt_h[3*i+1]) + float(x0)
            wid = float(popt_h[3*i+2])
            fwhm = 2.0 * abs(wid)
            centers.append(ctr)
            fwhms.append(fwhm)
        return {
            'profile': x_profile,
            'x': (x_x + x0),
            'peaks': (x_peaks + x0),
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
    # Use vertical FWHM only, and expand ROI to 2x FWHM (square ROI).
    width_px = max(1, int(round(float(v_fwhm) * 2.0)))
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
    # Single-ion mode: avoid multi-peak fitting and pick the brightest local region.
    if _is_single_ion_mode():
        rois = _generate_single_ion_roi(img)
        if rois:
            return rois
        # fall back to fit-based method if something went wrong

    results = lorentz_fit_profiles(img, plot)
    return generate_rois_from_analyze_results(results, img.shape)
