import numpy as np


_EDGE_MARGIN_ENV = "ION_ROI_EDGE_MARGIN_PX"


def _get_edge_margin_px() -> int:
    try:
        import os

        v = os.environ.get(_EDGE_MARGIN_ENV, "0")
        n = int(v)
        return max(0, n)
    except Exception:
        return 0


def otsu_from_array(arr: np.ndarray, nbins: int = 64) -> float:
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
    # Degenerate distributions (e.g., almost constant samples) can lead to var_b being all-NaN.
    # In that case, fall back to a robust central value instead of raising.
    if not np.any(np.isfinite(var_b)):
        return float(np.median(x))

    k = int(np.nanargmax(var_b))
    return float(edges[k])


def estimate_threshold_otsu_from_frames(frames: list[np.ndarray], nbins: int = 256) -> float:
    if not frames:
        raise ValueError("No frames provided for threshold estimation.")
    sums = np.array([np.asarray(f, dtype=float).sum() for f in frames], dtype=float)
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


def estimate_threshold_otsu_from_pixels(
    img: np.ndarray,
    *,
    roi: tuple[int, int, int, int] | None = None,
    nbins: int = 256,
) -> float:
    """1枚の画像のピクセル値分布から Otsu 閾値を推定する。

    - roi が指定されれば ROI 内ピクセルのみを対象
    - roi=None の場合は画像全体を対象
    """
    a = np.asarray(img)
    if roi is not None:
        a = _crop_roi_np(a, roi)
    return float(otsu_from_array(a.ravel(), nbins=int(max(2, nbins))))


def split_images_by_threshold(frames: list[np.ndarray], threshold: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    light, dark = [], []
    for f in frames:
        val = float(np.asarray(f, dtype=float).sum())
        (light if val > threshold else dark).append(f)
    return light, dark


# -----------------------------
# Ion-state detection helpers
# -----------------------------

def _crop_roi_np(
    img: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    edge_margin_px: int | None = None,
) -> np.ndarray:
    """Crop with ROI=(x_width,y_width,x_start,y_start).

    edge_margin_px を指定すると ROI の端を各辺 edge_margin_px ピクセル除外する。
    None の場合は環境変数 ION_ROI_EDGE_MARGIN_PX を参照。
    """
    xw, yw, xs, ys = map(int, roi)

    m = _get_edge_margin_px() if edge_margin_px is None else max(0, int(edge_margin_px))
    if m > 0:
        xs = xs + m
        ys = ys + m
        xw = xw - 2 * m
        yw = yw - 2 * m
        if xw <= 0 or yw <= 0:
            return np.zeros((0, 0), dtype=img.dtype)

    ys = max(0, ys)
    xs = max(0, xs)
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
    ROI?????????????????????????????OI?????????????????????
    S = sum(ROI) - sum(bg_roi)
    Returns:
      {
        "S_norm": float,     # S = sum(ROI) - sum(bg_roi)
        "S_raw": float,      # same as S_norm (compat)
        "Npx": int, "bg_mean": float, "bg_sum": float, "roi_sum": float
      }
    """
    roi_img = _crop_roi_np(np.asarray(img), roi)
    Npx = int(roi_img.size)
    if Npx == 0:
        raise ValueError("ROI has zero pixels")

    if bg_roi is not None:
        bg_img = _crop_roi_np(np.asarray(img), bg_roi)
        bg_mean = float(np.mean(bg_img)) if bg_img.size > 0 else 0.0
        bg_sum = float(np.sum(bg_img)) if bg_img.size > 0 else 0.0
    else:
        bg_mean = 0.0
        bg_sum = 0.0

    roi_sum = float(np.sum(roi_img))
    S_raw = roi_sum - bg_sum
    S_norm = S_raw

    return {
        "S_norm": float(S_norm),
        "S_raw": float(S_raw),
        "Npx": Npx,
        "bg_mean": float(bg_mean),
        "bg_sum": float(bg_sum),
        "roi_sum": float(roi_sum),
    }


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
