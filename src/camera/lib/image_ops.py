import numpy as np


_EDGE_MARGIN_ENV = "ION_ROI_EDGE_MARGIN_PX"


def _get_edge_margin_px() -> int:
    """Return ROI edge margin in pixels from env var.

    If set to a positive integer N, ROI crops will exclude a border of N pixels
    on each side (i.e., shrink ROI by 2N in width/height).
    """
    try:
        import os

        v = os.environ.get(_EDGE_MARGIN_ENV, "0")
        n = int(v)
        return max(0, n)
    except Exception:
        return 0


def crop_roi(
    img: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    edge_margin_px: int | None = None,
) -> np.ndarray:
    """ROI=(x_width,y_width,x_start,y_start) で切り出し。範囲外はクリップ。

    edge_margin_px を指定すると、ROI の端を各辺 edge_margin_px 分だけ除外する
    （=ROIを内側に縮める）。None の場合は環境変数 ION_ROI_EDGE_MARGIN_PX を参照。
    """
    xw, yw, xs, ys = map(int, roi)
    img2 = np.asarray(img)
    Y, X = int(img2.shape[0]), int(img2.shape[1])
    xw = max(1, min(xw, X))
    yw = max(1, min(yw, Y))
    xs = max(0, min(xs, X - xw))
    ys = max(0, min(ys, Y - yw))

    m = _get_edge_margin_px() if edge_margin_px is None else max(0, int(edge_margin_px))
    if m > 0:
        # shrink inside the ROI (avoid using boundary pixels)
        xs = xs + m
        ys = ys + m
        xw = xw - 2 * m
        yw = yw - 2 * m
        if xw <= 0 or yw <= 0:
            return np.zeros((0, 0), dtype=img2.dtype)
        xs = max(0, min(xs, X - xw))
        ys = max(0, min(ys, Y - yw))

    return img2[ys:ys+yw, xs:xs+xw]


def extract_rois_from_image(img: np.ndarray, rois: list) -> list:
    """複数ROIを切り出して返す。"""
    crops = []
    for roi in rois:
        try:
            xw, yw, xs, ys = map(int, roi)
        except Exception:
            continue
        crops.append(crop_roi(img, (xw, yw, xs, ys)))
    return crops


def apply_roi_npy(npy_path: str, roi: list) -> np.ndarray:
    """互換用ラッパー: ファイルを読み、`crop_roi` で切る。"""
    img = np.load(npy_path, allow_pickle=False)
    xw, yw, xs, ys = map(int, roi)
    return crop_roi(img, (xw, yw, xs, ys))
