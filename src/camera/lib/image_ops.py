import numpy as np


def crop_roi(img: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """ROI=(x_width,y_width,x_start,y_start) で切り出し。範囲外はクリップ。"""
    xw, yw, xs, ys = map(int, roi)
    img2 = np.asarray(img)
    Y, X = int(img2.shape[0]), int(img2.shape[1])
    xw = max(1, min(xw, X))
    yw = max(1, min(yw, Y))
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
