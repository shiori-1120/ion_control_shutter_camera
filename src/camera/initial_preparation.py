# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

import numpy as np
import time
import os
import datetime
import re
from typing import Optional, Dict
from .lib.io_session import build_session_dirs as io_build_session_dirs, list_session_frame_paths as io_list_session_frame_paths, load_session_frames as io_load_session_frames
from .lib.image_ops import extract_rois_from_image as ops_extract_rois_from_image, apply_roi_npy as ops_apply_roi_npy
# plotting is provided via lib.plotting
from .lib.analysis_profiles import generate_rois_from_image
from .lib.plotting import show_npy_2d, plot_photon_distribution, plot_filter_effects
from .lib.thresholding import estimate_threshold_otsu_from_frames, split_images_by_threshold
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

# ROI フィッティング前の平滑化（ベースライン除去は行わず、移動平均のみ）
# 窓幅は奇数推奨。データによって 5〜51 程度で調整してください。
MOVING_AVG_WINDOW_Y = 21  # 垂直プロファイル用（行方向）
MOVING_AVG_WINDOW_X = 21  # 水平プロファイル用（列方向）

# 画面端のピークをノイズとして無視するためのマージン設定
# 配列長の一定割合か、ピクセル固定値の大きい方を採用
EDGE_IGNORE_RATIO = 0.02   # 全長の2%
EDGE_IGNORE_MIN_PIX = 10   # 最低でも10px

# 実験のログ出力用のtxt
# 露光時間、ROI情報、撮影枚数などを記録する
# npy以外のプロットに必要なデータを格納する（フィッティングなど）
# プロットは時間がかかるから後からでもできるようにする
# テキストよりも最適なフォーマットがある？


def log_experiment_details(log_path: str, expose_time: float, rois: list, n_frames: int):
    pass


def build_session_dirs(timestamp: str, base_parent: Optional[str] = None) -> Dict[str, str]:
    # Use shared io_session implementation
    return io_build_session_dirs(timestamp, base_parent)


def list_session_frame_paths(raw_dir: str) -> list:
    # Timestamp-independent listing via shared lib
    return io_list_session_frame_paths(raw_dir)


def load_session_frames(raw_dir: str) -> list:
    # Use shared robust loader
    return io_load_session_frames(raw_dir)


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
    # Deprecated here; use lib.image_ops.apply_roi_npy directly
    return ops_apply_roi_npy(npy_path, roi)


# 汎用的な1Dプロット関数。
# plotting moved to lib.plotting


# plotting moved to lib.plotting


# 1D の多峰ローレンツ和（最後の引数はオフセット）
# fitting moved to lib.analysis_profiles


# fitting moved to lib.analysis_profiles


def _moving_average_1d(profile: np.ndarray, window: int) -> np.ndarray:
    """1D 移動平均フィルタ。エッジは端値でパディングして長さを保ちます。
    window<=1 の場合は元データを float にして返します。
    """
    prof = np.asarray(profile, dtype=float)
    w = int(window)
    if w <= 1:
        return prof.astype(float, copy=False)
    # 奇数にそろえる（偶数の場合は+1）
    if w % 2 == 0:
        w += 1
    pad_left = w // 2
    pad_right = w - 1 - pad_left
    prof_pad = np.pad(prof, (pad_left, pad_right), mode='edge')
    kernel = np.ones(w, dtype=float) / float(w)
    smoothed = np.convolve(prof_pad, kernel, mode='valid')
    return smoothed


def _edge_margin(n: int) -> int:
    """配列長 n に対する端マージン（ピーク無視）を返す。"""
    return max(int(round(n * float(EDGE_IGNORE_RATIO))), int(EDGE_IGNORE_MIN_PIX))


# 2D画像から垂直プロファイルを抽出し、ローレンツフィッティングを実行する。
# fitting moved to lib.analysis_profiles


#  2D画像から水平プロファイルを抽出し、多峰ローレンツフィッティングを実行する。
# fitting moved to lib.analysis_profiles


# fitting moved to lib.analysis_profiles


# plotting moved to lib.plotting


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


# generate_rois_from_image is provided by lib.analysis_profiles


# plotting moved to lib.plotting


# plotting moved to lib.plotting


def determine_ion_state(img: np.ndarray, threshold: float) -> bool:
    # y軸方向に積分して1次元の光子数分布をl取得
    photon_counts = img.sum(axis=0)

    # 閾値以下のデータ点の数を取得
    dark_count = np.sum(photon_counts <= threshold)

    # 全データ点に対する、閾値以下のデータ点の割合を計算
    dark_ratio = dark_count / len(photon_counts)

    # 暗状態のデータが半分以上なら暗状態(False)、そうでなければ明状態(True)
    return dark_ratio <= 0.5


# thresholding moved to lib.thresholding


# thresholding moved to lib.thresholding


def extract_rois_from_image(img: np.ndarray, rois: list) -> list:
    # Delegate to shared image ops
    return ops_extract_rois_from_image(img, rois)


# verify_ion_count_consistency removed; use analysis results directly where needed


# integrate_photon_countsをイオンの個数だけ繰り返し、さらに周波数ごとに繰り返して、周波数と励起成功確率の2D配列を作成する関数
def create_frequency_excitation_probability_matrix(spectrum_data, ion_counts):
    pass  # 実装はここに記述してください


# 周波数と励起成功確率の2Dndarrayをプロットする関数。ローレンチアンフィットも行う。中心周波数と線幅も求める。
# frequency plotting with fit removed for now; implement in lib.plotting if needed


def main():
    # ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 固定のセッションIDを使う場合は文字列で指定してください
    ts = "20251102_104353"
    s = build_session_dirs(ts)

    # 同一セッション内での保存ファイル連番をメモリで管理（未使用のため省略）

    # ガウシアンフィッティングして大きなROI決定してもいいかも

    # 明状態の取得
    # saved, next_idx = get_n_frames_from_buffer(
    #     expose_time=EXPOSE_TIME,
    #     roi=ROUGH_ROI,
    #     session_root=s["root"],
    #     start_index=next_idx,
    # )
    saved = np.load(
        "C:\\Users\\karishio\\Desktop\\single_ion_control\\src\\camera\\output\\20251102_104353\\raw-data\\2025_1102_105441data_000040.npy")
    # トリミング範囲決定（フィット可視化 + フィルタ効果の可視化も実施）
    rois = generate_rois_from_image(saved, plot=True)
    # フィルタ前後の比較図を保存
    # フィルタ効果の比較（必要なら有効化）
    # plot_filter_effects(saved, save_dir=s["plots"], prefix=f"{ts}_filter")
    print(f"[ROI] determined {len(rois)} ROIs: {rois}")
    # 暗状態の取得
    # saved_2, next_idx = get_n_frames_from_buffer(
    #     expose_time=EXPOSE_TIME,
    #     roi=ROUGH_ROI,
    #     session_root=s["root"],
    #     start_index=next_idx,
    # )

    # 今回セッションの全フレームを関数で読み込む
    frames = load_session_frames(s["raw"]) 
    print(f"[load] loaded {len(frames)} frames from session '{ts}'")
    # 全体画像の可視化（軸つき）
    show_npy_2d(frames[0], title="Full frame",
                save_dir=s["plots"], save_name=f"{ts}_full_frame.png")
    print
    all_crops = []
    for f in frames:
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)
    print(f"[extract] extracted {len(all_crops)} ROIs from all frames")

    # 閾値を推定し、Light/Dark に分けてプロット
    print("[threshold] estimating threshold...")
    images_for_hist = all_crops if len(all_crops) > 0 else frames
    th = estimate_threshold_otsu_from_frames(images_for_hist)
    print(f"[threshold] estimated by Otsu: {th:.3f} (sum over ROI/frame)")
    light_imgs, dark_imgs = split_images_by_threshold(images_for_hist, th)
    print(
        f"[threshold] split -> light={len(light_imgs)}, dark={len(dark_imgs)}")

    plot_photon_distribution(light_images=light_imgs,
                             dark_images=dark_imgs,
                             save_dir=s["plots"],
                             save_name=f"{ts}_photon_dist_split.png")

# 光子数分布をプロットする関数（色々をまとめた関数）


if __name__ == "__main__":
    main()
