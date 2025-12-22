# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

import numpy as np
import time
import os
import datetime
from typing import Optional

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .lib.io_session import build_session_dirs, load_session_frames
from .lib.image_ops import extract_rois_from_image
from .lib.thresholding import (
    normalize_count,
    quick_threshold_from_samples,
    classify_hysteresis,
)


EXPOSE_TIME = 0.050
WAIT_MARGIN_SEC = 0.02
IDLE_TIMEOUT_SEC = 10.0
ROUGH_ROI = [600, 100, 2976, 984]

# 実験のログ出力用のtxt
# 露光時間、ROI情報、撮影枚数などを記録する
# npy以外のプロットに必要なデータを格納する（フィッティングなど）
# プロットは時間がかかるから後からでもできるようにする
# テキストよりも最適なフォーマットがある？


def log_experiment_details(log_path: str, expose_time: float, rois: list, n_frames: int):
    pass


def get_n_frames_from_buffer(
    expose_time: float = 0.100,
    roi: Optional[list] = None,
    session_root: Optional[str] = None,
    start_index: Optional[int] = None,
) -> tuple[int, int]:

    # Camera environment is required for capture.
    from .lib.controlDevice import Control_qCMOScamera

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


def determine_ion_state(img: np.ndarray, threshold: float) -> bool:
    # y軸方向に積分して1次元の光子数分布をl取得
    photon_counts = img.sum(axis=0)

    # 閾値以下のデータ点の数を取得
    dark_count = np.sum(photon_counts <= threshold)

    # 全データ点に対する、閾値以下のデータ点の割合を計算
    dark_ratio = dark_count / len(photon_counts)

    # 暗状態のデータが半分以上なら暗状態(False)、そうでなければ明状態(True)
    return dark_ratio <= 0.5


def verify_ion_count_consistency(img: np.ndarray, ion_positions) -> bool:
    try:
        if img is None:
            return False
        expected_count = len(ion_positions or [])
        # Lazy import to avoid SciPy dependency unless used
        from .lib.analysis_profiles import fit_horizontal_profile
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
    from .lib.analysis_profiles import lorentz
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


############################
# Ion state thresholding utils are now in lib.thresholding
############################


def _self_test():
    """
    ダミーデータで正規化・閾値推定・ヒステリシス判定の自己検証を行う。
    - 露光時間のみで正規化するため、露光違いでも S_norm が揃うことを確認。
    - 明暗が混在するサンプルから Otsu で閾値を推定し、ヒステリシスで判定。
    出力は print のみ（プロット無し）。
    """
    rng = np.random.default_rng(42)

    # 画像サイズとROI設定（固定）
    H, W = 80, 160
    roi = (40, 16, 60, 32)  # (x_width, y_width, x_start, y_start)
    bg_roi = (40, 16, 10, 10)

    # 露光を交互に変える（0.05s と 0.1s）
    N = 40
    exposures = np.array(
        [0.05 if (i % 2 == 0) else 0.10 for i in range(N)], dtype=float)

    # 明/暗フラグ（前半は暗多め、後半は明多め）
    is_bright = np.array([(i % 4 in (2, 3)) for i in range(N)], dtype=bool)

    # 背景平均とノイズ（ADU）
    bg_mean_true = 100.0
    read_noise_sigma = 3.0

    # 明状態の信号（ROI合計の1秒あたりターゲット）
    # 正規化は exposure のみなので、平均的な S_norm ≈ signal_per_sec * Npx になる点に注意
    xw, yw, xs, ys = roi
    signal_per_sec = 8.0  # 1pxでなくROI合計/秒の強さではなく、後で* Npx 相当の分を織り込む

    imgs = []
    for i in range(N):
        img = rng.normal(loc=bg_mean_true, scale=read_noise_sigma,
                         size=(H, W)).astype(np.float64)
        if is_bright[i]:
            # ROI内に信号を加算：合計が exposure * signal_per_sec * Npx 付近になるように
            signal = signal_per_sec * exposures[i]
            img[ys:ys+yw, xs:xs+xw] += signal
        imgs.append(img)

    # 先頭10枚で初期閾値推定
    info_list = [normalize_count(
        imgs[i], roi, bg_roi=bg_roi, exposure_s=float(exposures[i])) for i in range(N)]
    samples = [d["S_norm"] for d in info_list[:10]]
    q = quick_threshold_from_samples(samples, provisional_tau=None)

    # 全フレームにヒステリシス判定
    state = None
    results = []
    for i in range(N):
        S = float(info_list[i]["S_norm"])
        state = classify_hysteresis(
            S, prev_state_bright=state, tau_on=q["tau_on"], tau_off=q["tau_off"])
        results.append(bool(state))

    # 検証サマリ
    s_dark = [info_list[i]["S_norm"] for i in range(N) if not is_bright[i]]
    s_bright = [info_list[i]["S_norm"] for i in range(N) if is_bright[i]]
    print("[SELFTEST] samples(first 10) ->", samples)
    print("[SELFTEST] tau/tau_on/tau_off:",
          {k: q[k] for k in ("tau", "tau_on", "tau_off", "mode")})
    print("[SELFTEST] mean S_norm dark/bright:",
          np.mean(s_dark), np.mean(s_bright))
    acc = float(np.mean(results == is_bright))
    print(
        f"[SELFTEST] hysteresis accuracy vs ground-truth: {acc*100:.1f}% (rough check)")


def main():
    # 環境変数 SELFTEST=1 のときはダミーデータで自己検証を実行
    if os.environ.get("SELFTEST", "0") == "1":
        _self_test()
        return

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    s = build_session_dirs(ts)

    # 同一セッション内での保存ファイル連番をメモリで管理
    next_idx = 1

    # ガウシアンフィッティングして大きなROI決定してもいいかも

    # 明状態の取得
    saved, next_idx = get_n_frames_from_buffer(
        expose_time=EXPOSE_TIME,
        roi=ROUGH_ROI,
        session_root=s["root"],
        start_index=next_idx,
    )

    # 暗状態も撮影した後にフレームを読み込み、最初のフレームからROIを決定

    # 暗状態の取得
    saved_2, next_idx = get_n_frames_from_buffer(
        expose_time=EXPOSE_TIME,
        roi=ROUGH_ROI,
        session_root=s["root"],
        start_index=next_idx,
    )

    # 今回セッションの全フレームを関数で読み込む
    frames = load_session_frames(s["raw"]) 
    # トリミング範囲決定（先頭フレームから推定）
    from .lib.analysis_profiles import generate_rois_from_image
    rois = generate_rois_from_image(frames[0], plot=False)

    # 全体画像の可視化（軸つき）
    from .lib.plotting import show_npy_2d, plot_photon_distribution
    show_npy_2d(frames[0], title="Full frame",
                save_dir=s["plots"], save_name=f"{ts}_full_frame.png")

    # 決定したトリミング範囲でと
    all_crops = []
    for f in frames:
        crops = extract_rois_from_image(f, rois)
        all_crops.extend(crops)

    for i, crop in enumerate(all_crops):
        np.save(os.path.join(s["raw"], f"{ts}_roi{i:02d}.npy"), crop)

    # 閾値を決める関数
    images_for_hist = all_crops if len(all_crops) > 0 else [frames[0]]
    # 閾値とデータを入れると分別しながらプロットしてくれる
    # 閾値があるときは分別してプロットして、light, darkにリストを渡してそれぞれプロットしてもらう（どちらかがない場合もある）
    plot_photon_distribution(light_images=images_for_hist,
                             save_dir=s["plots"],
                             save_name=f"{ts}_photon_dist_light.png")

# 光子数分布をプロットする関数（色々をまとめた関数）


if __name__ == "__main__":
    main()
