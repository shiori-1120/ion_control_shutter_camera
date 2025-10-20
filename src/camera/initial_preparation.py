import numpy as np
import time
import os
import datetime
import matplotlib.pyplot as plt
from lib.controlDevice import Control_CONTEC
from lib.controlDevice import Control_qCMOScamera

# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

# トリガーを送るのではなく、エクスターナルモードで
# check camera connectionの中でOpen/SetParameters/StartCaptureを済ませる
def check_camera_connection():
    connected = Control_CONTEC.is_connected()
    if connected:
        print("接続済み")
    else:
        ok, info = Control_CONTEC.check_connection()
        print("接続詳細:", ok, info)


def decide_threshold_value():
    usb_io = Control_CONTEC()
    qCMOS = Control_qCMOScamera()

    exposureTime = 0.300

    h_pos = 2536
    v_pos = 1624
    # TODO: h_width, v_width値確認
    h_width = 1152
    v_width = 372

    qCMOS.OpenCamera_GetHandle()
    qCMOS.SetParameters(exposureTime, h_width, v_width, h_pos, v_pos)
    qCMOS.StartCapture()

    freq_list = list(range(220900, 221000, 10))
    print(freq_list)

    # 保存用ディレクトリ（今日の日付を使用）
    d = datetime.date.today().isoformat()
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    try:
        for i in freq_list:
            print(i)

            # qCMOSカメラにトリガを送る
            usb_io.SendTrigger()
            # 露光時間だけスリープする
            time.sleep(exposureTime)
            time.sleep(0.1)
            # qCMOSカメラから画像データを持ってくる
            aFrame, img = qCMOS.GetLastFrame()
            time.sleep(0.1)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(img)

            plt.savefig(f"{d}/{d}_{i}.png", dpi=100)
            plt.close()

            np.save(f"{d}/{d}_{i}.npy", img)

    finally:
        qCMOS.StopCapture()
        qCMOS.ReleaseBuf()
        time.sleep(0.01)
        qCMOS.CloseUninitCamera()

# ...existing code...
# 新しい閾値評価関数を追加


def evaluate_thresholds(bright_imgs, dark_imgs, nbins=1000, target_total_error=None):
    """
    bright_imgs, dark_imgs: list of ndarray (ROI済み推奨)
    nbins: 閾値スイープ分解能
    target_total_error: 目標の (FP+FN) 合計率 (例 0.01) を満たす閾値を探す（なければ None）
    戻り値: dict with
      - 'best_min_total': {'threshold','fp_rate','fn_rate','total_error'}
      - 'best_target_error' (存在すれば同様の dict)
      - 'thresholds','fp_rates','fn_rates','total_rates' (arrays)
    """
    import numpy as _np

    bright = _np.concatenate([img.ravel() for img in bright_imgs])
    dark = _np.concatenate([img.ravel() for img in dark_imgs])

    lo = float(min(bright.min(), dark.min()))
    hi = float(max(bright.max(), dark.max()))
    if lo == hi:
        thresholds = _np.array([lo])
    else:
        thresholds = _np.linspace(lo, hi, nbins)

    fp_rates = _np.empty(len(thresholds))
    fn_rates = _np.empty(len(thresholds))
    total_rates = _np.empty(len(thresholds))

    for i, thr in enumerate(thresholds):
        fn = _np.mean(bright < thr)       # 明 を 0 と判定する割合
        fp = _np.mean(dark >= thr)       # 暗 を 1 と判定する割合
        fp_rates[i] = fp
        fn_rates[i] = fn
        total_rates[i] = fp + fn

    # 最小合計誤判定率
    idx_min = int(_np.argmin(total_rates))
    best_min = {'threshold': float(thresholds[idx_min]),
                'fp_rate': float(fp_rates[idx_min]),
                'fn_rate': float(fn_rates[idx_min]),
                'total_error': float(total_rates[idx_min])}

    best_target = None
    if target_total_error is not None:
        # 目標を満たす閾値のうち、合計誤判定率が最小のものを選ぶ
        mask = total_rates <= float(target_total_error)
        if mask.any():
            idxs = _np.where(mask)[0]
            idx_pick = int(idxs[_np.argmin(total_rates[idxs])])
            best_target = {'threshold': float(thresholds[idx_pick]),
                           'fp_rate': float(fp_rates[idx_pick]),
                           'fn_rate': float(fn_rates[idx_pick]),
                           'total_error': float(total_rates[idx_pick])}

    return {
        'best_min_total': best_min,
        'best_target_error': best_target,
        'thresholds': thresholds,
        'fp_rates': fp_rates,
        'fn_rates': fn_rates,
        'total_rates': total_rates
    }


def collect_threshold_calibration(qcmos: Control_qCMOScamera,
                                  usb_io: Control_CONTEC,
                                  n_per_state: int = 50,
                                  roi: tuple = None,
                                  marker_read: callable = None,
                                  target_total_error: float = None,
                                  out_path: str = "threshold_calibration.npz"):
    """
    キャリブレーションを行って閾値を保存する。
    - qcmos, usb_io: インスタンス（すでに Open/SetParameters/StartCapture 済みが望ましい）
    - n_per_state: 明・暗それぞれのフレーム枚数目標
    - roi: (y0, y1, x0, x1) で ROI 指定。None で全体。
    - marker_read: 各フレームを撮る直前に呼ぶコールバック。True=bright, False=dark を返す。
                   省略時は手動プロンプトでラベルを付ける。
    戻り値: dict を返し、ファイルにも保存する。
    """
    bright_imgs = []
    dark_imgs = []

    def default_marker():
        ans = input("現在レーザーを明にしていますか？ (y/n): ").strip().lower()
        return ans.startswith('y')

    if marker_read is None:
        marker_read = default_marker

    total_needed = n_per_state * 2
    attempt = 0
    max_attempts = total_needed * 3

    while (len(bright_imgs) < n_per_state or len(dark_imgs) < n_per_state) and attempt < max_attempts:
        attempt += 1
        # 外部システム同期があるならここで待機／確認する（marker_readに任せる）
        is_bright = marker_read()

        # カメラにトリガ（必要な場合）
        try:
            usb_io.SendTrigger()
        except Exception:
            # SendTrigger を使わない運用でも動くように例外吸収
            pass

        # 露光時間と余裕を置いて取得
        time.sleep(0.05)
        aFrame, img = qcmos.GetLastFrame()
        # img は numpy ndarray
        if roi is not None:
            y0, y1, x0, x1 = roi
            img = img[y0:y1, x0:x1]

        if is_bright:
            if len(bright_imgs) < n_per_state:
                bright_imgs.append(img.astype(np.float64))
        else:
            if len(dark_imgs) < n_per_state:
                dark_imgs.append(img.astype(np.float64))

        # 小休止
        time.sleep(0.02)

    if len(bright_imgs) == 0 or len(dark_imgs) == 0:
        raise RuntimeError(
            "bright または dark のサンプルが不足しています。接続と marker を確認してください。")

    bright_mean = np.mean(np.stack(bright_imgs), axis=0)
    bright_std = np.std(np.stack(bright_imgs), axis=0)
    dark_mean = np.mean(np.stack(dark_imgs), axis=0)
    dark_std = np.std(np.stack(dark_imgs), axis=0)

    # 閾値案（ピクセル毎）
    thresh_pixel = (bright_mean + dark_mean) / 2.0
    # ROI 単位の閾値（全体統計）
    thresh_global = (bright_mean.mean() + dark_mean.mean()) / 2.0
    thresh_dark_std = dark_mean + 3.0 * dark_std  # conservative

    result = {
        'bright_mean': bright_mean,
        'bright_std': bright_std,
        'dark_mean': dark_mean,
        'dark_std': dark_std,
        'thresh_pixel': thresh_pixel,
        'thresh_global': float(thresh_global),
        'thresh_dark_std': thresh_dark_std,
        'n_bright': len(bright_imgs),
        'n_dark': len(dark_imgs),
        'roi': roi
    }

    # 閾値評価（グローバル閾値選定用に ROI 内画素をまとめて評価）
    # target_total_error を指定すればその閾値も探す（例: 0.01 = 1%）
    eval_res = evaluate_thresholds(
        bright_imgs, dark_imgs, nbins=2000, target_total_error=target_total_error)

    # 結果を result に追加して保存
    result['threshold_scan'] = {
        'best_min_total': eval_res['best_min_total'],
        'best_target_error': eval_res['best_target_error'],
        'nbins': len(eval_res['thresholds'])
    }

    # 詳細配列はファイルに保存（サイズ大きければ省略可）
    np.savez_compressed(out_path.replace('.npz', '_scan.npz'),
                        thresholds=eval_res['thresholds'],
                        fp_rates=eval_res['fp_rates'],
                        fn_rates=eval_res['fn_rates'],
                        total_rates=eval_res['total_rates'])

    # コンソール出力（要点）
    print("閾値スキャン完了:")
    print(" 最小合計誤判定率閾値 :", result['threshold_scan']['best_min_total'])
    if result['threshold_scan']['best_target_error'] is not None:
        print(" 目標誤判定率を満たす閾値 :", result['threshold_scan']['best_target_error'])
    else:
        if target_total_error is not None:
            print(f" 目標誤判定率 {target_total_error} を満たす閾値は見つかりませんでした。")

    np.savez_compressed(out_path, **result)
    print(f"閾値と評価結果を保存しました: {out_path}")
    return result

# 使い方例:
# usb_io = Control_CONTEC()
# cam = Control_qCMOScamera()
# cam.OpenCamera_GetHandle(); cam.SetParameters(...); cam.StartCapture()
# def marker_from_contec():
#     # 実装例: CONTEC のデジタル入力を読む関数を作り、True/False を返す
#     return usb_io.read_marker_pin()
# res = collect_threshold_calibration(cam, usb_io, n_per_state=30, roi=(100,200,150,250), marker_read=marker_from_contec)
# cam.StopCapture(); cam.ReleaseBuf(); cam.CloseUninitCamera()
# ...existing code...


# TODO: check_camera_settings()
# 画像がとれているか確認
# 画像を表示させる。イオンの個数を確認

# TODO: roi_settings()
# 全体のトリミング範囲を設定（画像処理に使う範囲）
# イオンの個数とそれぞれに対応するトリミング範囲を決定（横長にとったものを分割？？）
