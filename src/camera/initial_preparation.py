import numpy as np
import time
import os
import datetime
import json
import matplotlib.pyplot as plt
from lib.controlDevice import Control_CONTEC
from lib.controlDevice import Control_qCMOScamera

expose_time = 0.100

def get_frame_from_buffer(camera, timeout=2.0, poll=0.01):
    """
    バッファからフレームを取得する汎用関数。
    """
    # TODO: 
    # output ディレクトリにファイル名を指定（タイムスタンプ付き）
    # カメラの取得
    # getHandle
    # startCapture
    # np.save





# TODO: miniforgeのpathを通して、ターミナルの規定値に登録できるようにする

# cameraの接続確認はoneshotの関数内で撮影で行う

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

# 保存された画像から行うようにする
def collect_threshold_calibration(qcmos: Control_qCMOScamera,
                                                                    usb_io: Control_CONTEC,
                                                                    n_per_state: int = 50,
                                                                    roi: tuple = None,
                                                                    marker_read: callable = None,
                                                                    target_total_error: float = None,
                                                                    out_path: str = "threshold_calibration.npz",
                                                                    send_trigger: bool = False):
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

    # 試行回数制限
    total_needed = n_per_state * 2
    attempt = 0
    max_attempts = total_needed * 3

    # dataset.json から露光時間を参照してバッファ取得の待ち時間を決める
    try:
        with open('./dataset.json') as f:
            _json = json.load(f)
            expose_time = float(_json.get('qCMOS.expose-time', 0.05))
    except Exception:
        expose_time = 0.05

    while (len(bright_imgs) < n_per_state or len(dark_imgs) < n_per_state) and attempt < max_attempts:
        attempt += 1
        # 外部システム同期があるならここで待機／確認する（marker_readに任せる）
        is_bright = marker_read()

        # (外部トリガ運用) デフォルトではトリガは送らない
        if send_trigger:
            try:
                usb_io.SendTrigger()
            except Exception:
                # SendTrigger が使えない環境でも続行
                pass

        # 露光時間分待ってから、バッファから取得する
        time.sleep(expose_time)
        # 少し余裕を与える
        time.sleep(0.01)
        img = get_frame_from_buffer(qcmos, timeout=max(1.0, expose_time + 1.0))
        if img is None:
            # 取得できない場合はログを残して次へ（またはカウントしない）
            print(f"Warning: フレーム取得失敗 attempt={attempt}")
            # 小休止して再試行ループへ
            time.sleep(0.02)
            continue
        # img は numpy ndarray
        # ROI と型変換（1 回だけ実行）
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

# 画像を

# TODO: check_camera_settings()
# 画像がとれているか確認
# 画像を表示させる。イオンの個数を確認

# TODO: roi_settings()
# 全体のトリミング範囲を設定（画像処理に使う範囲）
# イオンの個数とそれぞれに対応するトリミング範囲を決定（横長にとったものを分割？？）
