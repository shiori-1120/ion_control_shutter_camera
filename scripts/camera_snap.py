"""
ROIやしきい値処理は一切行わず、画像取得だけを行う。
"""
from src.camera.lib.ControlDevice import Control_qCMOScamera
import numpy as np
from pathlib import Path
import sys

if __name__ == "__main__":
    save_path = sys.argv[1] if len(sys.argv) > 1 else "camera_snap.npy"
    try:
        print("カメラスナップ開始")
        cam = Control_qCMOScamera(verbose=True)
        cam.OpenCamera_GetHandle()
        cam.SetParameters(0.1)  # 露光時間0.1秒（適宜調整）
        cam.StartCapture()
        ok, err = cam.wait_for_frame_ready(2.0)
        if not ok:
            raise RuntimeError(f"frame not ready: {err}")
        _, frame = cam.GetLastFrame()
        arr = np.asarray(frame)
        np.save(save_path, arr)
        print(f"画像を{save_path}に保存 shape={arr.shape}")
    except Exception as e:
        print(f"カメラスナップエラー: {e}")
    finally:
        try:
            del cam
        except Exception:
            pass
