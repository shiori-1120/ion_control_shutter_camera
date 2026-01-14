"""
カメラ接続確認用スクリプトだわん！
カメラが正しく認識・初期化できるかだけをチェックする。
画像取得やROI処理は一切行わない。
"""
from src.camera.lib.ControlDevice import Control_qCMOScamera

if __name__ == "__main__":
    try:
        print("カメラ接続テスト開始だわん！")
        cam = Control_qCMOScamera(verbose=True)
        cam.OpenCamera_GetHandle()
        print("カメラ接続OKだわん！")
    except Exception as e:
        print(f"カメラ接続エラーだわん: {e}")
    finally:
        try:
            del cam
        except Exception:
            pass
