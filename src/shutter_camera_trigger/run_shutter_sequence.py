from pathlib import Path
from datetime import datetime
import time
import nidaqmx

from .config.device_registry import resolve_output_root
from .sweep.session_config import write_manifest_json
# 固定の出力先ユーティリティ
def ensure_output_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def make_run_folder(base: Path) -> Path:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = base / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

OUTPUT_DIR = 'data/output/shutter'

def main():
    base = ensure_output_dir(str(resolve_output_root() / "shutter"))
    run_dir = make_run_folder(base)
    print(f"シャッターシーケンスの出力先: {run_dir}")
    # TODO: ここでユーザー定義の2進シーケンスを読み込み、実行する
    # 例: sequence = [0b0001, 0b0010, 0b0100, 0b1000]
    # 実機制御コードにシーケンスを渡して処理する
    time.sleep(0.1)
    # 実行結果のログやメタ情報を run_dir に保存する
    log_path = run_dir / "log.txt"
    log_path.write_text("sequence executed", encoding="utf-8")
    try:
        write_manifest_json(out_dir=run_dir, run_type="shutter", files={"log": log_path})
    except Exception:
        pass


if __name__ == '__main__':
    main()
# シーケンスを組んでいる
# lineごとにタスクを作成しないとうまく制御できない
# TrueとFalseが逆かもしれない。アクティブローかも。

# nidaqmx は上部でインポート済み

class Shutter:
    NM_397 = 0
    NM_397_SIGMA = 1
    CAMERA_TRIGGER = 2
    NM_729 = CAMERA_TRIGGER  # backward-compatible alias (was 729 shutter)
    NM_854 = 3

SHUTTER_MAP = {
    Shutter.NM_397: "Dev1/port1/line0",
    Shutter.NM_397_SIGMA: "Dev1/port1/line1",
    Shutter.CAMERA_TRIGGER: "Dev1/port1/line2",
    Shutter.NM_854: "Dev1/port1/line3",
}

tasks = {}
try:
    for shutter_key, channel_name in SHUTTER_MAP.items():
        task = nidaqmx.Task()
        task.do_channels.add_do_chan(channel_name)
        task.start()
        tasks[shutter_key] = task
        print(f"Task created for {channel_name}")

    print("\nシーケンスの直接実行を開始します。")
    print("Ctrl+Cで停止してください。")

    print("Initializing... All shutters OFF.")
    for task in tasks.values():
        task.write(False)
    
    while True:
        # 397:ON, 397_SIGMA:ON, Camera trigger:OFF, 854:ON
        tasks[Shutter.NM_397].write(True)
        tasks[Shutter.NM_397_SIGMA].write(True)
        tasks[Shutter.NM_854].write(True)
        time.sleep(0.002)

        # 397:OFF, 397_SIGMA:ON, Camera trigger:OFF, 854:OFF
        tasks[Shutter.NM_397].write(False)
        tasks[Shutter.NM_854].write(False)
        time.sleep(0.002)

        # 397:OFF, 397_SIGMA:OFF, Camera trigger:ON, 854:OFF
        tasks[Shutter.NM_397_SIGMA].write(False)
        tasks[Shutter.CAMERA_TRIGGER].write(True)
        time.sleep(0.010)

        # 397:ON, 397_SIGMA:OFF, Camera trigger:OFF, 854:OFF
        tasks[Shutter.CAMERA_TRIGGER].write(False)
        tasks[Shutter.NM_397].write(True)
        time.sleep(0.004)

        # 397:ON, 397_SIGMA:ON, Camera trigger:OFF, 854:ON
        tasks[Shutter.NM_397_SIGMA].write(True)
        tasks[Shutter.NM_854].write(True)
        time.sleep(0.010)

except KeyboardInterrupt:
    print("\n停止要求を受け取りました。タスクをクリーンアップします...")

finally:
    if tasks:
        print("全てのシャッターをOFFにして、タスクを閉じます。")
        for task in tasks.values():
            try:
                task.write(False)
                task.close()
            except Exception as e:
                print(f"Error closing a task: {e}")
