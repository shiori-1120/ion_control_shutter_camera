# シーケンスを組んでいる
# lineごとにタスクを作成しないとうまく制御できない
# TrueとFalseが逆かもしれない。アクティブローかも。

import nidaqmx
import time

class channel:
    camera = 0

SHUTTER_MAP = {
    channel.camera:       "Dev1/port0/line4",
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
        # 0.5 secごとにカメラトリガーのON/OFFを切り替え
        # 1 secごとに撮影
        tasks[channel.camera].write(True)
        time.sleep(1)

        tasks[channel.camera].write(False)
        time.sleep(1)


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