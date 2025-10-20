# シーケンスを組んでいる
# lineごとにタスクを作成しないとうまく制御できない
# TrueとFalseが逆かもしれない。アクティブローかも。

import nidaqmx
import time

class Shutter:
    NM_397 = 0
    NM_397_SIGMA = 1
    NM_729 = 2
    NM_854 = 3

SHUTTER_MAP = {
    Shutter.NM_397:       "Dev1/port0/line4",
    Shutter.NM_397_SIGMA: "Dev1/port0/line5",
    Shutter.NM_729:       "Dev1/port0/line6",
    Shutter.NM_854:       "Dev1/port0/line7",
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
        tasks[Shutter.NM_397].write(True)
        # delay about 0.6 ms
        tasks[Shutter.NM_397_SIGMA].write(True)
        # delay about 0.6 ms
        tasks[Shutter.NM_729].write(True)
        # delay about 0.6 ms
        tasks[Shutter.NM_854].write(True)
        # delay about 0.6 ms
        time.sleep(0.001)
        # delay about 0.6 ms
        
        tasks[Shutter.NM_397].write(False)
        # delay about 0.6 ms
        tasks[Shutter.NM_397_SIGMA].write(False)
        # delay about 0.6 ms
        tasks[Shutter.NM_729].write(False)
        # delay about 0.6 ms
        tasks[Shutter.NM_854].write(False)
        # delay about 0.6 ms
        time.sleep(0.001)
        # delay about 0.6 ms



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