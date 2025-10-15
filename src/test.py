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
        # 397:ON, 397_SIGMA:ON, 729:OFF, 854:ON
        tasks[Shutter.NM_397].write(True)
        tasks[Shutter.NM_397_SIGMA].write(True)
        tasks[Shutter.NM_854].write(True)
        time.sleep(0.002)

        # 397:OFF, 397_SIGMA:ON, 729:OFF, 854:OFF
        tasks[Shutter.NM_397].write(False)
        tasks[Shutter.NM_854].write(False)
        time.sleep(0.002)

        # 397:OFF, 397_SIGMA:OFF, 729:ON, 854:OFF
        tasks[Shutter.NM_397_SIGMA].write(False)
        tasks[Shutter.NM_729].write(True)
        time.sleep(0.010)

        # 397:ON, 397_SIGMA:OFF, 729:OFF, 854:OFF
        tasks[Shutter.NM_729].write(False)
        tasks[Shutter.NM_397].write(True)
        time.sleep(0.004)

        # 397:ON, 397_SIGMA:ON, 729:OFF, 854:ON
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