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

    print("Initializing... All shutters OFF.")
    for task in tasks.values():
        task.write(False)
    
    count = 0
    while count < 1:
    # while True:
        tasks[channel.camera].write(True)
        time.sleep(0.010)

        tasks[channel.camera].write(False)
        time.sleep(0.5)
        count += 1


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