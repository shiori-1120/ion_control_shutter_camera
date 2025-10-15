import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping
import time

class Shutter:
    OFF = 0
    NM_397 = 1          # 2^0 -> port0/line4
    NM_397_SIGMA = 2    # 2^1 -> port0/line5
    NM_729 = 4          # 2^2 -> port0/line6
    NM_854 = 8          # 2^3 -> port0/line7

sequence = [
    (0.002, Shutter.NM_397 | Shutter.NM_397_SIGMA),  # Step 1: Cooling
    (0.002, Shutter.NM_397_SIGMA),                   # Step 2: Initialization
    (0.010, Shutter.NM_729),                        # Step 3: Excitation
    (0.004, Shutter.NM_397),                        # Step 4: Detection
    (0.010, Shutter.NM_397 | Shutter.NM_397_SIGMA | Shutter.NM_854),  # Step 5: Reset
]

physical_channels = "Dev1/port0/line4:7"

try:
    with nidaqmx.Task() as task:
        task.do_channels.add_do_chan(
            physical_channels,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
        )

        print("ソフトウェアタイミングでシーケンス出力を開始します。")
        print("Ctrl+Cで停止してください。")
        
        task.start()

        try:
            while True:
                for duration_s, state in sequence:
                    task.write(state)
                    
                    time.sleep(duration_s)
                    
        except KeyboardInterrupt:
            print("\n停止要求を受け取りました。タスクをクリーンアップします...")
            
        finally:
            print("全てのシャッターをOFFにします。")
            task.write(Shutter.OFF)
            task.stop()
            print("タスクを停止しました。")

except nidaqmx.errors.DaqError as e:
    print(f"NI-DAQmxエラーが発生しました: {e}")