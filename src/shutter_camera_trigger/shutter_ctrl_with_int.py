import nidaqmx
from nidaqmx.constants import LineGrouping
import time

# --- 設定 ---
# 2チャンネルを同時に制御 (port0のline0とline1)
PHYSICAL_CHANNELS = "Dev1/port0/line4:5"

# 状態を切り替える時間間隔（秒）
DELAY_S = 0.1  # 10ミリ秒

# --- メイン処理 ---
try:
    with nidaqmx.Task() as task:
        # 1. 複数チャンネルを1つのグループとしてタスクに追加
        task.do_channels.add_do_chan(
            PHYSICAL_CHANNELS,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
        )
        
        print(f"'{PHYSICAL_CHANNELS}' から2-bitのカウントアップ信号を出力します。")
        print("オシロスコープで2つの波形を観測してください。")
        print("Ctrl+Cで停止します。")
        
        task.start()
        
        try:
            # 2. 無限ループで 0 -> 1 -> 2 -> 3 を繰り返す
            while True:
                # a) 0 (00b) を書き込む -> line1=LOW, line0=LOW
                print("Writing 0 (00b)...")
                task.write([True, False])
                time.sleep(DELAY_S)
                
                # b) 1 (01b) を書き込む -> line1=LOW, line0=HIGH
                print("Writing 1 (01b)...")
                task.write([False, True])
                time.sleep(DELAY_S)

                # c) 2 (10b) を書き込む -> line1=HIGH, line0=LOW
                print("Writing 2 (10b)...")
                task.write([True, True])
                time.sleep(DELAY_S)

                # d) 3 (11b) を書き込む -> line1=HIGH, line0=HIGH
                print("Writing 3 (11b)...")
                task.write([False, False])
                time.sleep(DELAY_S)
                
        except KeyboardInterrupt:
            print("\n停止要求を受け取りました。")

        finally:
            # 3. 終了処理：安全のため、必ず出力を0にする
            print("出力を0 (OFF) にしてタスクを停止します。")
            if not task.is_task_done():
                task.write(0)
                task.stop()

except nidaqmx.errors.DacaqError as e:
    print(f"NI-DAQmxエラーが発生しました: {e}")