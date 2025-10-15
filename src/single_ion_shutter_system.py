import numpy as np
import nidaqmx
from nidaqmx.constants import LineGrouping, AcquisitionType
import time


class Shutter:
    OFF = 0
    NM_397 = 1  # 2^0 -> port0/line0
    NM_397_SIGMA = 2  # 2^1 -> port0/line1
    NM_729 = 4  # 2^2 -> port0/line2
    NM_854 = 8  # 2^3 -> port0/line3


sequence = [
    (0.002, Shutter.NM_397 | Shutter.NM_397_SIGMA),  # Step 1: Cooling
    (0.002, Shutter.NM_397_SIGMA),                  # Step 2: Initialization
    (0.010, Shutter.NM_729),                        # Step 3: Excitation
    (0.004, Shutter.NM_397),                        # Step 4: Detection
    (0.010, Shutter.NM_397 | Shutter.NM_397_SIGMA | Shutter.NM_854),  # Step 5: Reset
]

# 100,000 Hz = 100 kHz -> 時間分解能は 1/100,000秒 = 10マイクロ秒 (µs)
sample_rate = 100000  # in Hz

print("波形データの生成を開始します...")
waveform_data = []
total_duration_s = 0

# --- シーケンス定義から、DACに書き込むための1次元配列を生成 ---
for duration_s, state in sequence:
    # このステップを維持すべきサンプル数（データ点数）を計算
    num_samples = int(duration_s * sample_rate)

    # 同じ状態(state)を num_samples 回繰り返したリストを、波形データの末尾に追加
    waveform_data.extend([state] * num_samples)
    total_duration_s += duration_s

# nidaqmxが扱うための形式 (NumPy配列, 32-bit符号なし整数) に変換
waveform_to_write = np.array(waveform_data, dtype=np.uint32)

print("--- 波形データ生成完了 ---")
print(f"時間分解能: {1e6 / sample_rate:.1f} µs")
print(f"総サンプル数: {len(waveform_to_write)} サンプル")
print(f"総シーケンス時間: {total_duration_s * 1000:.3f} ms")
print("--------------------------\n")


physical_channels = "Dev1/port0/line0:3"

try:
    with nidaqmx.Task() as task:
        # 1. デジタル出力チャンネルをタスクに追加
        task.do_channels.add_do_chan(
            physical_channels,
            line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
        )

        # 2. サンプルクロックのタイミングを設定（連続出力でバッファを循環）
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.CONTINUOUS,      # 繰り返し出力するためCONTINUOUSに戻す
            samps_per_chan=len(waveform_to_write)
        )

        # 出力バッファを明示的に確保（長い波形で必要）
        task.out_stream.output_buf_size = len(waveform_to_write)

        # 3. 生成した波形データを一度にバッファに書き込む（自動開始しない）
        writer = nidaqmx.stream_writers.DigitalSingleChannelWriter(
            task.out_stream)
        writer.write_many_sample_port_uint32(
            waveform_to_write, auto_start=False)

        # 4. タスクを開始してバッファを循環出力（Ctrl+Cで停止）
        task.start()
        print("連続出力を開始しました。Ctrl+Cで停止してください。")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("停止要求を受け取りました。タスクを停止します...")
        finally:
            task.stop()
            print("出力を停止しました。")

except nidaqmx.errors.DaqError as e:
    print(f"NI-DAQmxエラーが発生しました: {e}")