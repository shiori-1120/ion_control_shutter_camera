# USB-6002 の AO から 1 ms パルス(0V→5V→0V)を 1回出力
# 5 kS/s のハードウェアタイミング（FINITE）を使用
import numpy as np
import nidaqmx
import time
from nidaqmx.constants import AcquisitionType

DEVICE = "Dev1"
AO_CH  = f"{DEVICE}/ao0"   # ao1 に変えてもOK

RATE_HZ = 5000             # USB-6002 の最大（5 kS/s）
WIDTH_MS = 1.0             # パルス幅 1 ms
V_HIGH = 5.0               # TTL相当（必要に応じて 3.3V等に変更）
V_LOW  = 0.0

# 前後に 1 サンプルずつ LOW を入れて立ち上がり/立下りを明確に
n_high = max(1, int(WIDTH_MS/1000 * RATE_HZ))  # 1ms → 5 サンプル
wave = np.concatenate([
    np.array([V_LOW], dtype=np.float64),
    np.full(n_high, V_HIGH, dtype=np.float64),
    np.array([V_LOW], dtype=np.float64),
])

REPEAT_INTERVAL_S = 0.00100  # 1msごとに発生
with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan(AO_CH, min_val=0.0, max_val=5.0)
    task.timing.cfg_samp_clk_timing(RATE_HZ, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(wave))
    while True:
        task.write(wave, auto_start=False)
        task.start()
        task.wait_until_done(timeout=5.0)
        task.stop()
        time.sleep(REPEAT_INTERVAL_S)
