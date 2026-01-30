# USB-6002 の AO から 1 ms パルス(0V→5V→0V)を 1回出力
# 5 kS/s のハードウェアタイミング（FINITE）を使用
import numpy as np
import nidaqmx
import time
from nidaqmx.constants import AcquisitionType

DEVICE = "Dev1"
AO_CH  = f"{DEVICE}/ao0"  

RATE_HZ = 5000             # USB-6002 の最大（5 kS/s）
WIDTH_MS = 0.1            
V_HIGH = 3.3               # TTL相当（必要に応じて 3.3V等に変更）
V_LOW  = 0.0



# ON/OFF両方の1msパルスをクロックに同期して2回出力する波形を作成
n_pulse = max(1, int(WIDTH_MS/1000 * RATE_HZ))  # 1ms分のサンプル数
wave = np.concatenate([
    np.array([V_LOW], dtype=np.float64),
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
        np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
    np.full(n_pulse, V_HIGH, dtype=np.float64),   # 1ms ON
    np.full(n_pulse, V_LOW, dtype=np.float64),    # 1ms OFF
])


with nidaqmx.Task() as task:
    task.ao_channels.add_ao_voltage_chan(AO_CH, min_val=0.0, max_val=5.0)
    task.timing.cfg_samp_clk_timing(RATE_HZ, sample_mode=AcquisitionType.FINITE, samps_per_chan=len(wave))
    while True:
        # print("AO pulseを出力します...")
        task.write(wave, auto_start=False)
        task.start()
        task.wait_until_done(timeout=5.0)
        task.stop()
        # print(f"AO pulse出力完了。{sleep_interval_s}秒後に再度出力します。")
