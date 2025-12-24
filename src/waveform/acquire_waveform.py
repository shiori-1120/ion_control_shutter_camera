import pyvisa
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.lib.utils.config import load_yaml, ConfigError, resolve_visa_from_profiles

DEFAULT_VISA = 'TCPIP::192.168.1.16::INSTR'

# 設定ファイルの読み込み（存在しない場合はデフォルト）
def load_waveform_config():
    # device settings are unified in src/config/device.yaml
    cfg_path = Path('src/config/device.yaml')
    try:
        cfg = load_yaml(cfg_path)
    except ConfigError:
        cfg = {
            'sampling_rate_hz': 100000,
            'channels': [1, 2, 3, 4],
        }
    return cfg


def acquire_waveform_from_scope(inst, channel, points=50000):
    inst.write(f"DATA:SOURCE CH{channel}")
    inst.write("DATA:ENCdg ASCii")
    inst.write("DATA:WIDTH 1")
    inst.write(f"DATA:STOP {points}")
    
    nr_pt = int(inst.query("WFMPRe:NR_PT?"))
    xzero = float(inst.query("WFMPRe:XZEro?"))
    xincr = float(inst.query("WFMPRe:XINcr?"))
    yzero = float(inst.query("WFMPRe:YZEro?"))
    ymult = float(inst.query("WFMPRe:YMUlt?"))
    yoff = float(inst.query("WFMPRe:YOff?"))

    time_list = np.arange(nr_pt) * xincr + xzero
    
    curve = inst.query("CURVE?")
    raw_voltage_list = np.fromstring(curve, dtype=int, sep=',')
    voltage_list = (raw_voltage_list - yoff) * ymult + yzero
    
    return time_list.tolist(), voltage_list.tolist()


def save_waveforms_to_csv(waveforms_data, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_waveform.csv"
    filepath = os.path.join(output_dir, filename)
    
    df = pd.DataFrame({'Time (s)': waveforms_data['CH1']['time']})
    for ch_name, data in waveforms_data.items():
        df[f"{ch_name}_Voltage (V)"] = data['voltage']
        
    df.to_csv(filepath, index=False)
    print(f"\n波形データを {filepath} に保存しました。")
    return filepath


def plot_waveforms(waveforms_data, title="Oscilloscope Waveforms"):
    num_channels = len(waveforms_data)
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 10), sharex=True)
    if num_channels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=16)
    
    time_data = list(waveforms_data.values())[0]['time']
    for i, (ch_name, data) in enumerate(waveforms_data.items()):
        axes[i].scatter(time_data, data['voltage'], label=ch_name)
        axes[i].set_ylabel("Voltage (V)")
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
        
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def main():
    cfg = load_waveform_config()
    channels = cfg.get('channels', [1, 2, 3, 4])
    # 固定出力先（YAML指定は廃止）
    output_dir = 'data/output/waveform'
    # Priority: profiles > single value > default (YAML中心運用)
    visa = (
        resolve_visa_from_profiles(cfg)
        or cfg.get('visa_resource')
        or DEFAULT_VISA
    )
    rm = pyvisa.ResourceManager()
    inst = None
    all_waveforms = {}
    print(f"接続を試みています: {visa}")
    inst = rm.open_resource(visa, timeout=20000)
    print(f"接続成功: {inst.query('*IDN?').strip()}")
    
    inst.write("ACQuire:STATE STOP")
    time.sleep(0.1)

    for ch in channels:
        time_data, voltage_data = acquire_waveform_from_scope(inst, ch)
        all_waveforms[f'CH{ch}'] = {'time': time_data, 'voltage': voltage_data}

    saved_filepath = save_waveforms_to_csv(all_waveforms, output_dir)
    plot_waveforms(all_waveforms, title=f"Acquired Waveforms ({os.path.basename(saved_filepath)})")
    
    inst.write("ACQuire:STATE RUN")
    inst.close()
    print("リソースを解放しました。")

if __name__ == "__main__":
    main()