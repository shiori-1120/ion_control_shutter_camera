import pyvisa
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

VISA_RESOURCE_STRING = 'USB0::0x0699::0x03A2::C040073::INSTR'
# VISA_RESOURCE_STRING = 'USB0::0x0699::0x0363::C060618::INSTR'
CHANNELS_TO_MEASURE = [1, 2, 3, 4]
OUTPUT_DIRECTORY = "output"


def acquire_waveform_from_scope(inst, channel, points=100000):
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


def save_waveforms_to_csv(waveforms_data, output_dir="..\..\output"):
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
    if num_channels == 1: axes = [axes]
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
    rm = pyvisa.ResourceManager()
    inst = None
    all_waveforms = {}
    print(f"接続を試みています: {VISA_RESOURCE_STRING}")
    inst = rm.open_resource(VISA_RESOURCE_STRING, timeout=20000)
    print(f"接続成功: {inst.query('*IDN?').strip()}")
    
    inst.write("ACQuire:STATE STOP"); time.sleep(0.1)

    for ch in CHANNELS_TO_MEASURE:
        time_data, voltage_data = acquire_waveform_from_scope(inst, ch)
        all_waveforms[f'CH{ch}'] = {'time': time_data, 'voltage': voltage_data}

    saved_filepath = save_waveforms_to_csv(all_waveforms, OUTPUT_DIRECTORY)
    plot_waveforms(all_waveforms, title=f"Acquired Waveforms ({os.path.basename(saved_filepath)})")
    
    inst.write("ACQuire:STATE RUN")
    inst.close()
    print("リソースを解放しました。")

if __name__ == "__main__":
    main()