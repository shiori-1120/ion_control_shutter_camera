import pyvisa
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import time

def save_multi_channel_to_csv(waveforms_data, output_dir="..\output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"waveform_4ch_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    header = ["Time (s)"]
    header.extend([f"{ch}_Voltage (V)" for ch in waveforms_data.keys()])

    first_ch_name = list(waveforms_data.keys())[0]
    time_data = waveforms_data[first_ch_name]['time']

    rows_to_write = []
    for i in range(len(time_data)):
        row = [time_data[i]]
        for ch_name in waveforms_data.keys():
            row.append(waveforms_data[ch_name]['voltage'][i])
        rows_to_write.append(row)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_to_write)

    print(f"4チャンネルのデータを {filepath} に保存しました。")
    return filepath


def acquire_waveform(inst, channel, points=100000):
    print(f"Acquiring waveform from CH{channel}...")
    inst.write(f"DATA:SOURCE CH{channel}")
    inst.write("DATA:ENCdg ASCii")
    inst.write("DATA:WIDTH 1")
    inst.write("DATA:START 1")
    inst.write(f"DATA:STOP {points}")

    nr_pt = int(inst.query("WFMPRe:NR_PT?"))
    xzero = float(inst.query("WFMPRe:XZEro?"))
    xincr = float(inst.query("WFMPRe:XINcr?"))
    yzero = float(inst.query("WFMPRe:YZEro?"))
    ymult = float(inst.query("WFMPRe:YMUlt?"))
    yoff = float(inst.query("WFMPRe:YOff?"))

    time_list = [xzero + xincr * i for i in range(nr_pt)]

    curve = inst.query("CURVE?")
    raw_voltage_list = [int(v) for v in curve.split(",")]
    voltage_list = [(v - yoff) * ymult + yzero for v in raw_voltage_list]

    return time_list, voltage_list


def plot_stacked_from_csv(filepath):
    data = {}
    header = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for col_name in header:
            data[col_name] = []

        for row in reader:
            for i, value in enumerate(row):
                col_name = header[i]
                data[col_name].append(float(value))

    time_col_name = header[0]
    voltage_channels = header[1:]
    num_channels = len(voltage_channels)

    fig, axes = plt.subplots(num_channels, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(
        "Oscilloscope Waveforms (Simultaneous Acquisition)", fontsize=16)

    for i, ch_name in enumerate(voltage_channels):
        ax = axes[i]
        ax.plot(data[time_col_name], data[ch_name], label=ch_name)
        ax.set_ylabel("Voltage (V)")
        ax.grid(True)
        ax.legend(loc='upper right')

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

    print(f"{filepath} から4チャンネルのグラフを縦に並べて描画しました。")


def main():

    rm = pyvisa.ResourceManager()
    inst = None
    channels_to_measure = [1, 2, 3, 4]

    try:
        inst = rm.open_resource(
            'USB0::0x0699::0x03A2::C040073::INSTR', timeout=20000)
        print(f"接続成功: {inst.query('*IDN?').strip()}")

        print("即時取得モード: オシロを一時停止して内部メモリを読み出します。")
        inst.write("ACQuire:STATE STOP")
        time.sleep(0.1)
        print("内部メモリを読み出します。")

        all_waveforms = {}
        print("内部メモリから各チャンネルのデータを転送します...")
        for ch in channels_to_measure:
            print(f"Reading CH{ch} data...")
            time_data, voltage_data = acquire_waveform(inst, ch)
            all_waveforms[f'CH{ch}'] = {
                'time': time_data, 'voltage': voltage_data}

        if all_waveforms:
            saved_filepath = save_multi_channel_to_csv(all_waveforms)

            if saved_filepath:
                plot_stacked_from_csv(saved_filepath)

    except pyvisa.errors.VisaIOError as e:
        print("VISA I/O Error:", e)
    finally:
        if inst:
            inst.write("ACQuire:STOPAfter RUNSTop")
            inst.write("ACQuire:STATE RUN")
            inst.close()
        print("リソースを解放しました。")


if __name__ == "__main__":
    main()
