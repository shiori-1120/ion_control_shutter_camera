import pyvisa
import os
import csv
from datetime import datetime
import numpy as np

# TODO: ASCIIのファイルをバイナリに変更する
def save_to_csv(time_data, voltage_data, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_waveform.csv"
    filepath = os.path.join(output_dir, filename)

    rows = zip(time_data, voltage_data)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Voltage (V)"])
        writer.writerows(rows)

    print(f"データを {filepath} に保存しました。")


def get_waveform(inst, channel, points=10000):
    inst.write(f"DATA:SOURCE CH{channel}")
    inst.write("DATA:ENCdg RIBinary")
    inst.write("DATA:WIDTH 1")
    inst.write("DATA:START 1")
    inst.write(f"DATA:STOP {points}")

    nr_pt = int(inst.query("WFMPRe:NR_PT?"))
    xzero = float(inst.query("WFMPRe:XZEro?"))
    xincr = float(inst.query("WFMPRe:XINcr?"))
    yzero = float(inst.query("WFMPRe:YZEro?"))
    ymult = float(inst.query("WFMPRe:YMUlt?"))
    yoff = float(inst.query("WFMPRe:YOff?"))

    raw = np.array(inst.query_binary_values(
        "CURVE?", datatype='b', container=np.array))
    time_list = (np.arange(raw.size) * xincr + xzero).tolist()
    voltage_list = ((raw - yoff) * ymult + yzero).tolist()
    return time_list, voltage_list


def main():
    rm = pyvisa.ResourceManager()
    address = 'USB0::0x0699::0x03A2::C040073::INSTR'
    inst = rm.open_resource(address, timeout=20000)
    time_list, voltage_list = get_waveform(inst, channel=1, points=10000)
    save_to_csv(time_list, voltage_list)
    inst.close()

if __name__ == "__main__":
    main()
