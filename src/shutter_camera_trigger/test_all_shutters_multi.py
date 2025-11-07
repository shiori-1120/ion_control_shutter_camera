# port0 line4..7 を1タスクで制御（Active High／ビット列を直書き）
# bit割り当て: bit0=line4, bit1=line5, bit2=line6, bit3=line7
# 例) 0b0001 = line4のみON, 0b1100 = line6・line7がON

import time
import nidaqmx
from nidaqmx.constants import LineGrouping
from nidaqmx.stream_writers import DigitalSingleChannelWriter

PORT_RANGE = "Dev3/port0/line4:7"

# --- Active High: 1=ON, 0=OFF（演算なしで直書きする定数セット）---
ALL_OFF = 0b0000
ALL_ON = 0b1111
NM_397 = 0b0001  # line4
NM_397_SIG = 0b0010  # line5
NM_729 = 0b0100  # line6
NM_854 = 0b1000  # line7
NM_729_854 = 0b1100  # line6 & line7
NM_397_ONLY = 0b0001
NM_397SIG_ONLY = 0b0010


def main():
    task = nidaqmx.Task()
    try:
        task.do_channels.add_do_chan(
            PORT_RANGE, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
        writer = DigitalSingleChannelWriter(task.out_stream)
        # nidaqmxのDigitalSingleChannelWriterにはuint8版がないため、uint16版を使用します。
        # 値は0〜65535の範囲で、ここでは下位4ビットのみを使用します（line4〜line7）。
        write_port = writer.write_one_sample_port_uint16  # 呼び出し短縮

        print("Initializing... ALL OFF")
        write_port(ALL_OFF)

        print("シーケンス開始（Ctrl+Cで停止）")
        while True:
            # 4本すべて ON
            write_port(ALL_ON)
            time.sleep(0.001)

            # 4本すべて OFF
            write_port(ALL_OFF)
            time.sleep(0.001)

            # 例: 729と854だけON
            write_port(NM_729_854)
            time.sleep(0.001)
            write_port(ALL_OFF)
            time.sleep(0.001)

            # 例: 397だけON → 397_SIGMAだけON
            write_port(NM_397_ONLY)
            time.sleep(0.001)
            write_port(NM_397SIG_ONLY)
            time.sleep(0.001)
            write_port(ALL_OFF)
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            write_port(ALL_OFF)  # 終了時はOFF
        except Exception:
            pass
        task.close()
        print("終了")


if __name__ == "__main__":
    main()
