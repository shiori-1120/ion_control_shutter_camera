import pyvisa
import os
import csv
from datetime import datetime


def save_to_csv(time_data, voltage_data, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"waveform_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    rows = zip(time_data, voltage_data)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "Voltage (V)"])
        writer.writerows(rows)

    print(f"データを {filepath} に保存しました。")


def get_waveform(inst, channel, points=10000):

    inst.write(f"DATA:SOURCE CH{channel}")  # DATa:SOUrce 取得チャンネルを選択
    inst.write("DATA:ENCdg ASCii")  # DATa:ENCdg 転送する波形データのフォーマット形式を指定
    inst.write("DATA:WIDTH 1")  # DATa:WIDth 波形の1データポイントあたりを何バイトで表現するかを指定

    # DATa:STARt, DATa:STOP 取得したい波形データの開始点と終了点を指定
    inst.write("DATA:START 1")
    inst.write(f"DATA:STOP {points}")

    datapoint = int(inst.query("WFMPRE:NR_PT?"))

    xzero = float(inst.query("WFMP:XZE?"))
    xinc = float(inst.query("WFMP:XIN?"))
    time_list = [xzero + xinc * i for i in range(datapoint)]

    yzero = float(inst.query("WFMP:YZE?"))
    ymult = float(inst.query("WFMP:YMU?"))

    # --- CURVE? 応答は大きなデータ転送で時間がかかるため一時的に拡張 ---
    old_timeout = getattr(inst, "timeout", None)
    old_chunk = getattr(inst, "chunk_size", None)
    try:
        # ms 単位の十分に大きなタイムアウト（必要に応じて増やす）
        inst.timeout = 120000
        # chunk_size が小さいと分割による遅延が発生することがあるため拡張
        try:
            if old_chunk is None or old_chunk < 1024000:
                inst.chunk_size = 1024000
        except Exception:
            # 一部のバックエンドでは属性がない場合がある
            pass

        print(
            f"[INFO] CH{channel} CURVE? を取得（timeout={inst.timeout} ms, chunk={getattr(inst, 'chunk_size', 'N/A')})")
        try:
            curve = inst.query("CURVE?")
        except pyvisa.errors.VisaIOError as e:
            # 一度だけリトライ（より長いタイムアウト）
            print(f"[WARN] CURVE? でタイムアウト/エラー: {e}. 1回だけリトライします...")
            inst.timeout = 300000
            try:
                curve = inst.query("CURVE?")
            except Exception as e2:
                raise pyvisa.errors.VisaIOError(e2)

    finally:
        # 元の値へ復帰（存在した場合）
        if old_timeout is not None:
            inst.timeout = old_timeout
        if old_chunk is not None:
            try:
                inst.chunk_size = old_chunk
            except Exception:
                pass

    raw_voltage_list = [int(v) for v in curve.split(",")]
    voltage_list = [yzero + ymult * v for v in raw_voltage_list]

    return time_list, voltage_list


def main():
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    print(resources)

    # 利用可能なリソースを取得し、速い接続を優先する（USB -> 直接IP -> その他）
    available = list(resources)
    preferred_order = []
    # USB を最優先
    preferred_order += [r for r in available if r.startswith("USB0")]
    # 直接IP (ローカルIP) を次に
    preferred_order += [
        r for r in available if "TCPIP0::192.168" in r or "TCPIP::" in r and "inst0" in r]
    # 残りは最後
    preferred_order += [r for r in available if r not in preferred_order]

    # 優先順に試行して最初に成功した機器を使う
    inst = None
    chosen_addr = None
    for addr in preferred_order:
        try:
            inst = rm.open_resource(addr, timeout=5000)
            inst.read_termination = '\n'
            inst.write_termination = '\n'
            inst.timeout = 5000
            idn = inst.query("*IDN?")
            print(f"{addr} -> {idn.strip()}")
            chosen_addr = addr
            break
        except Exception as e:
            print(f"[WARN] {addr} に接続できませんでした: {e}")
            try:
                if inst:
                    inst.close()
            except Exception:
                pass
            inst = None

    if not inst:
        print("[ERROR] どの機器にも接続できませんでした。接続・ケーブルを確認してください。")
        return

    # 選ばれた機器から波形を取得して保存
    try:
        print(f"[INFO] {chosen_addr} から波形を取得します。")
        time_list, voltage_list = get_waveform(inst, channel=1, points=10000)
        save_to_csv(time_list, voltage_list)
    except Exception as e:
        print(f"[ERROR] {chosen_addr} からの波形取得で例外: {e}")
    finally:
        try:
            inst.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
