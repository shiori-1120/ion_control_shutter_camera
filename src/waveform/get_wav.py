import pyvisa
import os
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import time
import numpy as np
import pandas as pd
from pprint import pprint
import argparse # コマンドライン引数を扱うために追加

# python oscilloscope_tool.py --mode acquire
# python oscilloscope_tool.py --mode analyze --file ../output/waveform_4ch_20231027_103000.csv


def _find_crossings(time, voltage, threshold):
    """[Helper] 指定した電圧閾値を横切る時刻を線形補間で求める"""
    indices = np.where((voltage[:-1] - threshold) * (voltage[1:] - threshold) <= 0)[0]
    
    crossings = []
    for i in indices:
        v1, v2 = voltage[i], voltage[i+1]
        t1, t2 = time[i], time[i+1]
        if v2 - v1 != 0:
            t_cross = t1 + (t2 - t1) * (threshold - v1) / (v2 - v1)
            crossings.append(t_cross)
    return np.array(crossings)

def analyze_single_channel(time, voltage):
    """単一チャンネルの波形を包括的に分析する"""
    time, voltage = np.array(time), np.array(voltage)
    results = {}

    v_high = np.percentile(voltage, 95)
    v_low = np.percentile(voltage, 5)
    v_amp = v_high - v_low
    results['v_high'] = v_high
    results['v_low'] = v_low
    results['v_amplitude'] = v_amp

    if v_amp < 1e-3: return {'error': 'Amplitude too low'}

    v_10pct, v_50pct, v_90pct = v_low + 0.1 * v_amp, v_low + 0.5 * v_amp, v_low + 0.9 * v_amp
    t_cross_10, t_cross_50, t_cross_90 = _find_crossings(time, voltage, v_10pct), _find_crossings(time, voltage, v_50pct), _find_crossings(time, voltage, v_90pct)
    
    mid_cross_indices = np.where((voltage[:-1] - v_50pct) * (voltage[1:] - v_50pct) <= 0)[0]
    is_rising = voltage[mid_cross_indices + 1] > voltage[mid_cross_indices]
    
    t_rising_50, t_falling_50 = t_cross_50[is_rising], t_cross_50[~is_rising]
    results['edges'] = {'rising_50pct': t_rising_50, 'falling_50pct': t_falling_50}

    rise_times, fall_times = [], []
    for t_r_50 in t_rising_50:
        t10 = t_cross_10[t_cross_10 < t_r_50]
        t90 = t_cross_90[t_cross_90 > t_r_50]
        if t10.size > 0 and t90.size > 0: rise_times.append(t90[0] - t10[-1])
    for t_f_50 in t_falling_50:
        t90 = t_cross_90[t_cross_90 < t_f_50]
        t10 = t_cross_10[t_cross_10 > t_f_50]
        if t90.size > 0 and t10.size > 0: fall_times.append(t10[0] - t90[-1])

    if rise_times: results['rise_time'] = {'mean': np.mean(rise_times), 'std': np.std(rise_times), 'count': len(rise_times)}
    if fall_times: results['fall_time'] = {'mean': np.mean(fall_times), 'std': np.std(fall_times), 'count': len(fall_times)}

    if len(t_rising_50) > 1:
        periods = np.diff(t_rising_50)
        results.update({
            'period': {'mean': np.mean(periods), 'std': np.std(periods)},
            'frequency': {'mean': 1.0 / np.mean(periods)},
            'jitter_pk_pk': np.ptp(periods),
            'jitter_rms': np.std(periods)
        })

    overshoots, undershoots = [], []
    edge_indices = np.where(np.diff(voltage > v_50pct))[0]
    for i in range(len(edge_indices) - 1):
        segment = voltage[edge_indices[i]:edge_indices[i+1]]
        if voltage[edge_indices[i]+1] > v_50pct: overshoots.append(np.max(segment) - v_high)
        else: undershoots.append(v_low - np.min(segment))
            
    if overshoots: results['overshoot_v'] = {'mean': np.mean(overshoots), 'max': np.max(overshoots)}
    if undershoots: results['undershoot_v'] = {'mean': np.mean(undershoots), 'max': np.max(undershoots)}

    return results

def calculate_skew(all_results, ref_channel='CH1'):
    """複数チャンネルの分析結果からスキューを計算する"""
    if ref_channel not in all_results or 'edges' not in all_results[ref_channel]: return {'error': "Reference channel not found"}
    ref_edges = all_results[ref_channel]['edges']['rising_50pct']
    if len(ref_edges) == 0: return {'error': "No rising edges in ref channel"}

    skew_results = {}
    for ch_name, ch_results in all_results.items():
        if ch_name == ref_channel or 'edges' not in ch_results: continue
        target_edges = ch_results['edges']['rising_50pct']
        skews = [target_edges - ref_edge_time for ref_edge_time in ref_edges if len(target_edges) > 0]
        if skews:
            closest_skews = [s[np.argmin(np.abs(s))] for s in skews]
            skew_results[f"{ch_name}_vs_{ref_channel}"] = {'mean': np.mean(closest_skews), 'std': np.std(closest_skews), 'max': np.max(closest_skews), 'min': np.min(closest_skews)}
            
    return skew_results

def analyze_waveforms(all_waveforms_data, ref_channel='CH1'):
    """全チャンネルの波形データをまとめて分析する"""
    all_results = {ch: analyze_single_channel(data['time'], data['voltage']) for ch, data in all_waveforms_data.items()}
    skew_results = calculate_skew(all_results, ref_channel)
    return {'channel_metrics': all_results, 'skew_metrics': skew_results}

def acquire_waveform(inst, channel, points=100000):
    inst.write(f"DATA:SOURCE CH{channel}")
    inst.write("DATA:ENCdg ASCii")
    inst.write("DATA:WIDTH 1")
    inst.write(f"DATA:STOP {points}")
    nr_pt = int(inst.query("WFMPRe:NR_PT?"))
    xzero, xincr = float(inst.query("WFMPRe:XZEro?")), float(inst.query("WFMPRe:XINcr?"))
    yzero, ymult, yoff = float(inst.query("WFMPRe:YZEro?")), float(inst.query("WFMPRe:YMUlt?")), float(inst.query("WFMPRe:YOff?"))
    time_list = np.arange(nr_pt) * xincr + xzero
    curve = inst.query("CURVE?")
    raw_voltage_list = np.fromstring(curve, dtype=int, sep=',')
    voltage_list = (raw_voltage_list - yoff) * ymult + yzero
    return time_list.tolist(), voltage_list.tolist()

def save_waveforms_to_csv(waveforms_data, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"waveform_4ch_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    df = pd.DataFrame({'Time (s)': waveforms_data['CH1']['time']})
    for ch_name, data in waveforms_data.items():
        df[f"{ch_name}_Voltage (V)"] = data['voltage']
    df.to_csv(filepath, index=False)
    print(f"\n波形データを {filepath} に保存しました。")
    return filepath

def load_waveforms_from_csv(filepath):
    print(f"\n--- CSVファイルから波形データを読み込み中: {filepath} ---")
    df = pd.read_csv(filepath)
    waveforms = {}
    time_data = df.iloc[:, 0].tolist()
    for col in df.columns[1:]:
        ch_name = col.split('_')[0]
        waveforms[ch_name] = {'time': time_data, 'voltage': df[col].tolist()}
    print("データの読み込みが完了しました。")
    return waveforms

def plot_waveforms(waveforms_data):
    num_channels = len(waveforms_data)
    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 10), sharex=True)
    if num_channels == 1: axes = [axes]
    fig.suptitle("Oscilloscope Waveforms", fontsize=16)
    
    time_data = list(waveforms_data.values())[0]['time']
    for i, (ch_name, data) in enumerate(waveforms_data.items()):
        axes[i].plot(time_data, data['voltage'], label=ch_name)
        axes[i].set_ylabel("Voltage (V)")
        axes[i].grid(True)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()

def run_acquisition_mode(channels, visa_resource):
    rm = pyvisa.ResourceManager()
    inst = None
    all_waveforms = {}
    try:
        inst = rm.open_resource(visa_resource, timeout=20000)
        print(f"接続成功: {inst.query('*IDN?').strip()}")
        inst.write("ACQuire:STATE STOP"); time.sleep(0.1)

        for ch in channels:
            print(f"CH{ch} のデータを取得中...")
            time_data, voltage_data = acquire_waveform(inst, ch)
            all_waveforms[f'CH{ch}'] = {'time': time_data, 'voltage': voltage_data}

        saved_filepath = save_waveforms_to_csv(all_waveforms)
        plot_waveforms(all_waveforms)
        print(f"\nグラフ描画完了。CSVは {saved_filepath} に保存されました。")

    except pyvisa.errors.VisaIOError:
        print(f"VISA I/O Error: オシロスコープが見つかりません。リソース文字列 '{visa_resource}' を確認してください。")
        raise
    finally:
        if inst:
            inst.write("ACQuire:STATE RUN")
            inst.close()
            print("リソースを解放しました.")

def save_report_to_txt(report_data, output_filepath):
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("            WAVEFORM ANALYSIS REPORT\n")
        f.write("==================================================\n")
        source_file = os.path.basename(output_filepath).replace('_report.txt', '.csv')
        f.write(f"Source File: {source_file}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("--------------------------------------------------\n")
        f.write("         [ 数値の見方 (Interpretation Guide) ]\n")
        f.write("--------------------------------------------------\n")
        f.write(" - Jitter (ジッター)  : 信号の周期の『揺らぎ』。値が小さいほど信号が時間的に安定しています。\n")
        f.write(" - Rise/Fall Time   : 信号の立ち上がり/立ち下がり時間。速すぎるとノイズ源に、遅すぎるとタイミングエラーの原因になります。\n")
        f.write(" - Overshoot        : 信号がHighレベルを超えて突き抜ける現象。大きいと素子破壊のリスクがあります。\n")
        f.write(" - Undershoot       : 信号がLowレベルを下回る現象。ノイズマージンを減らし、誤動作の原因になります。\n")
        f.write(" - Skew (スキュー)    : 基準信号(CH1)に対する時間的な『ズレ』。Meanは平均のズレ時間を示します。\n\n")

        f.write("--------------------------------------------------\n")
        f.write("                 CHANNEL METRICS\n")
        f.write("--------------------------------------------------\n\n")

        channel_metrics = report_data.get('channel_metrics', {})
        for ch_name, metrics in channel_metrics.items():
            f.write(f"--- Channel: {ch_name} ---\n")
            if 'error' in metrics:
                f.write(f"  Error: {metrics['error']}\n\n")
                continue
            
            f.write(f"  Voltage Levels:\n")
            f.write(f"    - V_high     : {metrics.get('v_high', 0):.4f} V\n")
            f.write(f"    - V_low      : {metrics.get('v_low', 0):.4f} V\n")
            f.write(f"    - V_amplitude: {metrics.get('v_amplitude', 0):.4f} V\n\n")

            f.write(f"  Timing & Jitter:\n")
            freq = metrics.get('frequency', {})
            period = metrics.get('period', {})
            f.write(f"    - Frequency  : {freq.get('mean', 0):.4f} Hz\n")
            f.write(f"    - Period     : {period.get('mean', 0) * 1000:.4f} ms (Std: {period.get('std', 0) * 1000:.4f} ms)\n")
            f.write(f"    - Jitter Pk-Pk: {metrics.get('jitter_pk_pk', 0) * 1000:.4f} ms\n")
            f.write(f"    - Jitter RMS : {metrics.get('jitter_rms', 0) * 1000:.4f} ms\n\n")
            
            f.write(f"  Edge Characteristics:\n")
            rise = metrics.get('rise_time', {})
            fall = metrics.get('fall_time', {})
            f.write(f"    - Rise Time  : {rise.get('mean', 0) * 1e6:.4f} us (Std: {rise.get('std', 0) * 1e6:.4f} us)\n")
            f.write(f"    - Fall Time  : {fall.get('mean', 0) * 1e6:.4f} us (Std: {fall.get('std', 0) * 1e6:.4f} us)\n\n")

            f.write(f"  Signal Integrity:\n")
            overshoot = metrics.get('overshoot_v', {})
            undershoot = metrics.get('undershoot_v', {})
            f.write(f"    - Overshoot  : {overshoot.get('max', 0):.4f} V (Mean: {overshoot.get('mean', 0):.4f} V)\n")
            f.write(f"    - Undershoot : {undershoot.get('max', 0):.4f} V (Mean: {undershoot.get('mean', 0):.4f} V)\n\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("                   SKEW METRICS\n")
        f.write("-" * 50 + "\n\n")

        skew_metrics = report_data.get('skew_metrics', {})
        if 'error' in skew_metrics:
             f.write(f"  Error: {skew_metrics['error']}\n\n")
        elif not skew_metrics:
            f.write("No skew data available.\n")
        else:
            ref_ch = list(skew_metrics.keys())[0].split('_vs_')[-1]
            f.write(f"Reference Channel: {ref_ch}\n\n")
            for skew_name, values in skew_metrics.items():
                f.write(f"--- Skew: {skew_name} ---\n")
                f.write(f"    - Mean : {values.get('mean', 0) * 1000:.4f} ms\n")
                f.write(f"    - Std  : {values.get('std', 0) * 1000:.4f} ms\n")
                f.write(f"    - Min  : {values.get('min', 0) * 1000:.4f} ms\n")
                f.write(f"    - Max  : {values.get('max', 0) * 1000:.4f} ms\n\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("             END OF REPORT\n")
        f.write("=" * 50 + "\n")

def run_analysis_mode(filepath):
    print("\n--- モード: 波形評価 (Analysis) ---")
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません - {filepath}")
        return

    waveforms = load_waveforms_from_csv(filepath)
    plot_waveforms(waveforms)
    
    print("\n波形評価を実行中...")
    analysis_report = analyze_waveforms(waveforms, ref_channel='CH1')
    
    print("\n" + "="*50)
    print(" " * 10 + "WAVEFORM ANALYSIS REPORT (Raw Console Output)")
    print("="*50)
    pprint(analysis_report)
    print("\n" + "="*50)

    base_name, _ = os.path.splitext(filepath)
    report_filepath = base_name + "_report.txt"
    save_report_to_txt(analysis_report, report_filepath)
    print(f"\n評価レポートを {report_filepath} に保存しました。")
    print("波形評価が完了しました。")

# ==============================================================================
# メインの実行ブロック
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="オシロスコープ波形ツール: 波形の取得またはCSVからの分析を行います。")
    parser.add_argument(
        '--mode', 
        type=str, 
        required=True, 
        choices=['acquire', 'analyze'],
        help="'acquire': オシロスコープから波形を取得します。 'analyze': CSVファイルから波形を分析します。"
    )
    parser.add_argument(
        '--file',
        type=str,
        help="分析モード時に使用するCSVファイルのパスを指定します。"
    )
    parser.add_argument(
        '--resource',
        type=str,
        default='USB0::0x0699::0x03A2::C040073::INSTR', # ご自身の環境に合わせて変更
        help="オシロスコープのVISAリソース文字列。"
    )
    
    args = parser.parse_args()

    if args.mode == 'acquire':
        channels_to_measure = [1, 2, 3, 4]
        run_acquisition_mode(channels_to_measure, args.resource)

    elif args.mode == 'analyze':
        if not args.file:
            print("エラー: 'analyze'モードでは --file <ファイルパス> の指定が必須です。")
            return
        run_analysis_mode(args.file)

if __name__ == "__main__":
    main()