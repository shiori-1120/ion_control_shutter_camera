# analyze_from_csv.py

import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from pprint import pprint

# ==============================================================================
# Waveform Analysis Functions (波形評価関数群)
# ==============================================================================


def _find_crossings(time, voltage, threshold):
    """[Helper] 指定した電圧閾値を横切る時刻を線形補間で求める"""
    indices = np.where((voltage[:-1] - threshold)
                       * (voltage[1:] - threshold) <= 0)[0]

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

    v_high, v_low = np.percentile(voltage, 95), np.percentile(voltage, 5)
    v_amp = v_high - v_low
    results.update({'v_high': v_high, 'v_low': v_low, 'v_amplitude': v_amp})

    if v_amp < 1e-3:
        return {'error': 'Amplitude too low'}

    v_10pct, v_50pct, v_90pct = v_low + 0.1 * \
        v_amp, v_low + 0.5 * v_amp, v_low + 0.9 * v_amp
    t_cross_10, t_cross_50, t_cross_90 = _find_crossings(time, voltage, v_10pct), _find_crossings(
        time, voltage, v_50pct), _find_crossings(time, voltage, v_90pct)

    mid_cross_indices = np.where(
        (voltage[:-1] - v_50pct) * (voltage[1:] - v_50pct) <= 0)[0]
    is_rising = voltage[mid_cross_indices + 1] > voltage[mid_cross_indices]

    t_rising_50, t_falling_50 = t_cross_50[is_rising], t_cross_50[~is_rising]
    results['edges'] = {'rising_50pct': t_rising_50,
                        'falling_50pct': t_falling_50}

    rise_times, fall_times = [], []
    for t_r_50 in t_rising_50:
        t10 = t_cross_10[t_cross_10 < t_r_50]
        t90 = t_cross_90[t_cross_90 > t_r_50]
        if t10.size > 0 and t90.size > 0:
            rise_times.append(t90[0] - t10[-1])
    for t_f_50 in t_falling_50:
        t90 = t_cross_90[t_cross_90 < t_f_50]
        t10 = t_cross_10[t_cross_10 > t_f_50]
        if t90.size > 0 and t10.size > 0:
            fall_times.append(t10[0] - t90[-1])

    if rise_times:
        results['rise_time'] = {'mean': np.mean(
            rise_times), 'std': np.std(rise_times)}
    if fall_times:
        results['fall_time'] = {'mean': np.mean(
            fall_times), 'std': np.std(fall_times)}

    if len(t_rising_50) > 1:
        periods = np.diff(t_rising_50)
        results.update({
            'period': {'mean': np.mean(periods), 'std': np.std(periods)},
            'frequency': {'mean': 1.0 / np.mean(periods)},
            'jitter_pk_pk': np.ptp(periods), 'jitter_rms': np.std(periods)})

    overshoots, undershoots = [], []
    edge_indices = np.where(np.diff(voltage > v_50pct))[0]
    for i in range(len(edge_indices) - 1):
        segment = voltage[edge_indices[i]:edge_indices[i+1]]
        if voltage[edge_indices[i]+1] > v_50pct:
            overshoots.append(np.max(segment) - v_high)
        else:
            undershoots.append(v_low - np.min(segment))

    if overshoots:
        results['overshoot_v'] = {'mean': np.mean(
            overshoots), 'max': np.max(overshoots)}
    if undershoots:
        results['undershoot_v'] = {'mean': np.mean(
            undershoots), 'max': np.max(undershoots)}
    return results


def calculate_skew(all_results, ref_channel='CH1'):
    if ref_channel not in all_results or 'edges' not in all_results[ref_channel]:
        return {'error': "Reference channel not found"}
    ref_edges = all_results[ref_channel]['edges']['rising_50pct']
    if len(ref_edges) == 0:
        return {'error': "No rising edges in ref channel"}

    skew_results = {}
    for ch_name, ch_results in all_results.items():
        if ch_name == ref_channel or 'edges' not in ch_results:
            continue
        target_edges = ch_results['edges']['rising_50pct']
        skews = [target_edges -
                 ref_edge_time for ref_edge_time in ref_edges if len(target_edges) > 0]
        if skews:
            closest_skews = [s[np.argmin(np.abs(s))] for s in skews]
            skew_results[f"{ch_name}_vs_{ref_channel}"] = {'mean': np.mean(closest_skews), 'std': np.std(
                closest_skews), 'max': np.max(closest_skews), 'min': np.min(closest_skews)}
    return skew_results


def analyze_waveforms(all_waveforms_data, ref_channel='CH1'):
    all_results = {ch: analyze_single_channel(
        data['time'], data['voltage']) for ch, data in all_waveforms_data.items()}
    skew_results = calculate_skew(all_results, ref_channel)
    return {'channel_metrics': all_results, 'skew_metrics': skew_results}

# ==============================================================================
# Helper Functions for IO and Plotting
# ==============================================================================


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


# 単位変換 / 表示スケール（ここを変更すればすぐ元に戻せます）
TIME_SCALE_MS = 1e3      # 秒 -> ミリ秒変換係数
VOLTAGE_SCALE = 1      # プロット時に電圧を 1/10 にする。元に戻すには 1.0 に変更


def plot_waveforms(waveforms_data, title="Oscilloscope Waveforms"):
    num_channels = len(waveforms_data)

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 10), sharex=True)
    if num_channels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=16)

    for i, (ch_name, data) in enumerate(waveforms_data.items()):
        # 時間を ms にスケーリング、電圧を VOLTAGE_SCALE で縮小してプロット
        time_data = np.array(data['time']) * TIME_SCALE_MS
        voltage_data = np.array(data['voltage']) * VOLTAGE_SCALE

        axes[i].scatter(time_data, voltage_data, label=ch_name, s=5, alpha=0.7)

        axes[i].set_ylabel(f"Voltage (V) x{VOLTAGE_SCALE:.3g}")
        axes[i].grid(True)
        axes[i].legend(loc='upper right')

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Try to infer source filename from the figure title "Waveforms from <basename>"
    fig_title = fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else title
    prefix = "Waveforms from "
    out_saved = None
    if fig_title.startswith(prefix):
        src_name = fig_title[len(prefix):].strip()  # e.g. "data.csv"
        base = os.path.splitext(src_name)[0]        # e.g. "data"
        out_saved = "output\\" + base + "_report.png"
        fig.savefig(out_saved, dpi=200)
        print(f"\nプロットを {out_saved} に保存しました。")
    else:
        # Fallback: save with generic name
        out_saved = "waveforms_report.png"
        fig.savefig(out_saved, dpi=200)
        print(f"\nプロットを {out_saved} に保存しました。")

    plt.show()


# ==============================================================================
# Report Generation Function (★ここを修正★)
# ==============================================================================
def save_report_to_txt(report_data, output_filepath):
    """分析レポートをテキストファイルに保存する"""
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write("==================================================\n")
        f.write("            WAVEFORM ANALYSIS REPORT\n")
        f.write("==================================================\n")
        source_file = os.path.basename(
            output_filepath).replace('_report.txt', '.csv')
        f.write(f"Source File: {source_file}\n")
        f.write(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

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
            f.write(
                f"    - V_amplitude: {metrics.get('v_amplitude', 0):.4f} V\n\n")

            f.write(f"  Timing & Jitter:\n")
            freq = metrics.get('frequency', {})
            period = metrics.get('period', {})
            f.write(f"    - Frequency  : {freq.get('mean', 0):.4f} Hz\n")
            f.write(
                f"    - Period     : {period.get('mean', 0) * 1000:.4f} ms (Std: {period.get('std', 0) * 1000:.4f} ms)\n")
            f.write(
                f"    - Jitter Pk-Pk: {metrics.get('jitter_pk_pk', 0) * 1000:.4f} ms\n")
            f.write(
                f"    - Jitter RMS : {metrics.get('jitter_rms', 0) * 1000:.4f} ms\n\n")

            f.write(f"  Edge Characteristics:\n")
            rise = metrics.get('rise_time', {})
            fall = metrics.get('fall_time', {})
            # rise/fall time をミリ秒表示に変更
            f.write(
                f"    - Rise Time  : {rise.get('mean', 0) * 1e3:.6f} ms (Std: {rise.get('std', 0) * 1e3:.6f} ms)\n")
            f.write(
                f"    - Fall Time  : {fall.get('mean', 0) * 1e3:.6f} ms (Std: {fall.get('std', 0) * 1e3:.6f} ms)\n\n")

            f.write(f"  Signal Integrity:\n")
            overshoot = metrics.get('overshoot_v', {})
            undershoot = metrics.get('undershoot_v', {})
            f.write(
                f"    - Overshoot  : {overshoot.get('max', 0):.4f} V (Mean: {overshoot.get('mean', 0):.4f} V)\n")
            f.write(
                f"    - Undershoot : {undershoot.get('max', 0):.4f} V (Mean: {undershoot.get('mean', 0):.4f} V)\n\n")

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
                f.write(
                    f"    - Mean : {values.get('mean', 0) * 1000:.4f} ms (平均のズレ)\n")

        f.write("\n" + "-" * 50 + "\n")
        f.write("           [ RAW EDGE TIMESTAMPS (ms) ]\n")
        f.write("-" * 50 + "\n")
        f.write("# 各エッジが振幅の50%を通過した時刻 [ミリ秒] のリストです。\n\n")

        for ch_name, metrics in channel_metrics.items():
            f.write(f"--- Channel: {ch_name} ---\n")
            edges = metrics.get('edges', {})

            rising_edges = edges.get('rising_50pct', np.array([]))
            rising_str_list = [
                f"{(t * TIME_SCALE_MS):.6f}" for t in rising_edges]
            f.write(f"  Rising Edges  : [ {', '.join(rising_str_list)} ]\n")

            falling_edges = edges.get('falling_50pct', np.array([]))
            falling_str_list = [
                f"{(t * TIME_SCALE_MS):.6f}" for t in falling_edges]
            f.write(f"  Falling Edges : [ {', '.join(falling_str_list)} ]\n\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("             END OF REPORT\n")
        f.write("=" * 50 + "\n")

    print(f"\n評価レポートを {output_filepath} に保存しました。")


def main():
    filepath = input("評価したいCSVファイルのパスを入力してください: ").strip()

    if not os.path.exists(filepath):
        print(f"\n[エラー] ファイルが見つかりません: {filepath}")
        return

    try:
        waveforms = load_waveforms_from_csv(filepath)
        plot_waveforms(
            waveforms, title=f"Waveforms from {os.path.basename(filepath)}")

        analysis_report = analyze_waveforms(waveforms, ref_channel='CH1')

        # コンソールへのpprint表示は、デバッグ用に残しても良い
        # print("\n--- 詳細な評価結果 (Raw Data) ---")
        # pprint(analysis_report)
        # print("-" * 35)

        base_name, _ = os.path.splitext(filepath)
        report_filepath = base_name + "_report.txt"
        save_report_to_txt(analysis_report, report_filepath)

        print("\n--- 正常に終了しました ---")
    except Exception as e:
        print(f"\n[エラー] 処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()
