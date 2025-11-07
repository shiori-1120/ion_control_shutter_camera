import pandas as pd
import numpy as np


def analyze_waveform(filepath, threshold=1.8):
    try:
        # CSVファイルを読み込み
        df = pd.read_csv(filepath)
        print(f"'{filepath}' を正常に読み込みました。")
    except FileNotFoundError:
        print(f"エラー: ファイル '{filepath}' が見つかりません。")
        return

    # 時間のカラム名を取得
    time_col = df.columns[0]
    # 電圧チャンネルのカラム名を取得 (時間カラムを除く)
    voltage_cols = df.columns[1:]

    print(f"しきい値: {threshold} V\n")

    # 各チャンネルについてループ処理
    for channel in voltage_cols:
        print(f"--- {channel} の解析結果 ---")

        time_series = df[time_col]
        voltage_series = df[channel]

        # しきい値に基づいて電圧をHIGH(1)とLOW(0)に変換
        digital_signal = (voltage_series > threshold).astype(int)

        # 信号が変化した点のインデックスを検出 (立ち上がり/立ち下がり)
        diff_signal = digital_signal.diff()
        transition_indices = diff_signal[diff_signal != 0].index

        if len(transition_indices) < 2:
            print("信号の変化が検出されなかったため、解析をスキップします。")
            print("-" * (len(channel) + 15) + "\n")
            continue

        # 変化点における時間を取得
        transition_times = time_series.iloc[transition_indices].to_numpy()
        # 変化点における状態 (HIGH/LOW) を取得
        transition_states = digital_signal.iloc[transition_indices].to_numpy()

        # 各パルスの幅(時間)を計算（元は秒）
        pulse_widths = np.diff(transition_times)
        pulse_widths_ms = pulse_widths * 1e3  # ミリ秒に変換

        # 最初の遷移幅（最初のパルス幅）を解析から除外する
        if pulse_widths_ms.size > 0:
            pulse_widths_ms = pulse_widths_ms[1:]

        # パルス幅が計算された期間の状態 (パルスの開始時の状態)
        # 最後の遷移は幅の終わりなので、それより前の状態リストを使う
        pulse_states = transition_states[:-1]

        # 状態配列も先頭要素を除外して幅配列と長さを揃える
        if pulse_states.size > 0:
            pulse_states = pulse_states[1:]

        # HIGHパルスとLOWパルスの幅をそれぞれ抽出
        high_widths_ms = pulse_widths_ms[pulse_states == 1]
        low_widths_ms = pulse_widths_ms[pulse_states == 0]

        # 平均と分散を計算 (np.nanmean, np.nanvarでNaNを無視)
        # HIGH
        if len(high_widths_ms) > 0:
            high_avg_ms = np.mean(high_widths_ms)
            high_std_ms = np.std(high_widths_ms)
            print(f"HIGH状態:")
            print(f"  サンプル数: {len(high_widths_ms)}")
            print(f"  平均幅: {high_avg_ms:.6f} ms")
            print(f"  標準偏差: {high_std_ms:.6e} ms")
        else:
            print("HIGH状態のパルスが見つかりませんでした。")

        # LOW
        if len(low_widths_ms) > 0:
            low_avg_ms = np.mean(low_widths_ms)
            low_std_ms = np.std(low_widths_ms)
            print(f"LOW状態:")
            print(f"  サンプル数: {len(low_widths_ms)}")
            print(f"  平均幅: {low_avg_ms:.6f} ms")
            print(f"  標準偏差: {low_std_ms:.6e} ms")
        else:
            print("LOW状態のパルスが見つかりませんでした。")

        print("-" * (len(channel) + 15) + "\n")


if __name__ == '__main__':
    # ユーザーがアップロードしたファイル名を指定
    csv_file = 'C:\\Users\\karishio\\Desktop\\single_ion_control\\src\\waveform\\output\\20251107_114825_waveform.csv'
    # HIGH/LOWを判定するしきい値を設定 (データに合わせて調整してください)
    voltage_threshold = 1.8
    analyze_waveform(csv_file, voltage_threshold)
