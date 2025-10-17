import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# チャンネルごとの立ち上がり・立ち下がり時刻のデータ
# レポートからコピーしたデータをここに貼り付けます
data = {
    'CH1': {
        'rising': [0.0037506],
        'falling':  [-0.0124500, 0.0234100]

    },
    'CH2': {
        'rising': [-0.0076506],
        'falling': [0.0033100]

    },
    'CH3': {
        'rising': [0.0089100],
        'falling': [-0.0084700]
    },
    'CH4': {
        'rising':  [0.0094300],
        'falling': [-0.0120112, 0.0240700]

    }
}

# チャンネルごとの色を定義
colors = {
    'CH1': 'royalblue',
    'CH2': 'deepskyblue',
    'CH3': 'darkorange',
    'CH4': 'red'
}


def create_timeline_plot(data, colors, output_filename=None):
    """
    立ち上がり/立ち下がりデータからタイムラインプロットを作成し、画像として保存する関数。
    """
    # タイムスタンプ付きファイル名を生成
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_timeline.png"

    # 4つのサブプロットを縦に並べて作成 (sharex=TrueでX軸を共有)
    fig, axes = plt.subplots(4, 1, figsize=(15, 6), sharex=True)
    fig.suptitle('Signal High State Timeline', fontsize=16)

    # 各チャンネルのデータをプロット
    for i, (ch_name, ch_data) in enumerate(data.items()):
        ax = axes[i]
        high_intervals = []

        # 各立ち上がりエッジに対して、その直後の立ち下がりエッジを探す
        sorted_falling = sorted(ch_data['falling'])
        for r_time in sorted(ch_data['rising']):
            try:
                f_time = next(f for f in sorted_falling if f > r_time)
                high_intervals.append((r_time*1000, (f_time - r_time)*1000))
            except StopIteration:
                pass

        ax.broken_barh(high_intervals, yrange=(0.3, 0.4),
                       facecolors=colors.get(ch_name, 'gray'))

        ax.set_yticks([])
        ax.set_ylabel(ch_name, rotation=0, labelpad=30,
                      ha='right', va='center', fontsize=12)
        ax.grid(True, axis='x', linestyle=':')

    axes[-1].set_xlabel('Time (ms)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    plt.show()
    
    print(f"タイムライングラフを '{output_filename}' として保存しました。")


# --- メインの実行部分 ---
if __name__ == "__main__":
    create_timeline_plot(data, colors)
