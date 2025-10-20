# 特定の範囲を受け取ってその範囲内の光子数を積層する関数（横軸に対して縦を積分する）
def integrate_photon_counts(spectrum, start_wavelength, end_wavelength):
    pass  # 実装はここに記述してください

# イオンの個数が変化していないことを確認する関数
def verify_ion_count_consistency(ion_counts_before, ion_counts_after):
    pass  # 実装はここに記述してください

# integrate_photon_countsをイオンの個数だけ繰り返し、さらに周波数ごとに繰り返して、周波数と励起成功確率の2D配列を作成する関数
def create_frequency_excitation_probability_matrix(spectrum_data, ion_counts):
    pass  # 実装はここに記述してください

# 周波数と励起成功確率の2D配列をプロットする関数。ローレンチアンフィットも行う。中心周波数と線幅も求める。
def plot_frequency_excitation_probability(matrix, frequencies, excitation_probabilities):
    pass  # 実装はここに記述してください
