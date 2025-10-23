import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem as _sem


def integrate_photon_counts(bright_data: list[np.ndarray], dark_data: list[np.ndarray], plot: bool = False):

    def to_profile(frame):
        a = np.asarray(frame)
        return a.sum(axis=0).astype(float)

    #　空の場合のデータは表示しない。どちらかでもデータがあればプロットする
    if len(bright_data) == 0 or len(dark_data) == 0:
        raise ValueError('bright_data and dark_data must be non-empty lists')

    bright_profiles = [to_profile(f) for f in bright_data]
    dark_profiles = [to_profile(f) for f in dark_data]

    # ensure consistent length
    L = len(bright_profiles[0])
    if any(p.shape[0] != L for p in bright_profiles + dark_profiles):
        raise ValueError('All frames must have the same number of columns')

    x = np.arange(L)
    bright_stack = np.vstack(bright_profiles)
    dark_stack = np.vstack(dark_profiles)

    bright_mean = bright_stack.mean(axis=0)
    bright_sem = _sem(bright_stack, axis=0, nan_policy='omit')
    dark_mean = dark_stack.mean(axis=0)
    dark_sem = _sem(dark_stack, axis=0, nan_policy='omit')

    net_mean = bright_mean - dark_mean
    # propagate sems (assuming independence): sem_net = sqrt(sem_b^2 + sem_d^2)
    net_sem = np.sqrt(np.square(bright_sem) + np.square(dark_sem))

    if plot:
        plt.figure(figsize=(8, 4))
        plt.fill_between(x, bright_mean - bright_sem,
                         bright_mean + bright_sem, color='C1', alpha=0.3)
        plt.plot(x, bright_mean, color='C1', label='bright')
        plt.fill_between(x, dark_mean - dark_sem, dark_mean +
                         dark_sem, color='C0', alpha=0.3)
        plt.plot(x, dark_mean, color='C0', label='dark')
        plt.plot(x, net_mean, color='k',
                 label='net (bright - dark)', linewidth=1)
        plt.fill_between(x, net_mean - net_sem, net_mean +
                         net_sem, color='k', alpha=0.15)
        plt.xlabel('x (pixel)')
        plt.ylabel('integrated counts')
        plt.legend()
        plt.tight_layout()
        plt.show()

    return 

# イオンの個数が変化していないことを確認する関数


def verify_ion_count_consistency(ndarray, ion_numbers):
    
    pass  # 実装はここに記述してください

# integrate_photon_countsをイオンの個数だけ繰り返し、さらに周波数ごとに繰り返して、周波数と励起成功確率の2D配列を作成する関数


def create_frequency_excitation_probability_matrix(spectrum_data, ion_counts):
    pass  # 実装はここに記述してください

# 周波数と励起成功確率の2D配列をプロットする関数。ローレンチアンフィットも行う。中心周波数と線幅も求める。


def plot_frequency_excitation_probability(matrix, frequencies, excitation_probabilities):
    pass  # 実装はここに記述してください
