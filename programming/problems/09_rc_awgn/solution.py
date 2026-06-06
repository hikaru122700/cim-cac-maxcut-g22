"""問題2-3: スペクトル整形 QAM サンプル列への複素 AWGN 付加

  - α=1 raised cosine、2 sample/symbol で整形したサンプル列を生成 (サンプル数 10**6)。
  - SNR = 20 dB に相当する複素 AWGN を付加する。
    SNR はシンボル判定点 (ISI-free タイミング) での値で定義する。
    そこでは信号電力 = シンボル電力 = 1 なので、雑音電力 = 1/10**(20/10) = 0.01。
  - ISI が生じないタイミングでサンプル抽出し、コンステレーションを図示する。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
QAM_ORDERS = [4, 16, 64, 256]
BETA = 1.0
SPS = 2
SPAN = 12
N_SAMPLE = 10 ** 6          # 総サンプル数
N_SYM = N_SAMPLE // SPS     # シンボル数 (= 500000)
SNR_DB = 20.0
SEED = 3


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"サンプル数={N_SAMPLE} ({SPS} sps -> {N_SYM} シンボル), SNR={SNR_DB} dB\n")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        tx_sym = make_symbols(M, N_SYM)
        shaped, delay = comm.pulse_shape(tx_sym, SPS, BETA, SPAN)

        # 判定点で信号電力=1 になるよう、雑音はシンボル電力(=1)基準で付加する
        rx = comm.add_awgn(shaped, SNR_DB, signal_power=1.0, rng=rng)

        # ISI-free タイミングでダウンサンプル
        rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)

        snr_meas = 10 * np.log10(1.0 / np.mean(np.abs(rx_sym - tx_sym[:len(rx_sym)]) ** 2))
        print(f"{M:3d}QAM: ダウンサンプル後 実測SNR = {snr_meas:5.2f} dB")

        ax.scatter(rx_sym[:4000].real, rx_sym[:4000].imag, s=4, alpha=0.15)
        ax.set_title(f"{M}QAM, RC+AWGN, ISI-free sampled")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"RC-shaped + complex AWGN, after ISI-free downsampling (SNR={SNR_DB:.0f} dB)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation_rc_awgn.png")
    fig.savefig(out, dpi=120)
    print(f"\nコンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
