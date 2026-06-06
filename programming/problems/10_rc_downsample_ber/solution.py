"""問題2-4: ダウンサンプリング・QAM復調・デマッピングと BER

問2-3 で得た「複素AWGN付きスペクトル整形サンプル列」を、ISI-free タイミングで
ダウンサンプリングし、QAM復調(硬判定)+ QAMデマッピングしてビット系列を復元、
ビット誤り率(BER)を計算する。

raised cosine は Nyquist パルスなので、判定点では整形なし(第1週)と同じ受信シンボルになり、
BER も第1週・解析解と一致するはずである。
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
N_SYM = 500_000
SNR_DB = 20.0
SEED = 4


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"α={BETA}, {SPS} sps, シンボル数={N_SYM}, SNR={SNR_DB} dB\n")
    print(f"{'方式':>7} {'誤りビット数':>12} {'BER(整形,実測)':>16} "
          f"{'BER(解析解)':>14}")

    results = {}
    for M in QAM_ORDERS:
        k = int(np.log2(M))
        # 送信
        bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)
        tx_sym = comm.bits_to_symbols(bits, M)
        # 整形 -> 雑音 -> ダウンサンプリング
        shaped, delay = comm.pulse_shape(tx_sym, SPS, BETA, SPAN)
        rx = comm.add_awgn(shaped, SNR_DB, signal_power=1.0, rng=rng)
        rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)
        # 復調 (判定) + デマッピング
        rx_bits = comm.symbols_to_bits(rx_sym, M)
        n = min(len(bits), len(rx_bits))
        n_err = comm.count_bit_errors(bits[:n], rx_bits[:n])
        ber = n_err / n
        ber_th = float(comm.ber_theory_qam(M, SNR_DB))
        results[M] = (ber, ber_th)
        print(f"{M:5d}QAM {n_err:12d} {ber:16.3e} {ber_th:14.3e}")

    # --- 図: 整形あり実測 BER vs 解析解 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(QAM_ORDERS))
    ber_meas = [max(results[M][0], 1e-12) for M in QAM_ORDERS]
    ber_th = [results[M][1] for M in QAM_ORDERS]
    ax.bar(x, ber_meas, width=0.5, color="C2", alpha=0.7, label="RC-shaped (measured)")
    ax.plot(x, ber_th, "rD--", label="analytic BER")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([f"{M}QAM" for M in QAM_ORDERS])
    ax.set_ylabel("BER")
    ax.set_title(f"BER of RC-shaped QAM at SNR={SNR_DB:.0f} dB (ISI-free sampling)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "ber_rc.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER図を保存しました: {out}")


if __name__ == "__main__":
    main()
