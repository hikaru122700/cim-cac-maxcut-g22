"""問題2-5: スペクトル整形 QAM サンプル列に対する BER の SNR 依存性

問2-1 と同じ BER-SNR 曲線を、今度は raised cosine 整形 + ISI-free ダウンサンプリングの
チェーンで測定し、解析解と比較する。Nyquist パルスなので整形なし(問2-1)と同じ曲線になるはず。
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
N_SYM = 300_000
SNR_LIST = np.arange(0, 31, 2)
SEED = 5


def measure_ber_shaped(M: int, snr_db: float, rng: np.random.Generator) -> float:
    k = int(np.log2(M))
    bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)
    tx = comm.bits_to_symbols(bits, M)
    shaped, delay = comm.pulse_shape(tx, SPS, BETA, SPAN)
    rx = comm.add_awgn(shaped, snr_db, signal_power=1.0, rng=rng)
    rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)
    rx_bits = comm.symbols_to_bits(rx_sym, M)
    n = min(len(bits), len(rx_bits))
    return comm.count_bit_errors(bits[:n], rx_bits[:n]) / n


def main() -> None:
    rng = np.random.default_rng(SEED)
    snr_fine = np.linspace(0, 30, 300)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["C0", "C1", "C2", "C3"]
    print(f"α={BETA}, {SPS} sps, シンボル数/点={N_SYM}\n")

    for M, c in zip(QAM_ORDERS, colors):
        ax.plot(snr_fine, comm.ber_theory_qam(M, snr_fine), "-", color=c, lw=1.5,
                label=f"{M}QAM theory")
        ms, mb = [], []
        for snr in SNR_LIST:
            ber = measure_ber_shaped(M, snr, rng)
            if ber > 0:
                ms.append(snr); mb.append(ber)
        ax.plot(ms, mb, "s", color=c, ms=5, mfc="none", label=f"{M}QAM RC-sim")
        print(f"{M:3d}QAM: 測定できた最小BER = {min(mb):.2e}")

    ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.set_xlim(0, 30)
    ax.set_xlabel("SNR per symbol  Es/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title("BER vs SNR for RC-shaped QAM (α=1, 2 sps) vs theory")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "ber_vs_snr_rc.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER-SNR曲線(整形)を保存しました: {out}")


if __name__ == "__main__":
    main()
