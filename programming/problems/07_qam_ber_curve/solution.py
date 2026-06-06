"""問題2-1: BER の SNR 依存性 (4/16/64/256 QAM) と解析解との比較

各 QAM 方式について、SNR を変えながらモンテカルロ法で BER を測定し、
横軸 SNR [dB]、縦軸 BER (対数表示) のグラフを描く。
さらに解析解 (近似式) と重ねて比較する。

  BER ≈ (4/k)(1 - 1/√M) Q( sqrt( 3/(M-1) · γ_s ) ),  γ_s = 10^(SNR/10)
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
N_SYM = 500_000          # 1点あたりのシンボル数
SNR_LIST = np.arange(0, 31, 2)   # 測定する SNR [dB]
SEED = 2


def measure_ber(M: int, snr_db: float, rng: np.random.Generator) -> float:
    """ランダムビットを M-QAM で送受信し BER を測定する。"""
    k = int(np.log2(M))
    bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)
    tx = comm.bits_to_symbols(bits, M)
    rx = comm.add_awgn(tx, snr_db, signal_power=1.0, rng=rng)
    rx_bits = comm.symbols_to_bits(rx, M)
    return comm.count_bit_errors(bits, rx_bits) / len(bits)


def main() -> None:
    rng = np.random.default_rng(SEED)
    snr_fine = np.linspace(0, 30, 300)   # 解析解用の細かいSNR軸

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["C0", "C1", "C2", "C3"]

    print(f"シンボル数/点 = {N_SYM}\n")
    for M, c in zip(QAM_ORDERS, colors):
        # --- 解析解 ---
        ber_th = comm.ber_theory_qam(M, snr_fine)
        ax.plot(snr_fine, ber_th, "-", color=c, lw=1.5,
                label=f"{M}QAM theory")

        # --- 実測 (誤りが出た点のみプロット) ---
        meas_snr, meas_ber = [], []
        for snr in SNR_LIST:
            ber = measure_ber(M, snr, rng)
            if ber > 0:
                meas_snr.append(snr)
                meas_ber.append(ber)
        ax.plot(meas_snr, meas_ber, "o", color=c, ms=5, mfc="none",
                label=f"{M}QAM sim")
        print(f"{M:3d}QAM: 測定できた最小BER = {min(meas_ber):.2e} "
              f"(SNR={meas_snr[meas_ber.index(min(meas_ber))]:.0f} dB)")

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.set_xlim(0, 30)
    ax.set_xlabel("SNR per symbol  Es/N0 [dB]")
    ax.set_ylabel("BER")
    ax.set_title("BER vs SNR for square QAM (simulation vs theory)")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(direction="in", which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "ber_vs_snr.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER-SNR曲線を保存しました: {out}")


if __name__ == "__main__":
    main()
