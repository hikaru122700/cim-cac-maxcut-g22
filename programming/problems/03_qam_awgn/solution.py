"""問題1-3: QAM シンボル列への白色ガウス雑音 (AWGN) 付加

手順:
  1. 問1-2 と同じ手順で QAM シンボルを作り、繰り返してシンボル数を 10**6 に増やす。
  2. SNR = 20 dB に相当する複素 AWGN を付加する。
     (平均シンボル電力 1 に対し、雑音電力 = 1 / 10**(20/10) = 0.01)
  3. 雑音付加後のコンステレーションを図示する。
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
N_SYM = 10 ** 6           # シンボル数
SNR_DB = 20.0             # 信号電力対雑音電力比 [dB]
SEED = 1                  # 再現性のための乱数シード


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から M-QAM シンボルを作り、繰り返して n_sym シンボルに増やす。"""
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    bits = np.tile(prbs, k)                       # 1ブロック = 32767 シンボル分
    sym_block = comm.bits_to_symbols(bits, M)
    reps = int(np.ceil(n_sym / len(sym_block)))
    sym = np.tile(sym_block, reps)[:n_sym]
    return sym


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"シンボル数 : {N_SYM}")
    print(f"SNR        : {SNR_DB} dB  (雑音電力 = {10 ** (-SNR_DB / 10):.4f})")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    n_plot = 4000                                 # 図に描く点数 (全点だと真っ黒になる)

    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        sym = make_symbols(M, N_SYM)
        rx = comm.add_awgn(sym, SNR_DB, signal_power=1.0, rng=rng)

        meas_snr = 10 * np.log10(np.mean(np.abs(sym) ** 2)
                                 / np.mean(np.abs(rx - sym) ** 2))
        print(f"{M:3d}QAM: 実測SNR={meas_snr:5.2f} dB")

        ax.scatter(rx[:n_plot].real, rx[:n_plot].imag, s=4, alpha=0.15)
        ax.set_title(f"{M}QAM + AWGN (SNR = {SNR_DB:.0f} dB)")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
        lim = 1.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle(f"Received constellations with AWGN (N = {N_SYM:,} symbols)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation_awgn.png")
    fig.savefig(out, dpi=120)
    print(f"雑音付加コンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
