"""問題3-2: レーザ位相雑音を付加した QAM のコンステレーション

  - 1 sample/symbol、符号速度 32 Gbaud (= fs = 32 Gsample/s)。
  - QPSK(4QAM)、16QAM、64QAM、シンボル数 10**6。
  - 線幅 df = 10 kHz のレーザ位相雑音を付加 (コヒーレント受信での送受信レーザ位相揺らぎ)。
      rx[n] = tx[n] · exp( j·θ[n] )
  - コンステレーションを図示する。

位相雑音はゆっくりしたランダムウォークなので、シンボルは複素平面上で少しずつ回転し、
コンステレーションが「円弧状」に広がる(振幅は変わらず位相だけ回る)。
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
QAM_ORDERS = [4, 16, 64]
LABELS = {4: "QPSK (4QAM)", 16: "16QAM", 64: "64QAM"}
RS = 32e9               # 符号速度 [baud] = fs (1 sps)
DF = 10e3              # 線幅 [Hz]
N_SYM = 10 ** 6
SEED = 8


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def main() -> None:
    rng = np.random.default_rng(SEED)
    sigma2 = 2 * np.pi * DF / RS
    print(f"符号速度 Rs = fs = {RS/1e9:.0f} Gbaud, 線幅 df = {DF/1e3:.0f} kHz, "
          f"シンボル数 = {N_SYM}")
    print(f"位相増分の分散 σ_PN^2 = {sigma2:.2e} rad^2")
    # 系列全体での位相の標準偏差の目安
    print(f"系列全体での位相のばらつき ≈ √(N)·σ_PN = "
          f"{np.sqrt(N_SYM*sigma2):.2f} rad\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, M in zip(axes, QAM_ORDERS):
        tx = make_symbols(M, N_SYM)
        theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)   # レーザ位相雑音
        rx = tx * np.exp(1j * theta)                          # 位相回転を付加

        print(f"{LABELS[M]:14s}: 位相 θ の範囲 = [{theta.min():.2f}, {theta.max():.2f}] rad")

        ax.scatter(rx[:6000].real, rx[:6000].imag, s=4, alpha=0.15)
        ax.set_title(f"{LABELS[M]} + phase noise (df={DF/1e3:.0f} kHz)")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"Constellations with laser phase noise (1 sps, N={N_SYM:,})", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation_pn.png")
    fig.savefig(out, dpi=120)
    print(f"\nコンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
