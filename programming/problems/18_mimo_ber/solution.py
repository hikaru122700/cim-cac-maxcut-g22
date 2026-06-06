"""問題3-6: 偏波回転+AWGN+位相雑音を受けた両偏波 QAM の 2x2 MIMO 等化と BER

1 sample/symbol の両偏波 QPSK・16QAM。伝送路:
  偏波回転 (SU(2)) -> 複素AWGN -> コヒーレント受信に伴うレーザ位相雑音 (線幅10kHz)
SNR は QPSK:10 dB、16QAM:15 dB。

これを 2x2 MIMO + LMS で適応等化し、等化後 BER を求める。タップ数 1 と 10 を比較する。

1 sample/symbol では伝送路が「瞬時的な偏波混合 + 共通位相回転」だけ (符号間干渉なし) なので、
単一タップ MIMO でも偏波分離・位相追従ができ、BER は解析解に達する。
タップ数を増やしても (余分なタップが雑音を拾うぶん) わずかに劣化する程度。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [(4, 10.0, "QPSK"), (16, 15.0, "16QAM")]
RS = 32e9
DF = 10e3
N_SYM = 200_000
DELAY = 1000
WARMUP = 20_000
TAPS = [1, 10]
MU = {1: 2e-3, 10: 1e-3}
SEED = 12


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def channel(M: int, snr_db: float, rng):
    """送信両偏波信号と、伝送路通過後の受信両偏波信号を返す。"""
    x = make_symbols(M, N_SYM)
    y = np.roll(x, DELAY)
    s = np.vstack([x, y])                                   # 送信 (2,N)

    U = unitary_group.rvs(2, random_state=SEED)
    U = U / np.sqrt(np.linalg.det(U))                       # SU(2) 偏波回転
    r = U @ s
    r = comm.add_awgn(r, snr_db, signal_power=1.0, rng=rng)  # 複素AWGN
    theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)      # 共通レーザ位相雑音
    rx = r * np.exp(1j * theta)
    return s, rx


def ber_after_eq(s, v, valid, M):
    sl = slice(WARMUP, valid.stop)
    errs = tot = 0
    for p in range(2):
        tb = comm.symbols_to_bits(s[p, sl], M)
        rb = comm.symbols_to_bits(v[p, sl], M)
        nn = min(len(tb), len(rb))
        errs += comm.count_bit_errors(tb[:nn], rb[:nn]); tot += nn
    return errs / tot


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"1 sps, 線幅 {DF/1e3:.0f} kHz, シンボル数 {N_SYM}/偏波\n")
    print(f"{'方式':>6} {'SNR':>5} {'タップ':>6} {'等化後BER':>12} {'解析解':>12}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (M, snr, name) in enumerate(CASES):
        s, rx = channel(M, snr, rng)
        ber_th = float(comm.ber_theory_qam(M, snr))

        # 等化前 (x偏波) コンステレーション
        ax0 = axes[row, 0]
        ax0.scatter(rx[0, WARMUP:WARMUP + 5000].real, rx[0, WARMUP:WARMUP + 5000].imag,
                    s=4, alpha=0.15, color="C3")
        ax0.set_title(f"{name} x-pol before EQ")

        for col, L in enumerate(TAPS, start=1):
            v, valid = comm.mimo_lms(rx, s, L, MU[L])
            ber = ber_after_eq(s, v, valid, M)
            print(f"{name:>6} {snr:4.0f}dB {L:6d} {ber:12.3e} {ber_th:12.3e}")
            ax = axes[row, col]
            ax.scatter(v[0, WARMUP:WARMUP + 5000].real, v[0, WARMUP:WARMUP + 5000].imag,
                       s=4, alpha=0.15, color="C0")
            ax.set_title(f"{name} x-pol after MIMO (L={L}, BER={ber:.1e})")

        for ax in axes[row]:
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

    fig.suptitle("2x2 MIMO LMS equalization at 1 sample/symbol", fontsize=14)
    fig.tight_layout()
    out = os.path.join(HERE, "mimo_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
