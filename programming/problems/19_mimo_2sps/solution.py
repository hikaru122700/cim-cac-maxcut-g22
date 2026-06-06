"""問題3-7: 2 sample/symbol での 2x2 MIMO 等化 (単一タップ vs 10タップ)

問3-6 と同じ伝送路を、2 sample/symbol・パルス整形係数 α=0 で行う。受信側の
サンプリング位相が symbol 中心から半シンボル (T/2 = 1サンプル) ずれている状況を考える。

  - 単一タップ MIMO: タイミングずれを補えず、α=0 の長い sinc 裾による符号間干渉(ISI)で BER が大きく劣化。
  - 10タップ分数間隔 MIMO: 整合フィルタ + タイミング補償 + 偏波分離をまとめて学習し、
    BER が解析解に近づく。

これにより「2 sps では多タップ MIMO が必要」であることを示す。
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
N_SYM = 120_000
SPS = 2
BETA = 0.0              # α=0 (sinc パルス, 裾が長い)
SPAN = 16
DELAY = 1000
T_OFFSET = 1           # サンプリング位相のずれ [サンプル] (=T/2)
WARMUP = 20_000
SEED = 13
CONFIGS = [(1, 1e-3), (10, 5e-4)]    # (タップ数, μ)


def make_symbols(M, n):
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    b = comm.bits_to_symbols(np.tile(prbs, k), M)
    return np.tile(b, int(np.ceil(n / len(b))))[:n]


def channel(M, snr_db, rng):
    x = make_symbols(M, N_SYM)
    y = np.roll(x, DELAY)
    s = np.vstack([x, y])
    # 2 sps パルス整形 (両偏波)
    shx, delay = comm.pulse_shape(s[0], SPS, BETA, SPAN)
    shy, _ = comm.pulse_shape(s[1], SPS, BETA, SPAN)
    sh = np.vstack([shx, shy])
    # 偏波回転 -> AWGN -> 共通位相雑音
    U = unitary_group.rvs(2, random_state=SEED)
    U = U / np.sqrt(np.linalg.det(U))
    r = U @ sh
    r = comm.add_awgn(r, snr_db, signal_power=1.0, rng=rng)
    theta = comm.laser_phase_noise(r.shape[1], DF, RS, rng)
    rx = r * np.exp(1j * theta)
    return s, rx, delay


def ber_of(s, v, kmax, M):
    sl = slice(WARMUP, kmax)
    errs = tot = 0
    for p in range(2):
        tb = comm.symbols_to_bits(s[p, sl], M)
        rb = comm.symbols_to_bits(v[p, sl], M)
        nn = min(len(tb), len(rb))
        errs += comm.count_bit_errors(tb[:nn], rb[:nn]); tot += nn
    return errs / tot


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"2 sps, α={BETA}, サンプリング位相ずれ {T_OFFSET} サンプル(=T/2), "
          f"線幅 {DF/1e3:.0f} kHz\n")
    print(f"{'方式':>6} {'SNR':>6} {'タップ':>6} {'等化後BER':>12} {'解析解':>12}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (M, snr, name) in enumerate(CASES):
        s, rx, delay = channel(M, snr, rng)
        ber_th = float(comm.ber_theory_qam(M, snr))

        # 等化前: ずれた位相で symbol レート抽出した点 (x偏波)
        ax0 = axes[row, 0]
        idx = delay + T_OFFSET + np.arange(N_SYM) * SPS
        idx = idx[idx < rx.shape[1]]
        pre = rx[0, idx]
        ax0.scatter(pre[WARMUP:WARMUP + 4000].real, pre[WARMUP:WARMUP + 4000].imag,
                    s=4, alpha=0.15, color="C3")
        ax0.set_title(f"{name} before EQ (timing-offset sampling)")

        for col, (L, mu) in enumerate(CONFIGS, start=1):
            base = delay - L // 2 + T_OFFSET
            v, kmax = comm.mimo_lms_fse(rx, s, L, mu, SPS, base)
            ber = ber_of(s, v, kmax, M)
            print(f"{name:>6} {snr:5.0f}dB {L:6d} {ber:12.3e} {ber_th:12.3e}")
            ax = axes[row, col]
            ax.scatter(v[0, WARMUP:WARMUP + 4000].real, v[0, WARMUP:WARMUP + 4000].imag,
                       s=4, alpha=0.15, color="C0")
            ax.set_title(f"{name} after MIMO L={L} (BER={ber:.1e})")

        for ax in axes[row]:
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

    fig.suptitle("2 sps MIMO: single tap fails, 10 taps recovers (α=0, T/2 timing offset)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "mimo_2sps.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
