"""問題3-3: 単一タップ MIMO + LMS による適応等化

問3-2 の「レーザ位相雑音が付加された受信 QAM サンプル」に対し、単一タップ MIMO を
LMS アルゴリズムで適応等化する。単一タップ m の更新式 (問題指定):

    v = m * u                       # 等化器出力
    m = m + mu*(d - v)*conj(u)      # LMS 更新 (d: 参照符号, u: 入力符号)

位相雑音はゆっくりした回転なので、m はその時々の逆回転 e^{-jθ[n]} に追従し、
コンステレーションが元の格子に戻る。
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
LABELS = {4: "QPSK", 16: "16QAM", 64: "64QAM"}
RS = 32e9
DF = 10e3
N_SYM = 200_000          # 等化の収束・追従が見える程度
MU = 0.01                # LMS ステップサイズ
WARMUP = 5000            # 収束前の過渡区間 (図から除外)
SEED = 9


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def lms_single_tap(rx: np.ndarray, ref: np.ndarray, mu: float):
    """単一タップ LMS 等化。返り値 (等化出力 v, タップ履歴 m)。"""
    n = len(rx)
    v_out = np.empty(n, dtype=complex)
    m_hist = np.empty(n, dtype=complex)
    m = 1.0 + 0.0j                       # タップ初期値
    for i in range(n):
        u = rx[i]
        v = m * u                        # 等化器出力
        v_out[i] = v
        d = ref[i]                       # 参照符号 (データ援用)
        m = m + mu * (d - v) * np.conj(u)
        m_hist[i] = m
    return v_out, m_hist


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"符号速度 {RS/1e9:.0f} Gbaud, 線幅 {DF/1e3:.0f} kHz, μ={MU}, "
          f"シンボル数 {N_SYM}\n")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, M in enumerate(QAM_ORDERS):
        tx = make_symbols(M, N_SYM)
        theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)
        rx = tx * np.exp(1j * theta)                       # 位相雑音付き受信

        v, m_hist = lms_single_tap(rx, tx, MU)             # 単一タップ LMS 等化

        # 等化後の判定誤り (収束後の区間で評価)
        rx_bits = comm.symbols_to_bits(v[WARMUP:], M)
        tx_bits = comm.symbols_to_bits(tx[WARMUP:], M)
        nbit = min(len(rx_bits), len(tx_bits))
        ber = comm.count_bit_errors(tx_bits[:nbit], rx_bits[:nbit]) / nbit
        print(f"{LABELS[M]:6s}: 等化後 BER (収束後) = {ber:.2e}")

        # (上) 等化前
        axt = axes[0, col]
        axt.scatter(rx[WARMUP:WARMUP + 6000].real, rx[WARMUP:WARMUP + 6000].imag,
                    s=4, alpha=0.15, color="C3")
        axt.set_title(f"{LABELS[M]} before EQ (phase noise)")
        # (下) 等化後
        axb = axes[1, col]
        axb.scatter(v[WARMUP:WARMUP + 6000].real, v[WARMUP:WARMUP + 6000].imag,
                    s=4, alpha=0.15, color="C0")
        axb.set_title(f"{LABELS[M]} after single-tap LMS")
        for ax in (axt, axb):
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"Single-tap MIMO LMS equalization (μ={MU})", fontsize=14)
    fig.tight_layout()
    out = os.path.join(HERE, "lms_equalization.png")
    fig.savefig(out, dpi=120)
    print(f"\n等化前後のコンステレーションを保存しました: {out}")

    # --- タップ位相が位相雑音の逆回転を追っていることを確認 (16QAM) ---
    fig2, ax = plt.subplots(figsize=(9, 4))
    M = 16
    tx = make_symbols(M, N_SYM)
    theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)
    rx = tx * np.exp(1j * theta)
    _, m_hist = lms_single_tap(rx, tx, MU)
    n0 = 30000
    ax.plot(np.arange(n0), np.unwrap(np.angle(m_hist[:n0])), "C0", lw=1,
            label="tap phase  ∠m[n]")
    ax.plot(np.arange(n0), -theta[:n0], "r--", lw=1, label="−θ[n] (ideal inverse)")
    ax.set_xlabel("symbol index"); ax.set_ylabel("phase [rad]")
    ax.set_title("Single tap tracks the inverse of laser phase noise (16QAM)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
    fig2.tight_layout()
    out2 = os.path.join(HERE, "tap_tracking.png")
    fig2.savefig(out2, dpi=120)
    print(f"タップ追従の図を保存しました: {out2}")


if __name__ == "__main__":
    main()
