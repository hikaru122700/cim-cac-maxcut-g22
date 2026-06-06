"""問題1-6: ビット誤り率 (BER) の測定と処理時間の計測

問1-2〜1-5 を1本につなげた完全な送受信チェーン:
  PRBS生成 -> QAMマッピング -> AWGN付加 -> 判定 -> QAMデマッピング -> 誤り計数

各 QAM 方式について
  - 誤りビット数
  - ビット誤り率 BER = (誤りビット数) / (総ビット数)
を求め、解析解と比較する。さらに time ライブラリで処理時間を測る
(波形プロットは計測対象に含めない)。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
QAM_ORDERS = [4, 16, 64, 256]
N_SYM = 10 ** 6
SNR_DB = 20.0
SEED = 1


def run_chain(M: int, rng: np.random.Generator):
    """1方式ぶんの送受信チェーンを実行し、(誤り数, 総ビット, BER, SER) を返す。"""
    k = int(np.log2(M))

    # --- 送信ビット列 (PRBS を繰り返し N_SYM シンボル分) ---
    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(N_SYM / len(block)))
    tx_sym = np.tile(block, reps)[:N_SYM]
    tx_bits = comm.symbols_to_bits(tx_sym, M)        # 送信ビット (基準)

    # --- 伝送 ---
    rx_sym = comm.add_awgn(tx_sym, SNR_DB, signal_power=1.0, rng=rng)
    rx_bits = comm.symbols_to_bits(rx_sym, M)        # 判定 + デマッピング

    # --- 誤り計数 ---
    n_bit_err = comm.count_bit_errors(tx_bits, rx_bits)
    ber = n_bit_err / len(tx_bits)
    ser = np.count_nonzero(~np.isclose(comm.decide(rx_sym, M), tx_sym)) / N_SYM
    return n_bit_err, len(tx_bits), ber, ser


def main() -> None:
    rng = np.random.default_rng(SEED)

    results = {}
    print(f"シンボル数={N_SYM}, SNR={SNR_DB} dB\n")
    print(f"{'方式':>7} {'誤りビット数':>12} {'総ビット数':>12} "
          f"{'BER(実測)':>12} {'BER(解析解)':>12} {'SER/k':>10}")

    # ---- 処理時間の計測開始 (プロットは含めない) ----
    t_start = time.time()
    for M in QAM_ORDERS:
        n_err, n_bits, ber, ser = run_chain(M, rng)
        results[M] = (ber, ser)
        k = int(np.log2(M))
        ber_th = float(comm.ber_theory_qam(M, SNR_DB))
        print(f"{M:5d}QAM {n_err:12d} {n_bits:12d} "
              f"{ber:12.3e} {ber_th:12.3e} {ser / k:10.3e}")
    t_end = time.time()
    # ---- 計測終了 ----
    print(f"\nRequired time for programming: {t_end - t_start:.3f} s "
          f"(波形プロットを除く)")

    # --- 図: 実測BER vs 解析解 (棒 + 点) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(QAM_ORDERS))
    ber_meas = [max(results[M][0], 1e-12) for M in QAM_ORDERS]
    ber_theory = [float(comm.ber_theory_qam(M, SNR_DB)) for M in QAM_ORDERS]
    ax.bar(x, ber_meas, width=0.5, color="C0", alpha=0.7, label="measured BER")
    ax.plot(x, ber_theory, "rD--", label="analytic BER")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{M}QAM" for M in QAM_ORDERS])
    ax.set_ylabel("BER")
    ax.set_title(f"BER at SNR = {SNR_DB:.0f} dB ({N_SYM:,} symbols)")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(direction="in")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "ber.png")
    fig.savefig(out, dpi=120)
    print(f"BER図を保存しました: {out}")


if __name__ == "__main__":
    main()
