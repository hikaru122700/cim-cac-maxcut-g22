"""問題1-4: 受信 QAM サンプルの判定 (decision)

AWGN が付加された各受信サンプルについて、参照シンボル (マッピングに用いた M 個の
コンステレーション点) からユークリッド距離が最も近いシンボルを選ぶ。
この最近傍選択を「判定 (decision)」と呼ぶ。

方形QAMでは、判定は I 軸・Q 軸それぞれを最も近い PAM レベルへ丸めるだけでよい
(2次元の総当たりは不要)。共通ライブラリ comm.decide がこれを行う。
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
N_SYM = 10 ** 6
SNR_DB = 20.0
SEED = 1


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    sym_block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(sym_block)))
    return np.tile(sym_block, reps)[:n_sym]


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"シンボル数={N_SYM}, SNR={SNR_DB} dB")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        tx = make_symbols(M, N_SYM)                # 送信シンボル
        rx = comm.add_awgn(tx, SNR_DB, signal_power=1.0, rng=rng)  # 受信 (雑音付き)
        dec = comm.decide(rx, M)                    # 判定 (最近傍シンボル)

        # シンボル誤り率 (SER): 送信と判定が異なる割合
        sym_err = np.count_nonzero(~np.isclose(dec, tx))
        ser = sym_err / N_SYM
        print(f"{M:3d}QAM: シンボル誤り数={sym_err:7d}, SER={ser:.3e}")

        # 受信点を判定後の格子点で色分けして表示 (先頭 4000 点)
        n_plot = 4000
        ax.scatter(rx[:n_plot].real, rx[:n_plot].imag, s=4, alpha=0.12,
                   color="C0", label="received")
        # 参照シンボル (判定の基準点) を重ねる
        ref = comm.qam_constellation(M)
        ax.scatter(ref.real, ref.imag, s=40, marker="x", color="red",
                   label="reference symbols")
        ax.set_title(f"{M}QAM decision (SER = {ser:.1e})")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
        lim = 1.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        if M == 4:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Hard decision against reference constellation", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "decision.png")
    fig.savefig(out, dpi=120)
    print(f"判定図を保存しました: {out}")


if __name__ == "__main__":
    main()
