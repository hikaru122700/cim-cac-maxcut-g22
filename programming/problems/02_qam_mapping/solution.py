"""問題1-2: PRBS から QAM シンボル列への変換 (QAMマッピング)

手順:
  1. 周期 2**15 - 1 = 32767 ビットの PRBS を 1 周期生成する。
  2. これを k = log2(QAM多値数) 回だけ繰り返してビット列を作る
     (4QAM:k=2, 16QAM:k=4, 64QAM:k=6, 256QAM:k=8)。
  3. グレイ符号化した方形QAMへマッピングし、シンボル数 32767 のシンボル列を得る。
  4. 平均強度 (平均シンボル電力) が 1 になるよう規格化する。
  5. コンステレーション (複素平面上の点の散布図) を確認する。
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


def main() -> None:
    prbs = comm.generate_prbs(15)                  # 長さ 32767 の 0/1 系列
    n_sym_target = len(prbs)                        # 各QAMでシンボル数を 32767 に揃える
    print(f"PRBS 周期        : {len(prbs)} ビット")

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        k = int(np.log2(M))                        # 1シンボルあたりビット数 (=繰り返し数)
        bits = np.tile(prbs, k)                     # PRBS を k 回繰り返す
        bits = bits[: n_sym_target * k]            # 念のため長さを揃える
        sym = comm.bits_to_symbols(bits, M)        # QAMマッピング (規格化済み)

        avg_power = np.mean(np.abs(sym) ** 2)
        n_levels = len(np.unique(np.round(sym.real, 6)))
        print(f"{M:3d}QAM: ビット/シンボル={k}, シンボル数={len(sym)}, "
              f"平均電力={avg_power:.4f}, I軸レベル数={n_levels}")

        ax.scatter(sym.real, sym.imag, s=6, alpha=0.25)
        ax.set_title(f"{M}QAM  ({k} bits/symbol)")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
        lim = 1.6
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle("QAM constellations from PRBS15 (average power = 1)", fontsize=14)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation.png")
    fig.savefig(out, dpi=120)
    print(f"コンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
