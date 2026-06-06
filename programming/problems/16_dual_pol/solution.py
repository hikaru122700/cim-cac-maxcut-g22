"""問題3-4: 両偏波 QAM シンボル列の生成

1 sample/symbol で両偏波 QAM シンボル列を作る。
  - x偏波: 問3-2 と同じ QAM シンボル列。
  - y偏波: それに任意の遅延 (ここでは 1000 サンプル) を与えたもの。
  - 2次元 numpy 配列 (2, 10**6) にまとめる。

光ファイバは直交する2つの偏波 (x, y) に独立な信号を多重できる (偏波多重 = PDM)。
これにより同じ帯域で伝送容量が2倍になる。y を x の遅延版にするのは、
両偏波に「無相関なデータ」を載せた状況を簡単に作るための便法。
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
M = 16                  # 代表として16QAM
N_SYM = 10 ** 6
DELAY = 1000            # y偏波に与える遅延 [サンプル]
SEED = 10


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def main() -> None:
    x = make_symbols(M, N_SYM)               # x偏波 (問3-2のQAMシンボル)
    y = np.roll(x, DELAY)                     # y偏波 = x を DELAY サンプル遅延

    dual = np.vstack([x, y])                  # (2, N) の両偏波配列
    print(f"両偏波配列の形状 : {dual.shape}  (dtype={dual.dtype})")
    print(f"x偏波平均電力     : {np.mean(np.abs(dual[0])**2):.4f}")
    print(f"y偏波平均電力     : {np.mean(np.abs(dual[1])**2):.4f}")

    # x と y の相関 (遅延により無相関化されているか)
    corr = np.abs(np.vdot(dual[0], dual[1])) / N_SYM
    print(f"x·y* の正規化相関 : {corr:.4f}  (≈0 なら無相関)")
    print(f"y偏波は x偏波の {DELAY} サンプル遅延版: "
          f"{np.array_equal(dual[1][DELAY:DELAY+10], dual[0][:10])}")

    # --- 図: x偏波・y偏波それぞれのコンステレーション ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, pol, name in zip(axes, dual, ["x-pol", "y-pol"]):
        ax.scatter(pol[:5000].real, pol[:5000].imag, s=6, alpha=0.25)
        ax.set_title(f"{name} ({M}QAM)")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
        ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    fig.suptitle(f"Dual-polarization QAM (shape {dual.shape}, y = x delayed by {DELAY})",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "dual_pol.png")
    fig.savefig(out, dpi=120)
    print(f"\n両偏波コンステレーションを保存しました: {out}")


if __name__ == "__main__":
    main()
