"""問題3-5: 両偏波 QAM 信号への偏波回転 (SU(2)) の付加

光ファイバ伝送では2偏波が結合する (偏波回転)。これは 2×2 の特殊ユニタリ行列 SU(2) で
モデル化できる。scipy.stats.unitary_group.rvs(2) でランダムユニタリ行列 U を作り、
行列式が 1 になるよう正規化して SU(2) 行列にする:

    U_su = U / sqrt(det(U))          # det(U_su) = 1

偏波回転後の信号は r[n] = U_su · s[n]  (s は (2,1) の両偏波ベクトル)。
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
M = 16
N_SYM = 10 ** 6
DELAY = 1000
SEED = 11


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))
    return np.tile(block, reps)[:n_sym]


def make_su2(rng_seed: int) -> np.ndarray:
    """ランダムな SU(2) (特殊ユニタリ) 偏波回転行列を作る。"""
    U = unitary_group.rvs(2, random_state=rng_seed)      # ユニタリ行列
    U_su = U / np.sqrt(np.linalg.det(U))                  # det=1 に正規化 -> SU(2)
    return U_su


def main() -> None:
    # 両偏波信号 (問3-4)
    x = make_symbols(M, N_SYM)
    y = np.roll(x, DELAY)
    s = np.vstack([x, y])                                 # (2, N)

    U = make_su2(SEED)
    print("偏波回転行列 U (SU(2)):")
    print(U)
    print(f"\nユニタリ性 U·U^H = I : {np.allclose(U @ U.conj().T, np.eye(2))}")
    print(f"行列式 det(U)        : {np.linalg.det(U):.6f}  (|det|=1, 実質1)")

    # 偏波回転を付加: r = U·s  (各時刻 (2,1) ベクトルに行列を掛ける)
    r = U @ s                                             # (2, N)

    # 全電力 (2偏波合計) はユニタリ変換で不変
    p_in = np.mean(np.abs(s[0]) ** 2 + np.abs(s[1]) ** 2)
    p_out = np.mean(np.abs(r[0]) ** 2 + np.abs(r[1]) ** 2)
    print(f"\n回転前 2偏波合計平均電力 : {p_in:.4f}")
    print(f"回転後 2偏波合計平均電力 : {p_out:.4f}  (ユニタリなので不変)")

    # --- 図: 回転前後の各偏波コンステレーション ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    data = [(s, "before rotation"), (r, "after rotation")]
    for row, (sig, tag) in enumerate(data):
        for col, name in enumerate(["x-pol", "y-pol"]):
            ax = axes[row, col]
            ax.scatter(sig[col][:5000].real, sig[col][:5000].imag, s=5, alpha=0.2,
                       color="C0" if row == 0 else "C1")
            ax.set_title(f"{name} {tag}")
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    fig.suptitle("Polarization rotation by SU(2): two QAM grids get mixed", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "pol_rotation.png")
    fig.savefig(out, dpi=120)
    print(f"\n偏波回転前後の図を保存しました: {out}")


if __name__ == "__main__":
    main()
