"""問題3-5: 両偏波 QAM 信号への偏波回転 (SU(2)) の付加

【偏波回転とは】
光ファイバを伝搬する間に、x偏波と y偏波は機械的応力や曲げなどで互いに混ざり合う
(= 偏波回転)。受信側で見ると、送った x偏波・y偏波が「混ざった」状態で届く。
この混ざり合いは、電力を保存する (= エネルギーが増減しない) 線形変換なので、
2×2 のユニタリ行列で表せる。さらに位相の自由度を1つ固定すると、行列式が 1 の
特殊ユニタリ行列 SU(2) でモデル化できる。

【SU(2) 行列の作り方】
scipy.stats.unitary_group.rvs(2) でランダムな 2×2 ユニタリ行列 U を作り、
行列式が 1 になるよう正規化して SU(2) 行列にする:

    U_su = U / sqrt(det(U))          # det(U_su) = 1

【偏波回転の適用】
偏波回転後の信号は各時刻ごとに r[n] = U_su · s[n] (s は (2,1) の両偏波ベクトル)。
配列全体では r = U_su @ s と一括で計算できる。
ユニタリ変換なので、2偏波の合計電力は回転の前後で変わらない (エネルギー保存)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# 共通ライブラリ comm.py を import できるよう、隣の _common フォルダを検索パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))  # このファイルのあるフォルダ (図の保存先)
M = 16                  # QAM の多値数 (16QAM)
N_SYM = 10 ** 6         # 偏波あたりのシンボル数
DELAY = 1000            # y偏波に与える遅延 [サンプル] (問3-4と同じく無相関化のため)
SEED = 11               # SU(2) 行列を作る乱数シード (再現性確保)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から長さ n_sym の規格化 M-QAM シンボル列を作る (問3-4と同じ)。"""
    prbs = comm.generate_prbs(15)                   # M系列 (擬似ランダムビット列) を生成
    k = int(np.log2(M))                             # 1シンボルあたりビット数
    block = comm.bits_to_symbols(np.tile(prbs, k), M)  # ビット列を QAM シンボルへ変調
    reps = int(np.ceil(n_sym / len(block)))         # 必要数に届くまでの繰り返し回数
    return np.tile(block, reps)[:n_sym]             # 繰り返して先頭 n_sym だけ切り出す


def make_su2(rng_seed: int) -> np.ndarray:
    """ランダムな SU(2) (特殊ユニタリ) 偏波回転行列を作る。

    伝送路の偏波混合をモデル化する 2×2 行列を1つ返す。ユニタリ行列をランダムに
    引いてから、行列式が 1 になるよう正規化することで SU(2) に落とす。
    """
    # scipy で 2×2 のランダムユニタリ行列を生成 (U·U^H = I を満たす)
    U = unitary_group.rvs(2, random_state=rng_seed)      # ユニタリ行列
    # 行列式 det(U) (絶対値1の複素数) の平方根で割ると det=1 になり SU(2) になる
    U_su = U / np.sqrt(np.linalg.det(U))                  # det=1 に正規化 -> SU(2)
    return U_su


def main() -> None:
    # --- 送信側: 両偏波信号 (問3-4と同じ構成) ---
    x = make_symbols(M, N_SYM)                            # x偏波の QAM シンボル列
    y = np.roll(x, DELAY)                                 # y偏波 = x を DELAY 遅延 (無相関化)
    s = np.vstack([x, y])                                 # 送信両偏波ベクトル (2, N)

    # --- 伝送路: 偏波回転行列を生成 ---
    U = make_su2(SEED)                                    # ランダムな SU(2) 偏波回転行列
    print("偏波回転行列 U (SU(2)):")
    print(U)
    # ユニタリ性の確認: U·U^H が単位行列 I なら正しくユニタリ (電力を保存する変換)
    print(f"\nユニタリ性 U·U^H = I : {np.allclose(U @ U.conj().T, np.eye(2))}")
    # 行列式の確認: SU(2) なので det(U) は (数値誤差を除き) 1 のはず
    print(f"行列式 det(U)        : {np.linalg.det(U):.6f}  (|det|=1, 実質1)")

    # 偏波回転を付加: r = U·s  (各時刻の (2,1) 偏波ベクトルに行列 U を掛ける)
    # 行列積 @ により全 N 時刻をまとめて計算。これで x偏波・y偏波が混ざった受信信号になる
    r = U @ s                                             # (2, N) 偏波回転後の信号

    # 全電力 (2偏波合計) はユニタリ変換で不変であることを確認
    # |r_x|^2 + |r_y|^2 の平均は |s_x|^2 + |s_y|^2 の平均と一致する (エネルギー保存)
    p_in = np.mean(np.abs(s[0]) ** 2 + np.abs(s[1]) ** 2)   # 回転前の2偏波合計平均電力
    p_out = np.mean(np.abs(r[0]) ** 2 + np.abs(r[1]) ** 2)  # 回転後の2偏波合計平均電力
    print(f"\n回転前 2偏波合計平均電力 : {p_in:.4f}")
    print(f"回転後 2偏波合計平均電力 : {p_out:.4f}  (ユニタリなので不変)")

    # --- 図: 回転前後の各偏波コンステレーション (2行×2列) ---
    # 上段 = 回転前 (きれいな QAM 格子)、下段 = 回転後 (2偏波が混ざり格子が崩れる) を比較
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    data = [(s, "before rotation"), (r, "after rotation")]
    for row, (sig, tag) in enumerate(data):
        for col, name in enumerate(["x-pol", "y-pol"]):  # 左列=x偏波, 右列=y偏波
            ax = axes[row, col]
            # 横軸=I (実部)、縦軸=Q (虚部)。先頭5000点を散布図に (回転前は青, 回転後はオレンジ)
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
