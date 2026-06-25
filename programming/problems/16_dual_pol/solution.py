"""問題3-4: 両偏波 QAM シンボル列の生成

1 sample/symbol で両偏波 QAM シンボル列を作る。
  - x偏波: 問3-2 と同じ QAM シンボル列。
  - y偏波: それに任意の遅延 (ここでは 1000 サンプル) を与えたもの。
  - 2次元 numpy 配列 (2, 10**6) にまとめる。

【背景: 偏波多重 (PDM) とは】
光ファイバの中を進む光は、互いに直交する2つの偏波 (x偏波・y偏波) を持つ。
この2つの偏波はそれぞれ独立した「通り道」として使えるので、同じ波長・同じ帯域でも
別々のデータを2本同時に流せる (= 偏波多重 = PDM)。結果として伝送容量が2倍になる。

【なぜ y を x の遅延版にするのか】
本来 x偏波・y偏波には互いに無関係な (無相関な) データを載せる。ここでは簡単のため、
y偏波を「x偏波を 1000 サンプルずらしただけのもの」にしている。十分大きく遅延させると、
同時刻の x[n] と y[n] は元々別の場所のシンボルなので統計的に無相関になり、
「2偏波に独立データを載せた」状況を手軽に再現できる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 共通ライブラリ comm.py を import できるよう、隣の _common フォルダを検索パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))  # このファイルのあるフォルダ (図の保存先)
M = 16                  # QAM の多値数。代表として 16QAM (1シンボル=4ビット)
N_SYM = 10 ** 6         # 生成するシンボル数 (偏波あたり 100 万シンボル)
DELAY = 1000            # y偏波に与える遅延 [サンプル]
SEED = 10               # 乱数シード (この問題では再現性の参考値)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から長さ n_sym の規格化 M-QAM シンボル列を作る。

    通信路評価用の試験信号を作るヘルパー。PRBS (擬似ランダムビット列) を
    QAM シンボルに変調し、必要な長さになるまで繰り返してから切り出す。
    """
    # M系列 (周期 2^15-1 = 32767 の擬似ランダムビット列) を1周期分生成
    prbs = comm.generate_prbs(15)
    k = int(np.log2(M))                              # 1シンボルあたりのビット数 (16QAM なら 4)
    # PRBS を k 倍に並べてビット数を log2(M) の倍数に揃え、QAM シンボルへ変調 (1 block 分)
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))         # 必要数に届くまでの繰り返し回数
    # block を reps 回タイル状に並べ、先頭 n_sym シンボルだけを取り出して返す
    return np.tile(block, reps)[:n_sym]


def main() -> None:
    # --- 両偏波信号の生成 ---
    x = make_symbols(M, N_SYM)               # x偏波 (問3-2のQAMシンボル列)
    # np.roll で配列全体を DELAY だけ巡回シフト → x を DELAY サンプル遅らせた系列を y偏波に
    y = np.roll(x, DELAY)                     # y偏波 = x を DELAY サンプル遅延

    # x偏波を 0 行目、y偏波を 1 行目に縦積みし、(2, N) の両偏波配列にまとめる
    dual = np.vstack([x, y])                  # (2, N) の両偏波配列 (PDM 信号の標準的な表現)
    # 配列の形状と要素型を確認 (2 行 = 2偏波、N 列 = シンボル数、dtype は複素数)
    print(f"両偏波配列の形状 : {dual.shape}  (dtype={dual.dtype})")
    # 各偏波の平均電力 = 平均 |シンボル|^2。bits_to_symbols が電力1へ規格化済みなので ≈1 になる
    print(f"x偏波平均電力     : {np.mean(np.abs(dual[0])**2):.4f}")
    print(f"y偏波平均電力     : {np.mean(np.abs(dual[1])**2):.4f}")

    # x と y の相関 (遅延により無相関化されているか確認)
    # np.vdot(a,b) = Σ conj(a)·b。これを N で割ると正規化相関。十分遅延させた別データなので ≈0
    corr = np.abs(np.vdot(dual[0], dual[1])) / N_SYM
    print(f"x·y* の正規化相関 : {corr:.4f}  (≈0 なら無相関)")
    # y偏波の DELAY 番目以降の先頭10個が、x偏波の先頭10個と一致するか = 確かに遅延版か検証
    print(f"y偏波は x偏波の {DELAY} サンプル遅延版: "
          f"{np.array_equal(dual[1][DELAY:DELAY+10], dual[0][:10])}")

    # --- 図: x偏波・y偏波それぞれのコンステレーション (IQ 平面の点群) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))  # 左右2枚 (左=x偏波, 右=y偏波)
    for ax, pol, name in zip(axes, dual, ["x-pol", "y-pol"]):
        # 横軸=実部(I, 同相成分)、縦軸=虚部(Q, 直交成分)。先頭5000点だけ散布図に
        ax.scatter(pol[:5000].real, pol[:5000].imag, s=6, alpha=0.25)
        ax.set_title(f"{name} ({M}QAM)")
        ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")  # I/Q 等倍で格子が正方に
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
