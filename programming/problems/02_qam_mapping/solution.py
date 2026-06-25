"""問題1-2: PRBS から QAM シンボル列への変換 (QAMマッピング)

【この問題で学ぶこと】
  ディジタル変調の基本である QAM (直交振幅変調) のマッピングを体験する。
  QAM は 1 シンボルに複数ビットを載せる変調で、I 軸 (実部) と Q 軸 (虚部) の
  振幅の組み合わせで情報を表す。多値数 M を大きくするほど 1 シンボルあたりの
  ビット数 k=log2(M) が増え、伝送効率が上がる代わりに点の間隔が狭くなり
  雑音に弱くなる (この弱さは次の問題1-3で確認する)。

【手順】
  1. 周期 2**15 - 1 = 32767 ビットの PRBS を 1 周期生成する。
  2. これを k = log2(QAM多値数) 回だけ繰り返してビット列を作る
     (4QAM:k=2, 16QAM:k=4, 64QAM:k=6, 256QAM:k=8)。
     -> ビット数 = 32767 * k となり、k ビットで 1 シンボルなのでシンボル数は 32767。
  3. グレイ符号化した方形QAMへマッピングし、シンボル数 32767 のシンボル列を得る。
  4. 平均強度 (平均シンボル電力) が 1 になるよう規格化する。
  5. コンステレーション (複素平面上の点の散布図) を確認する。

このスクリプトは comm.py (共通通信ライブラリ) の関数を呼び出して実装する。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 隣の _common フォルダにある comm.py を import できるよう検索パスに追加する
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
QAM_ORDERS = [4, 16, 64, 256]   # 図示する QAM 多値数 (上から 4/16/64/256QAM)


def main() -> None:
    # comm.generate_prbs(15): 15段 LFSR で M系列 PRBS を 1 周期生成 (長さ 32767 の 0/1 系列)
    prbs = comm.generate_prbs(15)                  # 長さ 32767 の 0/1 系列
    n_sym_target = len(prbs)                        # 各QAMでシンボル数を 32767 に揃える
    print(f"PRBS 周期        : {len(prbs)} ビット")

    # 2x2 のサブプロットに 4 種類の QAM コンステレーションを並べる
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        k = int(np.log2(M))                        # 1シンボルあたりビット数 (=繰り返し数)
        bits = np.tile(prbs, k)                     # PRBS を k 回繰り返してビット数を k 倍に
        bits = bits[: n_sym_target * k]            # 念のため長さを 32767*k に揃える
        # comm.bits_to_symbols: k ビットずつ取り出してグレイ符号の方形QAM点に写像し、
        #   平均電力 1 へ規格化した複素シンボル列を返す (QAM 変調器の中核)
        sym = comm.bits_to_symbols(bits, M)        # QAMマッピング (規格化済み)

        # 規格化が効いて平均シンボル電力 ≈ 1 になっているか確認 (M によらず約 1 になるはず)
        avg_power = np.mean(np.abs(sym) ** 2)
        # I 軸 (実部) に現れる振幅レベルの種類数。方形QAMなら √M に一致するはず
        #   (4QAM->2, 16QAM->4, 64QAM->8, 256QAM->16)
        n_levels = len(np.unique(np.round(sym.real, 6)))
        print(f"{M:3d}QAM: ビット/シンボル={k}, シンボル数={len(sym)}, "
              f"平均電力={avg_power:.4f}, I軸レベル数={n_levels}")

        # コンステレーション: 各シンボルを複素平面 (横=I, 縦=Q) の点として散布図表示
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
    # この solution.py と同じフォルダに constellation.png として保存
    out = os.path.join(HERE, "constellation.png")
    fig.savefig(out, dpi=120)
    print(f"コンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
