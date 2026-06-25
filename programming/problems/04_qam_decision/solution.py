"""問題1-4: 受信 QAM サンプルの判定 (decision)

【この問題で学ぶこと】
雑音 (AWGN) によって理想点からずれた受信サンプルを、「最も近いコンステレーション
点 (理想シンボル)」へ割り当て直す操作=判定 (decision / 硬判定) を理解する。
受信機はどのシンボルが送られたか分からないので、観測した受信点に最も近い候補を
「そのシンボルが送られたはず」と推定する。これが復調の第一歩になる。

【判定の原理】
各受信サンプルについて、参照シンボル (マッピングに用いた M 個のコンステレーション
点) からユークリッド距離が最も近いシンボルを選ぶ。この最近傍選択が「判定」。

方形QAMでは I 軸 (実部) と Q 軸 (虚部) が独立な格子なので、判定は I 軸・Q 軸
それぞれを最も近い PAM レベルへ丸めるだけでよい (2次元の総当たり探索は不要で、
各軸の丸めで等価になる)。この計算は共通ライブラリ comm.decide が行う。

【このスクリプトの流れ】
4つのQAM方式 (4/16/64/256QAM) について、PRBS から作った送信シンボルに AWGN を
加え、判定し、シンボル誤り率 (SER) を測って標準出力する。さらに受信点群と参照点を
重ねたコンステレーション図 (decision.png) を保存して、判定の様子を可視化する。
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


HERE = os.path.dirname(os.path.abspath(__file__))  # このスクリプトがあるフォルダ (図の保存先)
QAM_ORDERS = [4, 16, 64, 256]   # 評価する QAM 多値数 (一覧表示・サブプロットで使う)
N_SYM = 10 ** 6                 # 送受信するシンボル数 (多いほど SER の推定が安定する)
SNR_DB = 20.0                   # 信号対雑音比 [dB] (大きいほど雑音が小さい)
SEED = 1                        # 乱数シード (固定して結果を再現可能にする)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から n_sym 個の M-QAM 送信シンボルを作る (試験信号の生成)。

    PRBS (擬似ランダムビット列) を必要なだけ繰り返してビット列を作り、
    QAM マッピングでシンボルに変換する。BER/SER 測定の「既知の送信信号」として使う。
    """
    prbs = comm.generate_prbs(15)               # M系列 PRBS を 1 周期 (32767 ビット) 生成
    k = int(np.log2(M))                          # 1 シンボルが運ぶビット数 (=log2 M)
    # PRBS を k 回繰り返してビット数を k の倍数にし、QAM シンボルへマッピング (1ブロック)
    sym_block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(sym_block)))  # n_sym シンボルを満たすのに必要な繰り返し回数
    # ブロックを繰り返し連結し、先頭 n_sym 個に切り詰めて返す
    return np.tile(sym_block, reps)[:n_sym]


def main() -> None:
    rng = np.random.default_rng(SEED)              # 乱数生成器 (シード固定で AWGN を再現可能に)
    print(f"シンボル数={N_SYM}, SNR={SNR_DB} dB")

    # 4方式を 2x2 のサブプロットに並べてコンステレーション図を描く
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))

    # 各サブプロット (ax) と各 QAM 多値数 (M) を対応づけて順に処理
    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        tx = make_symbols(M, N_SYM)                # 送信シンボル (理想点。雑音なし)
        # AWGN を付加して受信信号を作る。信号電力=1 (規格化済み) を明示して SNR を厳密に設定
        rx = comm.add_awgn(tx, SNR_DB, signal_power=1.0, rng=rng)  # 受信 (雑音付き)
        # 受信点を最近傍の理想シンボルへ硬判定 (各軸を最も近い PAM レベルへ丸める)
        dec = comm.decide(rx, M)                    # 判定 (最近傍シンボル)

        # シンボル誤り率 (SER): 判定結果が送信シンボルと一致しなかった割合。
        # 1 つでも I/Q がずれて別の格子点に丸められたら 1 シンボル誤りと数える。
        sym_err = np.count_nonzero(~np.isclose(dec, tx))  # 判定 != 送信 の個数
        ser = sym_err / N_SYM                       # SER = 誤りシンボル数 / 総シンボル数
        print(f"{M:3d}QAM: シンボル誤り数={sym_err:7d}, SER={ser:.3e}")

        # 受信点の雲を散布 (先頭 4000 点のみ。雑音による広がりを可視化)
        n_plot = 4000
        ax.scatter(rx[:n_plot].real, rx[:n_plot].imag, s=4, alpha=0.12,
                   color="C0", label="received")
        # 判定の基準となる M 個の参照シンボル (理想コンステレーション点) を赤×で重ねる
        ref = comm.qam_constellation(M)             # 規格化済みの全 M 個の格子点を取得
        ax.scatter(ref.real, ref.imag, s=40, marker="x", color="red",
                   label="reference symbols")
        ax.set_title(f"{M}QAM decision (SER = {ser:.1e})")
        ax.set_xlabel("In-phase (I)")              # 横軸 = 同相成分 (実部)
        ax.set_ylabel("Quadrature (Q)")            # 縦軸 = 直交成分 (虚部)
        ax.set_aspect("equal")                     # I/Q を等スケールにして格子を正方に保つ
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")             # 目盛りを内向きに
        lim = 1.7                                   # 表示範囲 (規格化電力1なので外周点が ±1付近)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        if M == 4:                                  # 凡例は左上 (4QAM) のサブプロットにだけ表示
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Hard decision against reference constellation", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "decision.png")
    fig.savefig(out, dpi=120)                       # コンステレーション図を PNG に保存
    print(f"判定図を保存しました: {out}")


if __name__ == "__main__":
    main()
