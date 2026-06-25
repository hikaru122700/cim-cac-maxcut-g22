"""問題1-3: QAM シンボル列への白色ガウス雑音 (AWGN) 付加

【この問題で学ぶこと】
  通信路の熱雑音などをモデル化した AWGN (加法的白色ガウス雑音) を QAM 信号に
  加えると、コンステレーションの点がどのように「ぼやける」かを観察する。
  同じ SNR でも、多値数 M が大きい QAM ほど点の間隔が狭いので、雑音による
  広がりに対して相対的に弱く、点の塊どうしが重なりやすくなる (= 誤りやすい)。

【SNR (信号電力対雑音電力比) について】
  SNR が大きいほど雑音が小さくクリーンな信号。ここでは SNR = 20 dB を使う。
  dB を真数に直すと 10**(20/10) = 100 倍なので、平均シンボル電力 1 に対し
  雑音電力 = 1 / 100 = 0.01 となる。

【手順】
  1. 問1-2 と同じ手順で QAM シンボルを作り、繰り返してシンボル数を 10**6 に増やす。
     (シンボル数を増やすと統計が安定し、実測 SNR が設定値に近づく)
  2. SNR = 20 dB に相当する複素 AWGN を付加する。
     (平均シンボル電力 1 に対し、雑音電力 = 1 / 10**(20/10) = 0.01)
  3. 雑音付加後のコンステレーションを図示する。

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
QAM_ORDERS = [4, 16, 64, 256]   # 図示する QAM 多値数
N_SYM = 10 ** 6           # シンボル数 (多いほど雑音統計が安定し実測 SNR が設定値に近づく)
SNR_DB = 20.0             # 信号電力対雑音電力比 [dB] (大きいほど低雑音)
SEED = 1                  # 再現性のための乱数シード (同じ seed なら毎回同じ雑音)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から M-QAM シンボルを作り、繰り返して n_sym シンボルに増やす。

    問1-2 と同じ PRBS -> QAM の流れで 1 ブロック (32767 シンボル) を作り、
    それを必要回数だけ並べて n_sym シンボルに伸ばす (タイル状に繰り返す)。
    """
    prbs = comm.generate_prbs(15)                 # PRBS 1 周期 (長さ 32767 の 0/1 系列)
    k = int(np.log2(M))                           # 1シンボルあたりビット数
    bits = np.tile(prbs, k)                       # PRBS を k 回繰り返す (1ブロック = 32767 シンボル分)
    sym_block = comm.bits_to_symbols(bits, M)     # QAM 変調 (規格化済みの 1 ブロック分シンボル)
    reps = int(np.ceil(n_sym / len(sym_block)))   # 目標シンボル数に届くまで何回繰り返すか
    sym = np.tile(sym_block, reps)[:n_sym]        # 並べてから先頭 n_sym 個に切り詰める
    return sym


def main() -> None:
    # SEED を固定した乱数生成器。これを add_awgn に渡すと雑音が毎回同じになり再現可能
    rng = np.random.default_rng(SEED)
    print(f"シンボル数 : {N_SYM}")
    # 雑音電力 = 1 / 10**(SNR/10)。SNR=20dB なら 0.01 と表示される
    print(f"SNR        : {SNR_DB} dB  (雑音電力 = {10 ** (-SNR_DB / 10):.4f})")

    # 2x2 のサブプロットに 4 種類の QAM の雑音付きコンステレーションを並べる
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    n_plot = 4000                                 # 図に描く点数 (全点だと真っ黒になる)

    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        sym = make_symbols(M, N_SYM)              # 送信シンボル列 (雑音なし、平均電力 1)
        # comm.add_awgn: SNR=SNR_DB になるよう複素ガウス雑音を生成して付加した受信信号 rx を返す
        #   signal_power=1.0 を明示 (シンボルが平均電力 1 に規格化済みのため)
        rx = comm.add_awgn(sym, SNR_DB, signal_power=1.0, rng=rng)

        # 実測 SNR = 信号電力 / 誤差(=雑音)電力 を dB で計算。設定値 20dB に近いはず
        #   分子 = 平均|送信|^2、分母 = 平均|受信-送信|^2 = 付加した雑音の平均電力
        meas_snr = 10 * np.log10(np.mean(np.abs(sym) ** 2)
                                 / np.mean(np.abs(rx - sym) ** 2))
        print(f"{M:3d}QAM: 実測SNR={meas_snr:5.2f} dB")

        # 受信点の散布図 (横=I, 縦=Q)。先頭 n_plot 点だけ描く。雑音で点が雲状に広がる
        ax.scatter(rx[:n_plot].real, rx[:n_plot].imag, s=4, alpha=0.15)
        ax.set_title(f"{M}QAM + AWGN (SNR = {SNR_DB:.0f} dB)")
        ax.set_xlabel("In-phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
        lim = 1.7
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.suptitle(f"Received constellations with AWGN (N = {N_SYM:,} symbols)",
                 fontsize=13)
    fig.tight_layout()
    # この solution.py と同じフォルダに constellation_awgn.png として保存
    out = os.path.join(HERE, "constellation_awgn.png")
    fig.savefig(out, dpi=120)
    print(f"雑音付加コンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
