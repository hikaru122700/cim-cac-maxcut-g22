"""問題2-3: スペクトル整形 QAM サンプル列への複素 AWGN 付加

この演習でやること:
  問題2-2 で作った「RC整形した QAM サンプル列」に通信路の雑音 (複素AWGN) を加え、
  受信側でシンボル中心 (ISI-free タイミング) を抜き出してコンステレーションを描く。
  雑音が加わると、理想格子点のまわりに点が「雲」のように広がる様子を観察する。

なぜ判定点 (ISI-free タイミング) で SNR を定義するか:
  整形後のサンプル列は時刻によって瞬時電力が変動するが、最終的に通信品質を
  決めるのは「シンボルを判定する瞬間」の信号と雑音の比である。そこで SNR は
  シンボル判定点での値で定義する。判定点では信号電力 = シンボル電力 = 1 に
  規格化されているので、雑音電力 = 1 / 10^(20/10) = 0.01 となる。

処理の流れ:
  - α=1 raised cosine、2 sample/symbol で整形したサンプル列を生成 (サンプル数 10**6)。
  - SNR = 20 dB に相当する複素 AWGN を付加する (信号電力1基準で雑音電力0.01)。
  - ISI が生じないタイミングでサンプル抽出 (ダウンサンプル) し、コンステレーションを図示。
  - 抽出後に実測 SNR を計算し、設定値 (20 dB) とほぼ一致することを確認する。
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


HERE = os.path.dirname(os.path.abspath(__file__))   # このスクリプトのあるフォルダ (図の保存先)
QAM_ORDERS = [4, 16, 64, 256]   # コンステレーションを描く QAM 多値数
BETA = 1.0                  # ロールオフ係数 α
SPS = 2                     # 1シンボルあたりサンプル数
SPAN = 12                   # RCフィルタ長 [シンボル]
N_SAMPLE = 10 ** 6          # 総サンプル数
N_SYM = N_SAMPLE // SPS     # シンボル数 (= 500000)
SNR_DB = 20.0               # 付加する雑音の SNR [dB] (判定点基準)
SEED = 3                    # 乱数シード (再現性のため固定)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から n_sym 個の M-QAM シンボル列を作る (足りなければタイルで繰り返す)。

    Args:
        M: QAM 多値数。
        n_sym: 欲しいシンボル数。

    Returns:
        長さ n_sym の複素QAMシンボル列。
    """
    prbs = comm.generate_prbs(15)                  # 試験用の擬似ランダムビット列 (PRBS15)
    k = int(np.log2(M))                            # 1シンボルあたりビット数
    block = comm.bits_to_symbols(np.tile(prbs, k), M)  # PRBS を変調した1ブロック分のシンボル
    reps = int(np.ceil(n_sym / len(block)))        # 必要数を満たすのに何回繰り返すか
    return np.tile(block, reps)[:n_sym]            # 繰り返して連結し、先頭 n_sym 個に切り詰め


def main() -> None:
    rng = np.random.default_rng(SEED)   # 雑音生成用の乱数 (SEED固定で再現可能)
    print(f"サンプル数={N_SAMPLE} ({SPS} sps -> {N_SYM} シンボル), SNR={SNR_DB} dB\n")

    # 4方式を 2x2 のサブプロットで並べて描く
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        tx_sym = make_symbols(M, N_SYM)                 # 送信シンボル列を生成
        # RC整形: アップサンプル + RCフィルタ畳み込み。delay は群遅延 (シンボル中心の起点)
        shaped, delay = comm.pulse_shape(tx_sym, SPS, BETA, SPAN)

        # 通信路: 複素AWGNを付加。判定点で信号電力=1 になるよう、雑音はシンボル電力(=1)基準。
        # signal_power=1.0 を明示することで、整形波形の瞬時電力変動に惑わされず
        # 「判定点 SNR = SNR_DB」を保証する (雑音電力 = 1/10^(SNR/10))。
        rx = comm.add_awgn(shaped, SNR_DB, signal_power=1.0, rng=rng)

        # 受信: ISI-free タイミング (シンボル中心) でダウンサンプルして判定点サンプルを得る
        rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)

        # 実測SNR = 1 / (受信点と送信点の誤差電力の平均) を dB 換算。設定の 20 dB に近いはず。
        snr_meas = 10 * np.log10(1.0 / np.mean(np.abs(rx_sym - tx_sym[:len(rx_sym)]) ** 2))
        print(f"{M:3d}QAM: ダウンサンプル後 実測SNR = {snr_meas:5.2f} dB")

        # 受信判定点の散布図 (横=I, 縦=Q)。先頭4000点だけ描く (重なり過ぎ防止)
        ax.scatter(rx_sym[:4000].real, rx_sym[:4000].imag, s=4, alpha=0.15)
        ax.set_title(f"{M}QAM, RC+AWGN, ISI-free sampled")
        ax.set_xlabel("I"); ax.set_ylabel("Q")          # 横軸=同相成分, 縦軸=直交成分
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"RC-shaped + complex AWGN, after ISI-free downsampling (SNR={SNR_DB:.0f} dB)",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation_rc_awgn.png")
    fig.savefig(out, dpi=120)
    print(f"\nコンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
