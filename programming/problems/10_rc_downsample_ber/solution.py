"""問題2-4: ダウンサンプリング・QAM復調・デマッピングと BER

問2-3 で得た「複素AWGN付きスペクトル整形サンプル列」を、ISI-free タイミングで
ダウンサンプリングし、QAM復調(硬判定)+ QAMデマッピングしてビット系列を復元、
ビット誤り率(BER)を計算する。

処理の流れ(送信機 -> 通信路 -> 受信機):
  1. ランダムビット生成 (送信したい情報)
  2. bits_to_symbols      … ビット -> 複素QAMシンボル (変調)
  3. pulse_shape          … シンボルを raised cosine で帯域制限波形に整形
                            (シンボル間に 0 を挿入してアップサンプル後、RCと畳み込み)
  4. add_awgn             … 通信路の白色ガウス雑音を付加
  5. downsample_isi_free  … シンボル中心(他シンボルの裾が 0 になる点)だけを抽出
                            = ISIが生じないタイミングでサンプリング
  6. symbols_to_bits      … 受信シンボルを硬判定 + デマッピングしてビットに戻す (復調)
  7. count_bit_errors     … 送信ビットと比べて誤りビット数 -> BER

ポイント: raised cosine は Nyquist パルス(判定点で他シンボルの寄与が 0)なので、
シンボル中心で抜き取れば整形しても判定点の受信シンボルは整形なし(第1週)と同じになる。
したがって実測BERは整形なし・解析解とほぼ一致するはずで、本問はそれを確認するもの。
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
QAM_ORDERS = [4, 16, 64, 256]   # 比較する QAM 多値数 (QPSK, 16/64/256QAM)
BETA = 1.0                       # raised cosine のロールオフ係数 α (帯域の広がり)
SPS = 2                          # 1シンボルあたりサンプル数 (oversampling factor)
SPAN = 12                        # RCフィルタ長 (シンボル数)
N_SYM = 500_000                  # 送信シンボル数 (多いほど低い BER まで安定して測れる)
SNR_DB = 20.0                    # 評価する SNR (Es/N0) [dB]
SEED = 4                         # 乱数シード (実行ごとに同じ結果を得る = 再現性)


def main() -> None:
    # 乱数生成器をシード固定で作る (ビット生成・雑音の再現性のため)
    rng = np.random.default_rng(SEED)
    print(f"α={BETA}, {SPS} sps, シンボル数={N_SYM}, SNR={SNR_DB} dB\n")
    # 結果テーブルのヘッダ: 方式 / 誤りビット数 / 実測BER(整形あり) / 解析解BER
    print(f"{'方式':>7} {'誤りビット数':>12} {'BER(整形,実測)':>16} "
          f"{'BER(解析解)':>14}")

    results = {}
    for M in QAM_ORDERS:
        k = int(np.log2(M))                          # 1シンボルが運ぶビット数 log2(M)
        # --- 送信機: ランダムビットを QAM シンボルへ変調 ---
        bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)  # 0/1 を N_SYM*k 個
        tx_sym = comm.bits_to_symbols(bits, M)        # ビット -> 規格化複素QAMシンボル (変調)
        # --- 整形 -> 通信路雑音 -> ISI-free ダウンサンプリング ---
        # pulse_shape: シンボル間に 0 を挿入してアップサンプルし RC と畳み込んで整形。
        #   返り値 delay は対称FIRの群遅延 [サンプル] で、判定点を拾う基準になる。
        shaped, delay = comm.pulse_shape(tx_sym, SPS, BETA, SPAN)
        # add_awgn: 平均シンボル電力=1 を前提に SNR_DB 相当の複素白色ガウス雑音を付加
        rx = comm.add_awgn(shaped, SNR_DB, signal_power=1.0, rng=rng)
        # downsample_isi_free: delay を起点に sps 間隔で抜き、各シンボル中心
        #   (他シンボルの裾が 0 = ISIなし のタイミング) の受信シンボルを取り出す
        rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)
        # --- 受信機: 硬判定(最近傍シンボル) + デマッピングでビットに戻す (復調) ---
        rx_bits = comm.symbols_to_bits(rx_sym, M)
        n = min(len(bits), len(rx_bits))             # 末尾切り詰めの可能性に備え短い方に揃える
        n_err = comm.count_bit_errors(bits[:n], rx_bits[:n])  # 送受信ビットの不一致数
        ber = n_err / n                              # 実測BER = 誤りビット数 / 比較ビット数
        ber_th = float(comm.ber_theory_qam(M, SNR_DB))  # 解析解(グレイ符号方形QAMの近似式)
        results[M] = (ber, ber_th)
        # 1行 = 1方式: 方式 / 誤りビット数 / 実測BER / 解析解BER
        print(f"{M:5d}QAM {n_err:12d} {ber:16.3e} {ber_th:14.3e}")

    # --- 図: 整形あり実測 BER (棒) vs 解析解 (点) を方式ごとに比較 ---
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(QAM_ORDERS))                   # 棒グラフの横位置 (方式ごとに 0,1,2,3)
    # log軸で 0 を描けないので、BER=0 のときは下限 1e-12 にクリップ
    ber_meas = [max(results[M][0], 1e-12) for M in QAM_ORDERS]
    ber_th = [results[M][1] for M in QAM_ORDERS]
    ax.bar(x, ber_meas, width=0.5, color="C2", alpha=0.7, label="RC-shaped (measured)")  # 実測BER
    ax.plot(x, ber_th, "rD--", label="analytic BER")  # 解析解BER (棒の頂点とほぼ重なるはず)
    ax.set_yscale("log")                             # BERは桁で変わるので縦軸は対数
    ax.set_xticks(x); ax.set_xticklabels([f"{M}QAM" for M in QAM_ORDERS])  # 横軸ラベル=方式名
    ax.set_ylabel("BER")
    ax.set_title(f"BER of RC-shaped QAM at SNR={SNR_DB:.0f} dB (ISI-free sampling)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "ber_rc.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER図を保存しました: {out}")  # 出力PNGのパスを標準出力に表示


if __name__ == "__main__":
    main()
