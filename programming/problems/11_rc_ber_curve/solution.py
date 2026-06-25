"""問題2-5: スペクトル整形 QAM サンプル列に対する BER の SNR 依存性

問2-1 と同じ BER-SNR 曲線を、今度は raised cosine 整形 + ISI-free ダウンサンプリングの
チェーンで測定し、解析解と比較する。

やること:
  - 各 QAM 方式 (4/16/64/256QAM) について、SNR を 0〜30 dB まで振りながら
    「整形 -> 雑音 -> ISI-freeダウンサンプリング -> 復調 -> BER測定」を繰り返す。
  - 横軸 SNR、縦軸 BER (対数) のグラフに、測定点 (マーカー) と解析解 (実線) を重ねる。

ポイント: raised cosine は Nyquist パルスなので、判定点(シンボル中心)では整形による
ISI が無く、整形なし(問2-1)・解析解と同じ BER-SNR 曲線になるはず。本問はそれを
SNR の関数として確認する(問2-4 を SNR 全域に拡張したもの)。
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
QAM_ORDERS = [4, 16, 64, 256]   # 曲線を描く QAM 多値数
BETA = 1.0                       # raised cosine ロールオフ係数 α
SPS = 2                          # 1シンボルあたりサンプル数
SPAN = 12                        # RCフィルタ長 (シンボル数)
N_SYM = 300_000                  # 1つの SNR 点あたりの送信シンボル数
SNR_LIST = np.arange(0, 31, 2)   # 実測する SNR 点 [dB] (0,2,4,...,30)
SEED = 5                         # 乱数シード (再現性)


def measure_ber_shaped(M: int, snr_db: float, rng: np.random.Generator) -> float:
    """1つの (方式 M, SNR) について整形チェーンで BER を実測して返す。

    送信 -> RC整形 -> AWGN -> ISI-freeダウンサンプリング -> 復調 -> 誤り計数、
    という問2-4 と同じ一連の処理を 1 回実行する補助関数。
    """
    k = int(np.log2(M))                              # シンボルあたりビット数
    bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)  # 送信ランダムビット
    tx = comm.bits_to_symbols(bits, M)               # ビット -> QAMシンボル (変調)
    shaped, delay = comm.pulse_shape(tx, SPS, BETA, SPAN)  # RC整形 (+群遅延 delay)
    rx = comm.add_awgn(shaped, snr_db, signal_power=1.0, rng=rng)  # 通信路雑音を付加
    rx_sym = comm.downsample_isi_free(rx, delay, SPS, N_SYM)  # シンボル中心を抽出 (ISIなし)
    rx_bits = comm.symbols_to_bits(rx_sym, M)        # 硬判定 + デマッピング (復調)
    n = min(len(bits), len(rx_bits))                 # 比較できる長さ
    return comm.count_bit_errors(bits[:n], rx_bits[:n]) / n   # BER = 誤り数 / 比較数


def main() -> None:
    rng = np.random.default_rng(SEED)                # シード固定の乱数生成器
    snr_fine = np.linspace(0, 30, 300)               # 解析解を滑らかな曲線で描くための細かいSNR軸

    fig, ax = plt.subplots(figsize=(7.5, 6))
    colors = ["C0", "C1", "C2", "C3"]                # 方式ごとの色 (理論線と測定点で共通)
    print(f"α={BETA}, {SPS} sps, シンボル数/点={N_SYM}\n")

    for M, c in zip(QAM_ORDERS, colors):
        # 解析解 BER 曲線 (実線): ber_theory_qam は配列SNRを受け各点のBERを返す
        ax.plot(snr_fine, comm.ber_theory_qam(M, snr_fine), "-", color=c, lw=1.5,
                label=f"{M}QAM theory")
        # SNR を振って整形チェーンの実測BERを集める
        ms, mb = [], []                              # ms: SNR点, mb: 対応する実測BER
        for snr in SNR_LIST:
            ber = measure_ber_shaped(M, snr, rng)    # この (M, snr) の実測BER
            if ber > 0:                              # BER=0 は対数軸に描けないので除外
                ms.append(snr); mb.append(ber)
        # 実測BER (中抜き四角マーカー)。Nyquistパルスなので理論線に重なるはず
        ax.plot(ms, mb, "s", color=c, ms=5, mfc="none", label=f"{M}QAM RC-sim")
        # その方式で測定できた最小BER (= シンボル数の限界でどこまで低く測れたか) を表示
        print(f"{M:3d}QAM: 測定できた最小BER = {min(mb):.2e}")

    ax.set_yscale("log"); ax.set_ylim(1e-6, 1); ax.set_xlim(0, 30)  # 縦軸=BER(対数), 横軸=SNR
    ax.set_xlabel("SNR per symbol  Es/N0 [dB]")      # 横軸: 1シンボルあたりSNR (Es/N0)
    ax.set_ylabel("BER")                             # 縦軸: ビット誤り率
    ax.set_title("BER vs SNR for RC-shaped QAM (α=1, 2 sps) vs theory")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "ber_vs_snr_rc.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER-SNR曲線(整形)を保存しました: {out}")  # 出力PNGのパスを表示


if __name__ == "__main__":
    main()
