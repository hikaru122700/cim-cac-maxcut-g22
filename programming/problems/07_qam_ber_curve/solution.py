"""問題2-1: BER の SNR 依存性 (4/16/64/256 QAM) と解析解との比較

この演習でやること:
  各 QAM 方式 (4/16/64/256) について、SNR を少しずつ変えながら
  「送信ビット -> QAM変調 -> AWGN通信路 -> 復調 -> 受信ビット」を実際に
  シミュレーションし、ビット誤り率 (BER) をモンテカルロ法で測定する。
  その測定値を、理論式から計算した BER 曲線と同じグラフに重ねて比較する。

なぜやるか:
  - QAM の多値数 M を上げる (4 -> 256) と 1 シンボルで多くのビットを送れる反面、
    コンステレーション点の間隔が狭まり雑音に弱くなる。同じ BER を得るのに
    必要な SNR が増えることを、曲線として目で確認するのが狙い。
  - 自作の変復調コード (comm.py) が正しいかを、独立に導いた解析解と
    突き合わせて検証する意味もある (実測と理論が重なれば実装が正しい)。

モンテカルロ法とは:
  乱数で大量の試行 (ここでは大量のランダムビット送受信) を行い、誤った割合を
  数えて確率 (BER) を推定する方法。試行数 (N_SYM) が多いほど低い BER まで
  精度よく測れる。

横軸 SNR [dB]、縦軸 BER (片対数: 縦が対数スケール) のグラフを描く。

解析解 (グレイ符号 方形QAM の高SNR近似式):
  BER ≈ (4/k)(1 - 1/√M) Q( sqrt( 3/(M-1) · γ_s ) ),  γ_s = 10^(SNR/10)
  ここで k=log2(M), √M は1軸あたりレベル数, γ_s はシンボルあたりSNR (真数),
  Q(·) は標準正規分布の上側確率。詳細は comm.ber_theory_qam を参照。
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


HERE = os.path.dirname(os.path.abspath(__file__))      # このスクリプトのあるフォルダ (図の保存先)
QAM_ORDERS = [4, 16, 64, 256]                          # 比較する QAM 多値数 M
N_SYM = 500_000          # 1つの SNR 点あたりに送るシンボル数 (多いほど低BERまで測れる)
SNR_LIST = np.arange(0, 31, 2)   # 実測する SNR [dB] の一覧 (0,2,4,...,30 dB)
SEED = 2                 # 乱数シード (固定して結果を再現可能にする)


def measure_ber(M: int, snr_db: float, rng: np.random.Generator) -> float:
    """ランダムビットを M-QAM で送受信し BER を測定する (1点分のモンテカルロ試行)。

    通信系を一往復シミュレートして、誤ったビットの割合 (BER) を返す。

    Args:
        M: QAM 多値数。
        snr_db: 通信路の SNR [dB]。
        rng: 乱数生成器 (ビット生成と雑音生成の両方に使う)。

    Returns:
        測定 BER = 誤りビット数 / 全送信ビット数。
    """
    k = int(np.log2(M))                                # 1シンボルが運ぶビット数 log2(M)
    # 0/1 のランダム送信ビット列を生成 (シンボル数 N_SYM × 1シンボル k ビット)
    bits = rng.integers(0, 2, size=N_SYM * k).astype(np.int8)
    # 変調: ビット列を規格化複素QAMシンボルへ (グレイ符号マッピング, 平均電力1)
    tx = comm.bits_to_symbols(bits, M)
    # 通信路: 複素AWGNを付加。signal_power=1.0 は送信シンボル電力が1に規格化
    #         されているため明示指定 (snr_db から雑音電力 = 1/10^(snr/10) を決める)
    rx = comm.add_awgn(tx, snr_db, signal_power=1.0, rng=rng)
    # 復調: 受信シンボルを最近傍点に硬判定し、ビット列へ戻す (デマッピング)
    rx_bits = comm.symbols_to_bits(rx, M)
    # 送信ビットと受信ビットを比較し、誤りビット数 ÷ 全ビット数 = BER を返す
    return comm.count_bit_errors(bits, rx_bits) / len(bits)


def main() -> None:
    # 乱数生成器を1つ作り、全 QAM・全 SNR の試行で使い回す (SEED 固定で再現可能)
    rng = np.random.default_rng(SEED)
    # 解析解 (理論曲線) は連続的な滑らかな線にしたいので、SNR を細かく刻む
    snr_fine = np.linspace(0, 30, 300)   # 解析解用の細かいSNR軸

    fig, ax = plt.subplots(figsize=(7.5, 6))   # 1枚のグラフに全方式を重ね描き
    colors = ["C0", "C1", "C2", "C3"]          # QAM 方式ごとの色 (理論と実測で同色)

    print(f"シンボル数/点 = {N_SYM}\n")
    # QAM 方式ごとに「理論曲線」と「実測点」を同じ色でプロットしていく
    for M, c in zip(QAM_ORDERS, colors):
        # --- 解析解 (理論曲線) ---
        # comm.ber_theory_qam: グレイ符号 方形QAM の BER 近似式を SNR 配列に対して計算
        ber_th = comm.ber_theory_qam(M, snr_fine)
        # 実線で理論曲線を描く (後で同色の実測点と重ねて一致を確認する)
        ax.plot(snr_fine, ber_th, "-", color=c, lw=1.5,
                label=f"{M}QAM theory")

        # --- 実測 (モンテカルロ測定。誤りが出た点のみプロット) ---
        meas_snr, meas_ber = [], []
        for snr in SNR_LIST:
            ber = measure_ber(M, snr, rng)        # この SNR で実際に送受信して BER を測定
            if ber > 0:
                # BER=0 (誤り 0 個) は対数軸に描けず信頼度も低いので除外し、
                # 1個以上の誤りが出た点だけ記録する
                meas_snr.append(snr)
                meas_ber.append(ber)
        # 実測点を中抜き丸 (○) でプロット (理論線と同色)
        ax.plot(meas_snr, meas_ber, "o", color=c, ms=5, mfc="none",
                label=f"{M}QAM sim")
        # 測定できた最も低い BER と、そのときの SNR を標準出力 (測定の到達限界の目安)
        print(f"{M:3d}QAM: 測定できた最小BER = {min(meas_ber):.2e} "
              f"(SNR={meas_snr[meas_ber.index(min(meas_ber))]:.0f} dB)")

    # 縦軸 (BER) は値が桁で変化するので対数スケールにする (片対数グラフ)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)                    # BER の表示範囲 (1e-6 〜 1)
    ax.set_xlim(0, 30)                      # SNR の表示範囲 [dB]
    ax.set_xlabel("SNR per symbol  Es/N0 [dB]")   # 横軸: シンボルあたりSNR Es/N0
    ax.set_ylabel("BER")                          # 縦軸: ビット誤り率
    ax.set_title("BER vs SNR for square QAM (simulation vs theory)")
    ax.grid(True, which="both", alpha=0.3)        # 主・補助目盛りの両方に薄いグリッド
    ax.tick_params(direction="in", which="both")  # 目盛りを内向きに
    ax.legend(ncol=2, fontsize=8)                 # 凡例を2列で表示
    fig.tight_layout()
    out = os.path.join(HERE, "ber_vs_snr.png")    # 保存先パス (このスクリプトと同じフォルダ)
    fig.savefig(out, dpi=120)
    print(f"\nBER-SNR曲線を保存しました: {out}")


if __name__ == "__main__":
    main()
