"""問題1-6: ビット誤り率 (BER) の測定と処理時間の計測

【この問題で学ぶこと】
問1-2〜1-5 で作った各部品を1本につなげ、完全な送受信チェーンを構築する。
そのうえで通信品質の最終指標である BER (ビット誤り率) を測定し、理論式 (解析解)
と比較してシミュレーションが正しく動いていることを検証する。

【送受信チェーン (各部品がどの問題に対応するか)】
  PRBS生成     (1-1) -> 試験用の擬似ランダムビット列
  QAMマッピング (1-3) -> ビットを複素シンボルへ変調
  AWGN付加     (1-2) -> 通信路の雑音をシミュレート
  判定         (1-4) -> 受信点を最近傍の理想点へ硬判定
  QAMデマッピング(1-5) -> シンボルをビットへ復調
  誤り計数            -> 送信ビットと受信ビットを突き合わせて誤りを数える

【出力する量の意味】
各 QAM 方式 (4/16/64/256QAM) について次を求めて表にする:
  - 誤りビット数        : 送信と受信で食い違ったビットの個数
  - BER (実測)          = 誤りビット数 / 総ビット数  (シミュレーションで測った値)
  - BER (解析解)        : 理論式 comm.ber_theory_qam による値 (実測の答え合わせ)
  - SER/k               : シンボル誤り率を 1 シンボルのビット数 k で割った概算 BER
                          (グレイ符号では 1 シンボル誤り≒1 ビット誤りなので BER に近い)
さらに time ライブラリで送受信チェーン全体の処理時間を測る
(波形プロットの描画時間は計測対象に含めない)。
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))  # このスクリプトがあるフォルダ (図の保存先)
QAM_ORDERS = [4, 16, 64, 256]   # 評価する QAM 多値数
N_SYM = 10 ** 6                 # 送受信シンボル数 (多いほど BER 推定が安定。低 BER ほど多く必要)
SNR_DB = 20.0                   # 信号対雑音比 [dB]
SEED = 1                        # 乱数シード (AWGN を再現可能に固定)


def run_chain(M: int, rng: np.random.Generator):
    """1方式ぶんの送受信チェーンを実行し、(誤り数, 総ビット, BER, SER) を返す。

    PRBS生成 -> マッピング -> AWGN -> 判定+デマッピング -> 誤り計数 を一気に行う。
    送信ビットを基準に、受信後に復調したビットと突き合わせて誤りを数える。
    """
    k = int(np.log2(M))                              # 1 シンボルあたりビット数

    # --- 送信ビット列 (PRBS を繰り返し N_SYM シンボル分) ---
    prbs = comm.generate_prbs(15)                    # M系列 PRBS を 1 周期生成
    block = comm.bits_to_symbols(np.tile(prbs, k), M)  # PRBS を k 回繰り返してマッピング (1ブロック)
    reps = int(np.ceil(N_SYM / len(block)))          # N_SYM を満たすのに必要な繰り返し回数
    tx_sym = np.tile(block, reps)[:N_SYM]            # ブロックを連結し N_SYM シンボルに切り詰め
    # 送信シンボルを一度デマッピングして「送信ビット列」を確定する (誤り計数の基準)
    tx_bits = comm.symbols_to_bits(tx_sym, M)        # 送信ビット (基準)

    # --- 伝送 (通信路の雑音を付加) ---
    # 信号電力=1 を明示して SNR を厳密に設定し、受信シンボルを得る
    rx_sym = comm.add_awgn(tx_sym, SNR_DB, signal_power=1.0, rng=rng)
    # 受信シンボルを復調 (symbols_to_bits は内部で硬判定してからビットに戻す)
    rx_bits = comm.symbols_to_bits(rx_sym, M)        # 判定 + デマッピング

    # --- 誤り計数 ---
    n_bit_err = comm.count_bit_errors(tx_bits, rx_bits)  # 送信/受信ビットの不一致数を数える
    ber = n_bit_err / len(tx_bits)                   # BER = 誤りビット数 / 総ビット数
    # SER も併せて算出: 受信シンボルを判定し、送信シンボルと異なる割合を数える
    ser = np.count_nonzero(~np.isclose(comm.decide(rx_sym, M), tx_sym)) / N_SYM
    return n_bit_err, len(tx_bits), ber, ser


def main() -> None:
    rng = np.random.default_rng(SEED)              # 乱数生成器 (AWGN を再現可能に固定)

    results = {}                                    # 各 M の (BER実測, SER) を後の作図用に保存
    print(f"シンボル数={N_SYM}, SNR={SNR_DB} dB\n")
    # 結果表のヘッダ行
    print(f"{'方式':>7} {'誤りビット数':>12} {'総ビット数':>12} "
          f"{'BER(実測)':>12} {'BER(解析解)':>12} {'SER/k':>10}")

    # ---- 処理時間の計測開始 (描画は含めず、送受信チェーンのみを計測) ----
    t_start = time.time()                           # 開始時刻
    for M in QAM_ORDERS:
        # 1方式ぶんの送受信チェーンを実行し、誤り数・総ビット・BER・SER を取得
        n_err, n_bits, ber, ser = run_chain(M, rng)
        results[M] = (ber, ser)                     # 作図用に保存
        k = int(np.log2(M))                         # ビット/シンボル (SER/k の計算に使う)
        ber_th = float(comm.ber_theory_qam(M, SNR_DB))  # 理論 BER (解析解。実測の答え合わせ)
        # 1 行ぶんの結果を表形式で出力 (SER/k は SER から概算した BER)
        print(f"{M:5d}QAM {n_err:12d} {n_bits:12d} "
              f"{ber:12.3e} {ber_th:12.3e} {ser / k:10.3e}")
    t_end = time.time()                             # 終了時刻
    # ---- 計測終了 ----
    # チェーン全体の所要時間を出力 (描画時間は含まない。ここまでが計測区間)
    print(f"\nRequired time for programming: {t_end - t_start:.3f} s "
          f"(波形プロットを除く)")

    # --- 図: 実測BER vs 解析解 (棒グラフ=実測, 点線=理論) ---
    # 縦軸を対数にして、方式ごとに実測 BER と理論 BER を重ねて比較する。
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(QAM_ORDERS))                  # 横軸の位置 (各 QAM 方式に対応)
    # 実測 BER。0 だと対数軸で描けないため下限 1e-12 でクリップ
    ber_meas = [max(results[M][0], 1e-12) for M in QAM_ORDERS]
    ber_theory = [float(comm.ber_theory_qam(M, SNR_DB)) for M in QAM_ORDERS]  # 理論 BER
    ax.bar(x, ber_meas, width=0.5, color="C0", alpha=0.7, label="measured BER")  # 実測を棒で
    ax.plot(x, ber_theory, "rD--", label="analytic BER")  # 理論を赤ダイヤ点線で重ねる
    ax.set_yscale("log")                            # BER は桁が大きく変わるので対数軸
    ax.set_xticks(x)
    ax.set_xticklabels([f"{M}QAM" for M in QAM_ORDERS])  # 横軸ラベルを方式名に
    ax.set_ylabel("BER")
    ax.set_title(f"BER at SNR = {SNR_DB:.0f} dB ({N_SYM:,} symbols)")
    ax.grid(True, which="both", alpha=0.3)          # 主・副目盛りともグリッド表示
    ax.tick_params(direction="in")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "ber.png")
    fig.savefig(out, dpi=120)                       # BER 比較図を PNG に保存
    print(f"BER図を保存しました: {out}")


if __name__ == "__main__":
    main()
