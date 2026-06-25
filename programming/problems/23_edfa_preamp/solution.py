"""問題4-4: EDFA 前置増幅構成での QPSK の BER vs 入力パワー Pin

■ 何をする問題か
  単一偏波 QPSK の微弱な光信号を、受信機の直前に置いた EDFA (光増幅器) で増幅してから
  コヒーレント光受信する「前置増幅 (preamplifier) 構成」をシミュレーションする。
  EDFA は信号を増幅するのと同時に、自然放出由来の ASE 雑音を必ず加える。この ASE が
  受信品質を支配するので、EDFA 入力パワー Pin を変えながら BER がどう変わるかを調べる
  (レーザ位相雑音は無視。雑音指数 NF = 4 dB)。

■ なぜ前置増幅構成か
  受信機自身の熱雑音より先に信号を十分増幅しておくと、系全体の雑音は EDFA の ASE で
  決まる「量子雑音限界」に近づく。よって微弱光をどこまで小さいパワーまで受信できるか
  (= 受信感度) を理論的に見積もれる、教科書的な題材になっている。

■ 物理 (理論曲線の根拠)
  EDFA出力 ASE PSD (1偏波・片側): S_ASE = (NF_lin/2)·hν·(G-1)   ← fiber.ase_psd が計算
    NF_lin: 雑音指数の真数, hν: 光子1個のエネルギー, G: 線形利得
  1 sample/symbol (サンプリングレート fs = 符号速度 Rs) なので、1シンボルに乗る ASE 雑音
  の電力は「PSD × 帯域」= S_ASE·Rs。増幅後の信号電力は G·Pin。よって
    1シンボルSNR  Es/N0 = G·Pin / (S_ASE·Rs) ≈ 2·Pin / (NF_lin·hν·Rs)  (高利得 G≫1 のとき)
  QPSK の BER は Q(√(Es/N0)) = (1/2)erfc(√(Es/N0)/√2)。
  → Pin を上げれば SNR が線形に上がり、BER は急速に下がる (片対数で右肩下がりの直線的曲線)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm    # noqa: E402
import fiber   # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
RS = 32e9              # 符号速度 [baud]。1 sps なのでサンプリングレート fs = Rs に等しい
NF_DB = 4.0           # EDFA の雑音指数 [dB] (ASE 雑音の強さを決める)
GAIN_DB = 20.0        # EDFA 利得 [dB] (前置増幅、高利得)
N_SYM = 400_000       # 送信シンボル数 (多いほど低い BER まで誤り数を統計的に拾える)
SEED = 20             # 乱数シード (ASE 雑音の再現性のため)


def dbm_to_w(dbm):
    """光パワーの単位を dBm → W に換算する。0 dBm = 1 mW を基準とする対数表記。"""
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)            # ASE 雑音生成用の乱数生成器
    G = 10 ** (GAIN_DB / 10.0)                    # 利得 dB → 線形倍率 (例: 20 dB = 100 倍)
    s_ase = fiber.ase_psd(G, NF_DB)              # ASE の PSD S_ASE [W/Hz] (= NF_lin/2·hν·(G-1))
    hnu = fiber.photon_energy()                  # 光子1個のエネルギー hν [J] (1550 nm, 表示用)
    print(f"Rs={RS/1e9:.0f} Gbaud, NF={NF_DB} dB, G={GAIN_DB} dB")
    print(f"hν = {hnu:.3e} J, S_ASE = {s_ase:.3e} W/Hz\n")

    # --- 送信 QPSK シンボル列を用意 (平均電力 1 に規格化済み) ---
    prbs = comm.generate_prbs(15)               # PRBS15 (試験用の擬似ランダムビット列) を1周期生成
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)  # ビット数を 2 の倍数 (QPSK=2bit/sym) に揃えてマッピング
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]  # ブロックを繰り返して N_SYM シンボルに
    bits_tx = comm.symbols_to_bits(s, 4)        # 送信ビット列 (BER 比較の基準。後で受信側と突き合わせる)

    # --- EDFA 入力パワー Pin を掃引して BER を測定 ---
    # Pin が小さいほど ASE に埋もれて BER が悪化する。前置増幅構成は微弱光受信なので
    # 範囲は -50〜-38 dBm (10 fW〜0.1 µW 程度) と非常に低いパワー域を見る。
    pin_dbm = np.arange(-50, -37, 1.0)
    ber_sim, ber_th, snr_db_arr = [], [], []     # 実測BER / 理論BER / SNR[dB] を貯める
    for pdbm in pin_dbm:
        Pin = dbm_to_w(pdbm)                      # この点の EDFA 入力パワー [W]
        A_in = np.sqrt(Pin) * s                  # 入力光振幅 (√W): |A_in|² の平均が Pin になる
        A_out = fiber.edfa(A_in, G, NF_DB, RS, rng)  # EDFA: 振幅を √G 倍し ASE 雑音 (複素AWGN) を付加
        r = A_out / np.sqrt(G * Pin)             # 信号成分を単位電力に正規化 (判定スケールへ戻す)
        bits_rx = comm.symbols_to_bits(r, 4)     # 受信シンボルを硬判定して復調 (QPSK→ビット列)
        nb = min(len(bits_tx), len(bits_rx))     # 送受で長さを揃える
        ber = comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb  # 誤りビット数 / 総ビット数

        snr_lin = G * Pin / (s_ase * RS)         # 1シンボルSNR Es/N0 = 信号電力 / (ASE PSD × 帯域)
        ber_sim.append(ber)
        ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))  # 同じ SNR の QPSK 理論 BER
        snr_db_arr.append(10 * np.log10(snr_lin))

    # --- 標準出力: Pin ごとに SNR・実測BER・理論BER を一覧表示 ---
    # 実測と理論がよく一致していれば、EDFA/ASE モデルと QPSK 復調が正しいことの確認になる。
    print(f"{'Pin[dBm]':>9} {'Es/N0[dB]':>10} {'BER(sim)':>11} {'BER(theory)':>12}")
    for p, sn, bs, bt in zip(pin_dbm, snr_db_arr, ber_sim, ber_th):
        print(f"{p:9.0f} {sn:10.2f} {bs:11.2e} {bt:12.2e}")

    # ---- 図: BER vs Pin (縦軸 BER は対数、横軸 Pin は dBm) ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(pin_dbm, ber_th, "r-", lw=2, label="theory")  # 理論曲線 (赤実線)
    sim = np.array(ber_sim)
    m = sim > 0                                   # BER=0 は対数軸に描けないので除外するマスク
    ax.semilogy(pin_dbm[m], sim[m], "ko", ms=5, mfc="none", label="simulation")  # 実測 (白丸)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("EDFA input power Pin [dBm]")   # 横軸: EDFA 入力パワー (大きいほど SNR 良)
    ax.set_ylabel("BER")                          # 縦軸: ビット誤り率 (小さいほど高品質)
    ax.set_title(f"Preamplified QPSK BER vs Pin (NF={NF_DB} dB, Rs={RS/1e9:.0f} Gbaud)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "edfa_preamp_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    # --- BER=1e-3 となる入力パワー = 受信感度 を対数補間で求める ---
    # 受信感度: 「ある目標 BER を達成できる最小の入力パワー」。前置増幅構成の性能指標。
    # BER は片対数で滑らかなので、log10(BER) を線形補間して Pin を逆引きする。
    sim_arr = np.array(ber_sim)
    valid = sim_arr > 0                           # 0 (誤りなし) の点は log が取れないので除外
    # interp は x が昇順である必要がある。Pin↑ で BER↓ なので [::-1] で BER 昇順に並べ替えて補間
    sens = np.interp(np.log10(1e-3), np.log10(sim_arr[valid][::-1]), pin_dbm[valid][::-1])
    print(f"受信感度 (BER=1e-3) ≈ {sens:.1f} dBm")


if __name__ == "__main__":
    main()
