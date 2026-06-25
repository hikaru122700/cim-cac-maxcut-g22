"""問題4-5: 損失 + EDFA 多中継伝送での QPSK の BER vs Pin

■ 何をする問題か
  長距離光伝送は「ファイバで減衰 → EDFA で元のパワーへ増幅」を繰り返す多中継 (multi-span)
  構成で行う。1 スパン = 標準SMF 100 km (損失のみ、0.3 dB/km × 100 km = 30 dB) + EDFA
  (スパン損失を完全補償する利得 30 dB、NF=4 dB)。各スパンの入力パワーはどこも同じ Pin。
  1, 2, 5, 10, 20, 50 スパン中継したあとコヒーレント受信し、BER の Pin 依存性を調べる
  (波長分散・非線形・レーザ位相雑音は無視し、損失と ASE 雑音だけを扱う)。

■ なぜ多中継で雑音が増えるか (理論曲線の根拠)
  各スパンで損失と利得がちょうど釣り合うので信号パワーは保たれるが、EDFA を通るたびに
  新しい ASE 雑音が独立に加算されていく。同じ EDFA を N 台通れば ASE 雑音電力は1台ぶんの
  N 倍になるので、受信 SNR は
    SNR_N = Pin / (N · S_ASE · Rs) = SNR_1 / N
  となる (S_ASE: ASE PSD, Rs: 帯域)。SNR が 1/N になる ⇒ 同じ BER を保つには Pin を N 倍
  (= 10·log10(N) dB) 上げる必要がある。よって BER 曲線はスパン数が2倍になるごとに約 3 dB
  右 (高パワー側) へ平行移動する。
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
RS = 32e9                   # 符号速度 [baud]。1 sps なので帯域 fs = Rs
NF_DB = 4.0                 # 各 EDFA の雑音指数 [dB]
L_SPAN_KM = 100.0          # 1スパンのファイバ長 [km]
ALPHA_DB = 0.3              # ファイバ損失 [dB/km] (標準SMF)
SPAN_LOSS_DB = ALPHA_DB * L_SPAN_KM    # 1スパンの総損失 = 30 dB
N_SYM = 100_000            # 送信シンボル数
SPANS = [1, 2, 5, 10, 20, 50]          # 比較する中継スパン数 (= 伝送距離 100km〜5000km)
SEED = 21                  # 乱数シード (ASE 雑音の再現性)


def dbm_to_w(dbm):
    """光パワーの単位を dBm → W に換算する (0 dBm = 1 mW)。"""
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)      # ASE 雑音生成用の乱数生成器
    G = 10 ** (SPAN_LOSS_DB / 10.0)         # EDFA 利得 = スパン損失 (完全補償なので線形利得で 1000 倍)
    alpha = fiber.alpha_from_db(ALPHA_DB)   # 損失 dB/km → 振幅減衰係数 α [1/km] (loss_step 用)
    s_ase = fiber.ase_psd(G, NF_DB)        # EDFA 1台あたりの ASE PSD S_ASE [W/Hz]
    print(f"1スパン: {L_SPAN_KM:.0f} km, 損失 {SPAN_LOSS_DB:.0f} dB, 利得 {SPAN_LOSS_DB:.0f} dB, "
          f"NF {NF_DB} dB")
    print(f"S_ASE(1台) = {s_ase:.3e} W/Hz, 1スパン雑音電力 S_ASE·Rs = {s_ase*RS:.3e} W\n")

    # --- 送信 QPSK シンボル列 (平均電力 1 に規格化済み) ---
    prbs = comm.generate_prbs(15)          # PRBS15 (試験用擬似ランダムビット列)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)  # 2bit/sym に揃えて QPSK マッピング
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]  # N_SYM シンボルへ繰り返し
    bits_tx = comm.symbols_to_bits(s, 4)   # 送信ビット列 (BER の基準)

    # --- 入力パワー Pin を掃引。スパンが増えるほど ASE が増えるので範囲は -16〜+12 dBm と高め ---
    pin_dbm = np.arange(-16, 13, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))  # スパン数ごとに色を変える

    # --- 外側ループ: スパン数 nspan ごとに 1 本の BER 曲線を描く ---
    for nspan, c in zip(SPANS, colors):
        ber_sim, ber_th = [], []
        for pdbm in pin_dbm:               # 内側ループ: 各 Pin で BER を測る
            Pin = dbm_to_w(pdbm)
            A = np.sqrt(Pin) * s           # 1スパン目の入力光 (平均パワー Pin)
            for _ in range(nspan):                  # nspan 回の中継を順に適用
                A = fiber.loss_step(A, alpha, L_SPAN_KM)   # ファイバ損失で振幅を減衰
                A = fiber.edfa(A, G, NF_DB, RS, rng)       # EDFA: √G 倍に増幅 + ASE 雑音を加算
            r = A / np.sqrt(Pin)           # 信号成分を単位電力に戻して判定スケールへ
            bits_rx = comm.symbols_to_bits(r, 4)    # 硬判定して復調
            nb = min(len(bits_tx), len(bits_rx))
            ber_sim.append(comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb)  # 実測 BER
            snr_lin = Pin / (nspan * s_ase * RS)    # 理論 SNR_N = Pin/(N·S_ASE·Rs) (N台ぶんで雑音N倍)
            ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))    # 同 SNR の QPSK 理論 BER

        ber_sim = np.array(ber_sim)
        ax.semilogy(pin_dbm, ber_th, "-", color=c, lw=1.5)              # 理論: 実線
        m = ber_sim > 0                    # BER=0 は対数軸に描けないので除外
        ax.semilogy(pin_dbm[m], ber_sim[m], "o", color=c, ms=4, mfc="none",
                    label=f"{nspan} spans")                            # 実測: 白丸マーカー
        # --- このスパン数での受信感度 (BER=1e-3 を満たす最小 Pin) を対数補間で算出 ---
        v = ber_sim > 0
        # interp は x 昇順が必要。Pin↑ で BER↓ なので [::-1] で反転して BER 昇順にして逆引き
        sens = np.interp(np.log10(1e-3), np.log10(ber_sim[v][::-1]), pin_dbm[v][::-1])
        print(f"{nspan:2d} spans ({nspan*L_SPAN_KM:5.0f} km): 受信感度(BER=1e-3) ≈ {sens:6.1f} dBm")

    # --- 図の仕上げ: 横軸 Pin[dBm]、縦軸 BER(対数)。スパン数が増えると曲線が右へずれるのを確認 ---
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("span input power Pin [dBm]")   # 横軸: 各スパン入力パワー
    ax.set_ylabel("BER")                          # 縦軸: ビット誤り率
    ax.set_title("Multi-span (loss + EDFA) QPSK BER vs Pin\n(lines: theory, markers: simulation)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(title="relay spans")
    fig.tight_layout()
    out = os.path.join(HERE, "multispan_loss_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
