"""問題4-6: 損失 + 波長分散の多中継伝送、ディジタル分散補償後の BER vs Pin

■ 何をする問題か
  問4-5 (損失 + EDFA 多中継) の構成に「波長分散 (CD: chromatic dispersion, β2 = -22 ps²/km)」
  を加える。波長分散はパルスを時間方向に広げ、隣接シンボルどうしを重ねてしまう (ISI)。
  各スパンで「分散 → 損失 → EDFA (損失を完全補償)」を適用し、受信後に総波長分散を
  ディジタル信号処理で完全補償してから復調する。BER の Pin 依存性を問4-5 と比べる。

■ なぜ分散補償後は問4-5 と一致するはずか (理論の要点)
  波長分散は線形で全域通過 (|H(ω)|=1) の純粋な位相フィルタ。伝送で掛かった位相
  exp(j β2 ω² z_total / 2) の逆位相 exp(-j β2 ω² z_total / 2) を受信側で掛ければ、信号の
  ISI は完全に元へ戻る。さらに ASE 雑音は全域通過フィルタを通っても電力 (分散) が変わらない
  ため、分散補償をしても雑音は増えも減りもしない。
  → 分散補償後の BER は、分散の無い問4-5 (ASE のみで決まる SNR_N = Pin/(N·S_ASE·Rs)) と
    一致するはず。これがこの問題の確認ポイント。
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
RS = 32e9              # 符号速度 [baud]。1 sps なので帯域 fs = Rs
NF_DB = 4.0           # 各 EDFA の雑音指数 [dB]
L_SPAN_KM = 100.0     # 1スパンのファイバ長 [km]
ALPHA_DB = 0.3        # ファイバ損失 [dB/km]
BETA2 = -22.0          # 二次分散 β2 [ps²/km] (標準SMF。負号は異常分散領域)
N_SYM = 100_000       # 送信シンボル数
SPANS = [1, 2, 5, 10, 20, 50]          # 比較する中継スパン数
SEED = 22             # 乱数シード


def dbm_to_w(dbm):
    """光パワーの単位を dBm → W に換算する (0 dBm = 1 mW)。"""
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (ALPHA_DB * L_SPAN_KM / 10.0)  # EDFA 利得 = スパン損失 (30 dB を線形倍率に)
    alpha = fiber.alpha_from_db(ALPHA_DB)    # 損失 dB/km → 振幅減衰係数 α [1/km]
    s_ase = fiber.ase_psd(G, NF_DB)         # EDFA 1台あたりの ASE PSD [W/Hz]

    # 分散ステップは周波数領域で計算するので、FFT 用の角周波数グリッド ω を用意する。
    # 1 sample/symbol: サンプル間隔 dt = 1/Rs。ここでは ps 単位に換算して β2[ps²/km] と
    # 単位を整合させる (fiber 側の単位系は [ps],[km],[W])。
    dt_ps = 1e12 / RS                                    # サンプル間隔 [ps] (= 10^12 / Rs[Hz])
    omega = 2 * np.pi * np.fft.fftfreq(N_SYM, d=dt_ps)    # 角周波数 ω [rad/ps] (FFT 折り返し順)

    print(f"β2={BETA2} ps^2/km, スパン {L_SPAN_KM:.0f} km, 損失/利得 {ALPHA_DB*L_SPAN_KM:.0f} dB, "
          f"NF {NF_DB} dB\n")

    # --- 送信 QPSK シンボル列 (平均電力 1) ---
    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]
    bits_tx = comm.symbols_to_bits(s, 4)    # 送信ビット列 (BER の基準)

    pin_dbm = np.arange(-16, 13, 1.0)       # 入力パワー掃引範囲 [dBm]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))  # スパン数ごとの色

    cap = {}     # 後段のコンステレーション図用に N=10 の「補償前/補償後」波形を保存
    for nspan, c in zip(SPANS, colors):     # 外側ループ: スパン数ごとに 1 本の BER 曲線
        ber_sim, ber_th = [], []
        for pdbm in pin_dbm:                # 内側ループ: 各 Pin で BER を測定
            Pin = dbm_to_w(pdbm)
            A = np.sqrt(Pin) * s            # 1スパン目の入力光
            for _ in range(nspan):          # nspan 回の中継: 分散 → 損失 → EDFA の順
                A = fiber.dispersion_step(A, BETA2, L_SPAN_KM, omega)  # 波長分散 (2次位相を付加し ISI 発生)
                A = fiber.loss_step(A, alpha, L_SPAN_KM)               # ファイバ損失で振幅減衰
                A = fiber.edfa(A, G, NF_DB, RS, rng)                   # EDFA: √G 倍 + ASE 雑音
            A_disp = A.copy()               # 分散補償する直前 (ISI で乱れた状態) を保存
            # 受信側ディジタル分散補償: 累積分散 (β2 × 総距離) の逆位相を一括で掛けて ISI を除去
            A = fiber.dispersion_step(A, BETA2, -nspan * L_SPAN_KM, omega)  # 距離に負号 → 逆位相
            r = A / np.sqrt(Pin)            # 信号を単位電力に正規化
            bits_rx = comm.symbols_to_bits(r, 4)    # 硬判定して復調
            nb = min(len(bits_tx), len(bits_rx))
            ber_sim.append(comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb)  # 実測 BER
            snr_lin = Pin / (nspan * s_ase * RS)    # 理論 SNR_N (問4-5 と同じ ASE 限界。分散は補償で消える)
            ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))
            # N=10・Pin=2dBm の点だけ、補償前後のコンステレーションを後で図示するため保存
            if nspan == 10 and abs(pdbm - 2.0) < 1e-9:
                cap["before"] = A_disp / np.sqrt(Pin)   # 補償前 (分散で滲んだ QPSK)
                cap["after"] = r                        # 補償後 (きれいな QPSK + ASE)

        ber_sim = np.array(ber_sim)
        ax.semilogy(pin_dbm, ber_th, "-", color=c, lw=1.5)              # 理論 (= 問4-5 の ASE 限界): 実線
        m = ber_sim > 0
        ax.semilogy(pin_dbm[m], ber_sim[m], "o", color=c, ms=4, mfc="none",
                    label=f"{nspan} spans")                            # 実測: 白丸 (理論と重なれば補償成功)
        v = ber_sim > 0
        # 受信感度 (BER=1e-3 を満たす最小 Pin) を対数補間で逆引き
        sens = np.interp(np.log10(1e-3), np.log10(ber_sim[v][::-1]), pin_dbm[v][::-1])
        print(f"{nspan:2d} spans ({nspan*L_SPAN_KM:5.0f} km): 受信感度(BER=1e-3) ≈ {sens:6.1f} dBm")

    # --- BER 図: 横軸 Pin、縦軸 BER。実測(丸)が理論(線=ASEのみ)に重なれば分散補償が完璧な証拠 ---
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("span input power Pin [dBm]")
    ax.set_ylabel("BER")
    ax.set_title("Loss + CD multi-span, with digital CD compensation\n"
                 "(matches loss-only case: lines = ASE theory)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(title="relay spans")
    fig.tight_layout()
    out = os.path.join(HERE, "multispan_cd_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER図を保存しました: {out}")

    # --- 分散補償の前後コンステレーション (10スパン, Pin=2dBm) ---
    # 補償前は分散による ISI で QPSK の 4 点が円状に滲み、補償後は 4 点に戻る (ASE のばらつきだけ残る)。
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5))
    a1.scatter(cap["before"][:4000].real, cap["before"][:4000].imag, s=4, alpha=0.15, color="C3")
    a1.set_title("before CD compensation (10 spans)\nQPSK smeared by dispersion (ISI)")
    a2.scatter(cap["after"][:4000].real, cap["after"][:4000].imag, s=4, alpha=0.15, color="C0")
    a2.set_title("after CD compensation\nclean QPSK + ASE")
    for a in (a1, a2):                       # 両図の体裁を揃える (I-Q 平面、等方アスペクト)
        a.set_xlabel("I"); a.set_ylabel("Q"); a.set_aspect("equal")
        a.grid(True, alpha=0.3); a.tick_params(direction="in")
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "cd_compensation.png")
    fig2.savefig(out2, dpi=120)
    print(f"分散補償前後の図を保存しました: {out2}")


if __name__ == "__main__":
    main()
