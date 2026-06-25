"""問題4-7: 損失 + 波長分散 + 非線形効果の多中継伝送、分散補償後の BER vs Pin

■ 何をする問題か
  問4-6 (損失 + 波長分散) に「非線形効果 (Kerr 効果, γ = 2.1 W⁻¹km⁻¹)」を加えた最も現実的な
  多中継伝送。各スパンを Split-Step Fourier 法 (SSFM) で「損失 + 分散 + 非線形」を同時に伝搬
  させ、EDFA でスパン損失を補償する。受信後、総波長分散だけをディジタル補償して復調し、
  BER の Pin 依存性を調べる。

■ なぜ BER に「最適パワー (U字曲線)」が現れるか (この問題の核心)
  - Pin を上げる → 信号が ASE 雑音に対して相対的に強くなり、ASE 制限の BER は改善する。
  - しかし非線形効果 (SPM) は光強度に比例して位相を回すため、Pin を上げるほど非線形ひずみが
    増大する。線形効果 (損失・分散) は受信側で補償できるが、非線形ひずみは分散補償では消えない。
  → 低 Pin では ASE 制限、高 Pin では非線形制限となり、その中間に BER が最小になる「最適入力
    パワー」が存在する。BER vs Pin は片対数で下に凸の U 字曲線になる。

■ モデル化の簡略化 (注意点)
  - 信号-ASE 間の非線形相互作用は2次的に小さいので、ここでは「無雑音で非線形伝搬 → 受信端で
    まとめて ASE を付加」(noiseless propagation + ASE loading) という標準的な近似を使う。
    ASE 量は問4-5/4-6 と同じ SNR_N = Pin/(N·S_ASE·Rs) になるよう add_awgn で与える。
  - SPM による決定論的な平均位相回転 (全シンボル共通の回転) は実機ならキャリア位相回復で
    除けるので、ここではデータ援用 (送信シンボル既知) で平均位相を 1 つだけ推定して取り除く。
  - 非線形は波形 (パルス整形後の連続波形) に効くので、問4-6 の 1 sps ではなく SPS=2 で
    オーバーサンプルし、レイズドコサインでパルス整形してから伝搬させる。
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
RS = 32e9             # 符号速度 [baud]
NF_DB = 4.0          # 各 EDFA の雑音指数 [dB]
L_SPAN_KM = 100.0    # 1スパンのファイバ長 [km]
ALPHA_DB = 0.3       # ファイバ損失 [dB/km]
BETA2 = -22.0         # 二次分散 β2 [ps²/km]
GAMMA = 2.1          # 非線形係数 γ [1/(W·km)] (Kerr 効果の強さ)
SPS = 2              # 1シンボルあたりサンプル数 (非線形を波形に正しく効かせるためオーバーサンプル)
BETA_RC = 0.1        # レイズドコサインのロールオフ係数 (帯域の広がり)
SPAN_FILT = 12       # パルス整形フィルタ長 [シンボル数]
STEPS_PER_SPAN = 10  # 1スパンを SSFM で割るステップ数 (多いほど高精度)
N_SYM = 16384        # 送信シンボル数 (SSFM は FFT を多用するので 2 の冪が効率的)
WARMUP = 2000        # 先頭・末尾で捨てるシンボル数 (フィルタ過渡や端の畳み込み歪みを除外)
SPANS = [1, 2, 5, 10, 20, 50]          # 比較する中継スパン数
SEED = 23            # 乱数シード


def dbm_to_w(dbm):
    """光パワーの単位を dBm → W に換算する (0 dBm = 1 mW)。"""
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (ALPHA_DB * L_SPAN_KM / 10.0)  # EDFA 利得 = スパン損失 (30 dB) の線形倍率
    alpha = fiber.alpha_from_db(ALPHA_DB)    # 損失 dB/km → 振幅減衰係数 α [1/km]
    s_ase = fiber.ase_psd(G, NF_DB)         # EDFA 1台あたりの ASE PSD [W/Hz]
    # 有効長 L_eff: 損失で減衰しながらでも非線形が効く「実効的な距離」。スパン長 100km でも
    # 損失のため実質 ~22km 程度しか非線形が蓄積しない。γ·L_eff が 1スパンあたりの非線形位相の目安。
    L_eff = (1 - np.exp(-alpha * L_SPAN_KM)) / alpha
    print(f"β2={BETA2} ps^2/km, γ={GAMMA} 1/(W·km), スパン {L_SPAN_KM:.0f} km")
    print(f"有効長 L_eff = {L_eff:.2f} km, スパン非線形位相係数 γ·L_eff = "
          f"{GAMMA*L_eff:.2f} rad/W/span\n")

    # --- 送信 QPSK シンボル列 (平均電力 1) ---
    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]

    # --- パルス整形: シンボル列を SPS=2 でアップサンプルし RC フィルタで連続波形に近づける ---
    # 非線形 (SPM) は瞬時パワー |A|² に効くので、矩形シンボルではなく実際の波形で伝搬させる必要がある。
    shaped, delay = comm.pulse_shape(s, SPS, BETA_RC, SPAN_FILT)  # delay は整形フィルタの群遅延 [サンプル]
    w = shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))     # 平均電力 1 に規格化した波形 (これに √Pin を掛けて使う)
    fs = SPS * RS                                          # 波形のサンプリングレート [Hz]
    dt_ps = 1e12 / fs                                      # サンプル間隔 [ps] (分散ステップの周波数グリッド用)
    omega = 2 * np.pi * np.fft.fftfreq(len(w), d=dt_ps)    # 角周波数 ω [rad/ps] (CD補償の dispersion_step 用)

    def propagate(nspan, Pin):
        """無雑音で nspan スパン伝搬 → CD補償 → ISI-free 抽出 → 平均位相除去。

        ASE 雑音はここでは加えず (noiseless propagation)、戻り値の単位電力シンボル列に対して
        呼び出し側で add_awgn する。残るのは非線形ひずみ (+ 補償残差) のみ。
        """
        A = np.sqrt(Pin) * w                           # 波形に √Pin を掛けて平均パワー Pin の光に
        for _ in range(nspan):                         # nspan 回の中継
            # 1スパンを SSFM で伝搬: 損失 + 分散 + 非線形 (SPM) を微小ステップで交互適用
            A = fiber.propagate_ssfm(A, L_SPAN_KM, dz=L_SPAN_KM / STEPS_PER_SPAN, dt=dt_ps,
                                     beta2=BETA2, gamma=GAMMA, alpha=alpha)
            A = A * np.sqrt(G)                          # EDFA 利得で損失を補償 (√G 倍。ASE は後で別途付加)
        # 受信側ディジタル分散補償: 累積分散の逆位相を一括で除去 (線形なので完全に戻る)
        A = fiber.dispersion_step(A, BETA2, -nspan * L_SPAN_KM, omega)
        rs = comm.downsample_isi_free(A, delay, SPS, N_SYM)  # 群遅延を起点に SPS 間隔でシンボル中心を抽出
        ph = np.angle(np.vdot(s, rs))                  # SPM による全シンボル共通の平均位相回転を推定 (データ援用)
        rs = rs * np.exp(-1j * ph)                      # その平均位相を取り除く (キャリア位相回復に相当)
        return rs / np.sqrt(np.mean(np.abs(rs) ** 2))  # 単位電力に正規化して返す

    def ber_of(r):
        """受信シンボル列 r の BER を測る。端 WARMUP シンボルは過渡なので除外して比較。"""
        rb = comm.symbols_to_bits(r[WARMUP:N_SYM - WARMUP], 4)  # 受信を硬判定して復調
        tb = comm.symbols_to_bits(s[WARMUP:N_SYM - WARMUP], 4)  # 対応区間の送信ビット
        n = min(len(rb), len(tb))
        return comm.count_bit_errors(tb[:n], rb[:n]) / n

    # --- 入力パワー Pin を掃引。U字曲線を捉えるため低〜高パワーを 1.5dB 刻みで見る ---
    pin_dbm = np.arange(-12, 9, 1.5)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))
    cap = {}     # N=10 の代表 3 点 (低Pin/最適/高Pin) のコンステレーションを保存

    # 標準出力: スパン数ごとに「BER が最小になる最適 Pin」と「その最小 BER」を表示
    print(f"{'spans':>6} {'最適Pin[dBm]':>12} {'最小BER':>12}")
    for nspan, c in zip(SPANS, colors):     # 外側ループ: スパン数ごとに 1 本の U 字曲線
        ber_nl, ber_lin_th = [], []
        for pdbm in pin_dbm:                # 内側ループ: 各 Pin
            Pin = dbm_to_w(pdbm)
            snr_ase = Pin / (nspan * s_ase * RS)           # ASE のみの理論 SNR (問4-5/4-6 と同じ)
            r = propagate(nspan, Pin)                      # 非線形ひずみ込みで伝搬 (無雑音)
            # 上で求めた ASE 限界 SNR ぶんの AWGN を受信端でまとめて付加 (ASE loading)
            r = comm.add_awgn(r, 10 * np.log10(snr_ase), signal_power=1.0, rng=rng)
            ber_nl.append(ber_of(r))                       # 非線形 + ASE の実測 BER
            ber_lin_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_ase))))  # 線形 (ASEのみ) 理論 BER
            # N=10 の代表 3 点だけコンステレーションを保存 (Pin に最も近い掃引点を拾う)
            if nspan == 10 and abs(pdbm - (-6.0)) < 0.75:
                cap["low"] = r            # 低Pin: ASE 制限 (点が雑音で太る)
            if nspan == 10 and abs(pdbm - 0.0) < 0.75:
                cap["opt"] = r            # 最適: 雑音と非線形のバランスが良い
            if nspan == 10 and abs(pdbm - 6.0) < 0.75:
                cap["high"] = r           # 高Pin: 非線形制限 (点が回転・歪む)

        ber_nl = np.array(ber_nl)
        ax.semilogy(pin_dbm, ber_lin_th, "--", color=c, lw=1.0, alpha=0.6)       # 線形限界 (破線)
        m = ber_nl > 0
        ax.semilogy(pin_dbm[m], ber_nl[m], "o-", color=c, ms=4, label=f"{nspan} spans")  # 非線形 (実線+丸, U字)
        imin = int(np.argmin(ber_nl))      # BER 最小の掃引点 = 最適入力パワー
        print(f"{nspan:6d} {pin_dbm[imin]:12.1f} {ber_nl[imin]:12.2e}")

    # --- BER 図: U字 (実線) が破線 (線形限界) から右側 (高Pin) で乖離 = 非線形ペナルティ ---
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("span input power Pin [dBm]")
    ax.set_ylabel("BER")
    ax.set_title("Loss + CD + nonlinearity, after CD compensation\n"
                 "solid+markers = nonlinear (U-shape), dashed = linear (ASE only)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(title="relay spans", ncol=2, fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "multispan_nl_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\nBER図を保存しました: {out}")

    # --- N=10 のコンステレーション (低Pin / 最適 / 高Pin) ---
    # 同じ 10 スパンでも Pin により劣化の質が変わるのを可視化:
    #   低Pin = ASE 制限 (点が一様に太る), 高Pin = 非線形制限 (点が捻れる/広がる), 最適 = その中間で最もクリア。
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    titles = [("low", "Pin=-6 dBm: ASE-limited"),
              ("opt", "Pin=0 dBm: optimum"),
              ("high", "Pin=+6 dBm: nonlinearity-limited")]
    for ax2, (k, title) in zip(axes, titles):
        d = cap[k]
        ax2.scatter(d[WARMUP:WARMUP + 4000].real, d[WARMUP:WARMUP + 4000].imag,
                    s=4, alpha=0.15, color="C0")       # WARMUP 以降 4000 点を I-Q 平面に散布
        ax2.set_title(title)
        ax2.set_xlabel("I"); ax2.set_ylabel("Q"); ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")
        ax2.set_xlim(-2.2, 2.2); ax2.set_ylim(-2.2, 2.2)
    fig2.suptitle("QPSK after 10 spans (loss+CD+NL, CD compensated)", fontsize=12)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "nl_constellation.png")
    fig2.savefig(out2, dpi=120)
    print(f"コンステレーション図を保存しました: {out2}")


if __name__ == "__main__":
    main()
