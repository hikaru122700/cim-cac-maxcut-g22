"""問題4-7: 損失 + 波長分散 + 非線形効果の多中継伝送、分散補償後の BER vs Pin

問4-6 に「非線形効果 (γ = 2.1 W^-1 km^-1)」を加える。各スパンを Split-Step Fourier 法で
(損失 + 分散 + 非線形) 伝搬させ、EDFA で損失を補償。受信後、総波長分散を完全補償して復調する。

線形効果 (損失・分散) は補償で消えるが、非線形効果は分散補償では消えない。入力パワー Pin を
上げると ASE 制限が改善する一方で非線形ひずみが増えるため、BER に最適パワー (U字曲線) が現れる。

注: 信号-ASE 間の非線形相互作用は2次的なので、ここでは「無雑音で非線形伝搬 → 受信端で ASE 付加」
(noiseless propagation + ASE loading) という標準的な簡略化を用いる。また SPM による決定論的な
平均位相回転はキャリア位相回復で除けるので、データ援用で1つだけ取り除く。
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
RS = 32e9
NF_DB = 4.0
L_SPAN_KM = 100.0
ALPHA_DB = 0.3
BETA2 = -22.0
GAMMA = 2.1
SPS = 2
BETA_RC = 0.1
SPAN_FILT = 12
STEPS_PER_SPAN = 10
N_SYM = 16384
WARMUP = 2000
SPANS = [1, 2, 5, 10, 20, 50]
SEED = 23


def dbm_to_w(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (ALPHA_DB * L_SPAN_KM / 10.0)
    alpha = fiber.alpha_from_db(ALPHA_DB)
    s_ase = fiber.ase_psd(G, NF_DB)
    L_eff = (1 - np.exp(-alpha * L_SPAN_KM)) / alpha
    print(f"β2={BETA2} ps^2/km, γ={GAMMA} 1/(W·km), スパン {L_SPAN_KM:.0f} km")
    print(f"有効長 L_eff = {L_eff:.2f} km, スパン非線形位相係数 γ·L_eff = "
          f"{GAMMA*L_eff:.2f} rad/W/span\n")

    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]

    shaped, delay = comm.pulse_shape(s, SPS, BETA_RC, SPAN_FILT)
    w = shaped / np.sqrt(np.mean(np.abs(shaped) ** 2))     # 単位平均電力の波形
    fs = SPS * RS
    dt_ps = 1e12 / fs
    omega = 2 * np.pi * np.fft.fftfreq(len(w), d=dt_ps)

    def propagate(nspan, Pin):
        """無雑音で nspan スパン伝搬 → CD補償 → ISI-free 抽出 → 平均位相除去。"""
        A = np.sqrt(Pin) * w
        for _ in range(nspan):
            A = fiber.propagate_ssfm(A, L_SPAN_KM, dz=L_SPAN_KM / STEPS_PER_SPAN, dt=dt_ps,
                                     beta2=BETA2, gamma=GAMMA, alpha=alpha)
            A = A * np.sqrt(G)                              # EDFA 利得 (雑音は後で)
        A = fiber.dispersion_step(A, BETA2, -nspan * L_SPAN_KM, omega)  # 総分散補償
        rs = comm.downsample_isi_free(A, delay, SPS, N_SYM)
        ph = np.angle(np.vdot(s, rs))                      # SPM 平均位相回転
        rs = rs * np.exp(-1j * ph)
        return rs / np.sqrt(np.mean(np.abs(rs) ** 2))      # 単位電力に正規化

    def ber_of(r):
        rb = comm.symbols_to_bits(r[WARMUP:N_SYM - WARMUP], 4)
        tb = comm.symbols_to_bits(s[WARMUP:N_SYM - WARMUP], 4)
        n = min(len(rb), len(tb))
        return comm.count_bit_errors(tb[:n], rb[:n]) / n

    pin_dbm = np.arange(-12, 9, 1.5)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))
    cap = {}

    print(f"{'spans':>6} {'最適Pin[dBm]':>12} {'最小BER':>12}")
    for nspan, c in zip(SPANS, colors):
        ber_nl, ber_lin_th = [], []
        for pdbm in pin_dbm:
            Pin = dbm_to_w(pdbm)
            snr_ase = Pin / (nspan * s_ase * RS)           # 問4-5/4-6 と同じ ASE 限界
            r = propagate(nspan, Pin)                      # 非線形ひずみ (無雑音)
            r = comm.add_awgn(r, 10 * np.log10(snr_ase), signal_power=1.0, rng=rng)
            ber_nl.append(ber_of(r))
            ber_lin_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_ase))))
            if nspan == 10 and abs(pdbm - (-6.0)) < 0.75:
                cap["low"] = r
            if nspan == 10 and abs(pdbm - 0.0) < 0.75:
                cap["opt"] = r
            if nspan == 10 and abs(pdbm - 6.0) < 0.75:
                cap["high"] = r

        ber_nl = np.array(ber_nl)
        ax.semilogy(pin_dbm, ber_lin_th, "--", color=c, lw=1.0, alpha=0.6)
        m = ber_nl > 0
        ax.semilogy(pin_dbm[m], ber_nl[m], "o-", color=c, ms=4, label=f"{nspan} spans")
        imin = int(np.argmin(ber_nl))
        print(f"{nspan:6d} {pin_dbm[imin]:12.1f} {ber_nl[imin]:12.2e}")

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
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    titles = [("low", "Pin=-6 dBm: ASE-limited"),
              ("opt", "Pin=0 dBm: optimum"),
              ("high", "Pin=+6 dBm: nonlinearity-limited")]
    for ax2, (k, title) in zip(axes, titles):
        d = cap[k]
        ax2.scatter(d[WARMUP:WARMUP + 4000].real, d[WARMUP:WARMUP + 4000].imag,
                    s=4, alpha=0.15, color="C0")
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
