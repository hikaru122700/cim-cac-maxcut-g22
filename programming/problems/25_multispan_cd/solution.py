"""問題4-6: 損失 + 波長分散の多中継伝送、ディジタル分散補償後の BER vs Pin

問4-5 の構成に「波長分散 (β2 = -22 ps^2/km)」を加える。EDFA はスパン損失を完全補償。
受信後、総波長分散をディジタルで完全補償してから復調する。

波長分散は線形・全域通過 (|H(ω)|=1) の位相フィルタなので、逆位相 exp(-jβ2ω²z_total/2) を
掛ければ完全に除去できる。雑音 (ASE) は全域通過フィルタを通っても分散が変わらない。
したがって分散補償後の BER は、分散の無い問4-5 と一致するはずである。
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
BETA2 = -22.0          # 二次分散 [ps^2/km]
N_SYM = 100_000
SPANS = [1, 2, 5, 10, 20, 50]
SEED = 22


def dbm_to_w(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (ALPHA_DB * L_SPAN_KM / 10.0)
    alpha = fiber.alpha_from_db(ALPHA_DB)
    s_ase = fiber.ase_psd(G, NF_DB)

    # 1 sample/symbol: 時間刻み dt = 1/Rs。ps 単位に換算して β2[ps^2/km] と整合させる。
    dt_ps = 1e12 / RS
    omega = 2 * np.pi * np.fft.fftfreq(N_SYM, d=dt_ps)    # [rad/ps]

    print(f"β2={BETA2} ps^2/km, スパン {L_SPAN_KM:.0f} km, 損失/利得 {ALPHA_DB*L_SPAN_KM:.0f} dB, "
          f"NF {NF_DB} dB\n")

    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]
    bits_tx = comm.symbols_to_bits(s, 4)

    pin_dbm = np.arange(-16, 13, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))

    cap = {}     # 後段の図用に N=10 の伝搬途中・補償後を保存
    for nspan, c in zip(SPANS, colors):
        ber_sim, ber_th = [], []
        for pdbm in pin_dbm:
            Pin = dbm_to_w(pdbm)
            A = np.sqrt(Pin) * s
            for _ in range(nspan):
                A = fiber.dispersion_step(A, BETA2, L_SPAN_KM, omega)  # 分散
                A = fiber.loss_step(A, alpha, L_SPAN_KM)               # 損失
                A = fiber.edfa(A, G, NF_DB, RS, rng)                   # 増幅 + ASE
            A_disp = A.copy()
            A = fiber.dispersion_step(A, BETA2, -nspan * L_SPAN_KM, omega)  # 総分散補償
            r = A / np.sqrt(Pin)
            bits_rx = comm.symbols_to_bits(r, 4)
            nb = min(len(bits_tx), len(bits_rx))
            ber_sim.append(comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb)
            snr_lin = Pin / (nspan * s_ase * RS)
            ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))
            if nspan == 10 and abs(pdbm - 2.0) < 1e-9:
                cap["before"] = A_disp / np.sqrt(Pin)
                cap["after"] = r

        ber_sim = np.array(ber_sim)
        ax.semilogy(pin_dbm, ber_th, "-", color=c, lw=1.5)
        m = ber_sim > 0
        ax.semilogy(pin_dbm[m], ber_sim[m], "o", color=c, ms=4, mfc="none",
                    label=f"{nspan} spans")
        v = ber_sim > 0
        sens = np.interp(np.log10(1e-3), np.log10(ber_sim[v][::-1]), pin_dbm[v][::-1])
        print(f"{nspan:2d} spans ({nspan*L_SPAN_KM:5.0f} km): 受信感度(BER=1e-3) ≈ {sens:6.1f} dBm")

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
    fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.5))
    a1.scatter(cap["before"][:4000].real, cap["before"][:4000].imag, s=4, alpha=0.15, color="C3")
    a1.set_title("before CD compensation (10 spans)\nQPSK smeared by dispersion (ISI)")
    a2.scatter(cap["after"][:4000].real, cap["after"][:4000].imag, s=4, alpha=0.15, color="C0")
    a2.set_title("after CD compensation\nclean QPSK + ASE")
    for a in (a1, a2):
        a.set_xlabel("I"); a.set_ylabel("Q"); a.set_aspect("equal")
        a.grid(True, alpha=0.3); a.tick_params(direction="in")
        a.set_xlim(-2.5, 2.5); a.set_ylim(-2.5, 2.5)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "cd_compensation.png")
    fig2.savefig(out2, dpi=120)
    print(f"分散補償前後の図を保存しました: {out2}")


if __name__ == "__main__":
    main()
