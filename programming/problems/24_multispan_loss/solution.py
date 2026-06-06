"""問題4-5: 損失 + EDFA 多中継伝送での QPSK の BER vs Pin

単一偏波 QPSK。1スパン = 標準SMF 100 km(損失のみ、0.3 dB/km = 30 dB)+ EDFA(損失を完全補償、
利得 30 dB、NF=4 dB)。各スパンの入力パワーは同一(Pin)。1,2,5,10,20,50 スパン中継後にコヒーレント受信。
BER の Pin 依存性をシミュレーションする(レーザ位相雑音は無視)。

各 EDFA が同じ ASE を加えるので、Nスパン後の雑音は1スパンの N 倍 → SNR_N = SNR_1 / N。
よって BER 曲線はスパン数が2倍になるごとに約 3 dB 右へずれる。
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
ALPHA_DB = 0.3              # [dB/km]
SPAN_LOSS_DB = ALPHA_DB * L_SPAN_KM    # 30 dB
N_SYM = 100_000
SPANS = [1, 2, 5, 10, 20, 50]
SEED = 21


def dbm_to_w(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (SPAN_LOSS_DB / 10.0)         # 利得 = スパン損失
    alpha = fiber.alpha_from_db(ALPHA_DB)
    s_ase = fiber.ase_psd(G, NF_DB)
    print(f"1スパン: {L_SPAN_KM:.0f} km, 損失 {SPAN_LOSS_DB:.0f} dB, 利得 {SPAN_LOSS_DB:.0f} dB, "
          f"NF {NF_DB} dB")
    print(f"S_ASE(1台) = {s_ase:.3e} W/Hz, 1スパン雑音電力 S_ASE·Rs = {s_ase*RS:.3e} W\n")

    # QPSK シンボル
    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]
    bits_tx = comm.symbols_to_bits(s, 4)

    pin_dbm = np.arange(-16, 13, 1.0)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(SPANS)))

    for nspan, c in zip(SPANS, colors):
        ber_sim, ber_th = [], []
        for pdbm in pin_dbm:
            Pin = dbm_to_w(pdbm)
            A = np.sqrt(Pin) * s
            for _ in range(nspan):                  # 多中継
                A = fiber.loss_step(A, alpha, L_SPAN_KM)   # ファイバ損失
                A = fiber.edfa(A, G, NF_DB, RS, rng)       # EDFA: 増幅 + ASE
            r = A / np.sqrt(Pin)                    # 信号を単位電力へ
            bits_rx = comm.symbols_to_bits(r, 4)
            nb = min(len(bits_tx), len(bits_rx))
            ber_sim.append(comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb)
            snr_lin = Pin / (nspan * s_ase * RS)    # SNR_N = Pin/(N·S_ASE·Rs)
            ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))

        ber_sim = np.array(ber_sim)
        ax.semilogy(pin_dbm, ber_th, "-", color=c, lw=1.5)
        m = ber_sim > 0
        ax.semilogy(pin_dbm[m], ber_sim[m], "o", color=c, ms=4, mfc="none",
                    label=f"{nspan} spans")
        # BER=1e-3 感度
        v = ber_sim > 0
        sens = np.interp(np.log10(1e-3), np.log10(ber_sim[v][::-1]), pin_dbm[v][::-1])
        print(f"{nspan:2d} spans ({nspan*L_SPAN_KM:5.0f} km): 受信感度(BER=1e-3) ≈ {sens:6.1f} dBm")

    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("span input power Pin [dBm]")
    ax.set_ylabel("BER")
    ax.set_title("Multi-span (loss + EDFA) QPSK BER vs Pin\n(lines: theory, markers: simulation)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend(title="relay spans")
    fig.tight_layout()
    out = os.path.join(HERE, "multispan_loss_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
