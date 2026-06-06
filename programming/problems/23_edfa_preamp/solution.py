"""問題4-4: EDFA 前置増幅構成での QPSK の BER vs 入力パワー Pin

単一偏波 QPSK 光信号を EDFA で増幅した後、コヒーレント光受信する (前置増幅構成)。
EDFA の ASE 雑音は複素 AWGN でモデル化し、雑音指数 NF = 4 dB とする。
BER の EDFA 入力パワー Pin 依存性を求める (レーザ位相雑音は無視)。

物理:
  EDFA出力 ASE PSD (1偏波): S_ASE = (NF_lin/2)·hν·(G-1)
  シンボルあたり (1 sps, fs=Rs) の雑音電力 = S_ASE·Rs、信号電力 = G·Pin
  -> 1シンボルSNR  Es/N0 = G·Pin/(S_ASE·Rs) ≈ 2·Pin/(NF_lin·hν·Rs)  (高利得)
  QPSK BER = Q(√(Es/N0))。
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
RS = 32e9              # 符号速度 [baud]
NF_DB = 4.0           # 雑音指数 [dB]
GAIN_DB = 20.0        # EDFA 利得 [dB] (前置増幅、高利得)
N_SYM = 400_000
SEED = 20


def dbm_to_w(dbm):
    return 10 ** (dbm / 10.0) * 1e-3


def main() -> None:
    rng = np.random.default_rng(SEED)
    G = 10 ** (GAIN_DB / 10.0)
    s_ase = fiber.ase_psd(G, NF_DB)
    hnu = fiber.photon_energy()
    print(f"Rs={RS/1e9:.0f} Gbaud, NF={NF_DB} dB, G={GAIN_DB} dB")
    print(f"hν = {hnu:.3e} J, S_ASE = {s_ase:.3e} W/Hz\n")

    # QPSK シンボル (単位電力)
    prbs = comm.generate_prbs(15)
    block = comm.bits_to_symbols(np.tile(prbs, 2), 4)
    s = np.tile(block, int(np.ceil(N_SYM / len(block))))[:N_SYM]
    bits_tx = comm.symbols_to_bits(s, 4)

    pin_dbm = np.arange(-50, -37, 1.0)
    ber_sim, ber_th, snr_db_arr = [], [], []
    for pdbm in pin_dbm:
        Pin = dbm_to_w(pdbm)
        A_in = np.sqrt(Pin) * s                      # 入力光 (√W)
        A_out = fiber.edfa(A_in, G, NF_DB, RS, rng)  # 増幅 + ASE
        r = A_out / np.sqrt(G * Pin)                 # 正規化して判定
        bits_rx = comm.symbols_to_bits(r, 4)
        nb = min(len(bits_tx), len(bits_rx))
        ber = comm.count_bit_errors(bits_tx[:nb], bits_rx[:nb]) / nb

        snr_lin = G * Pin / (s_ase * RS)             # Es/N0
        ber_sim.append(ber)
        ber_th.append(float(comm.ber_theory_qam(4, 10 * np.log10(snr_lin))))
        snr_db_arr.append(10 * np.log10(snr_lin))

    print(f"{'Pin[dBm]':>9} {'Es/N0[dB]':>10} {'BER(sim)':>11} {'BER(theory)':>12}")
    for p, sn, bs, bt in zip(pin_dbm, snr_db_arr, ber_sim, ber_th):
        print(f"{p:9.0f} {sn:10.2f} {bs:11.2e} {bt:12.2e}")

    # ---- 図 ----
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.semilogy(pin_dbm, ber_th, "r-", lw=2, label="theory")
    sim = np.array(ber_sim)
    m = sim > 0
    ax.semilogy(pin_dbm[m], sim[m], "ko", ms=5, mfc="none", label="simulation")
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("EDFA input power Pin [dBm]")
    ax.set_ylabel("BER")
    ax.set_title(f"Preamplified QPSK BER vs Pin (NF={NF_DB} dB, Rs={RS/1e9:.0f} Gbaud)")
    ax.grid(True, which="both", alpha=0.3); ax.tick_params(direction="in", which="both")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(HERE, "edfa_preamp_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    # BER=1e-3 となる受信感度 (補間)
    sim_arr = np.array(ber_sim)
    valid = sim_arr > 0
    sens = np.interp(np.log10(1e-3), np.log10(sim_arr[valid][::-1]), pin_dbm[valid][::-1])
    print(f"受信感度 (BER=1e-3) ≈ {sens:.1f} dBm")


if __name__ == "__main__":
    main()
