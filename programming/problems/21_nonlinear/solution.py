"""問題4-2: 非線形効果 (自己位相変調 SPM) による位相変化

標準単一モード光ファイバ (非線形係数 γ = 2.1 W^-1 km^-1) を、非線形効果のみ考えて伝搬させる。
強度ピーク 1 W、強度 FWHM 10 ps のガウスパルスを入力し、ピーク位置における位相変化量の
z 依存性を求める。さらに解析解と比較する。

解析解 (分散・損失を無視):
    |A| は変化せず、位相だけが φ(z,T) = γ|A(0,T)|² z 増える (自己位相変調)。
    ピーク (T=0, |A|²=P0) では  φ_max(z) = γ P0 z  (z に比例)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import fiber  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
GAMMA = 2.1            # 非線形係数 [1/(W·km)]
P_PEAK = 1.0
FWHM0 = 10.0


def main() -> None:
    N = 2 ** 14
    Tspan = 400.0
    t = (np.arange(N) - N / 2) * (Tspan / N)
    dt = Tspan / N

    A0 = fiber.gaussian_pulse(t, P_PEAK, FWHM0)
    i_peak = np.argmax(np.abs(A0))
    L_NL = 1.0 / (GAMMA * P_PEAK)
    print(f"γ = {GAMMA} 1/(W·km), ピーク強度 = {P_PEAK} W")
    print(f"非線形長 L_NL = 1/(γ P0) = {L_NL:.4f} km\n")

    z_list = np.linspace(0, 5, 51)
    phi_sim = []
    for z in z_list:
        A = fiber.nonlinear_step(A0, GAMMA, z)        # 非線形のみ (分散なし)
        # ピーク位置の位相 (基準 A0 に対する増分)
        phi = np.angle(A[i_peak] * np.conj(A0[i_peak]))
        phi_sim.append(phi)
    phi_sim = np.unwrap(np.array(phi_sim))
    phi_theory = GAMMA * P_PEAK * z_list

    print(f"{'z [km]':>8} {'φ_peak(sim) [rad]':>18} {'φ_peak(theory) [rad]':>22}")
    for z in (0, 1, 2, 5):
        idx = int(np.argmin(np.abs(z_list - z)))     # 連続unwrap済み配列から取り出す
        print(f"{z:8.1f} {phi_sim[idx]:18.3f} {phi_theory[idx]:22.3f}")

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(z_list, phi_theory, "r-", lw=2, label="analytic  γ·P0·z")
    ax1.plot(z_list, phi_sim, "ko", ms=4, mfc="none", label="simulation")
    ax1.set_xlabel("distance z [km]"); ax1.set_ylabel("peak phase shift [rad]")
    ax1.set_title("SPM peak phase shift vs distance")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) SPM によるスペクトル広がり
    def spectrum(A):
        S = np.abs(np.fft.fftshift(np.fft.fft(A))) ** 2
        return S / S.max()
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))    # [1/ps = THz]
    ax2.plot(f * 1e3, 10 * np.log10(spectrum(A0) + 1e-12), label="z = 0 km")
    for z in (2, 5):
        A = fiber.nonlinear_step(A0, GAMMA, z)
        ax2.plot(f * 1e3, 10 * np.log10(spectrum(A) + 1e-12), label=f"z = {z} km")
    ax2.set_xlim(-200, 200); ax2.set_ylim(-40, 2)
    ax2.set_xlabel("frequency offset [GHz]"); ax2.set_ylabel("normalized spectrum [dB]")
    ax2.set_title("SPM-induced spectral broadening")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "nonlinear.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    max_err = np.max(np.abs(phi_sim - phi_theory))
    print(f"数値 vs 解析解 最大絶対誤差 = {max_err:.2e} rad")


if __name__ == "__main__":
    main()
