"""問題4-3: 基本ソリトン (N=1) の伝搬

波長分散と非線形効果の双方を考える。ソリトン次数 N=1 となる sech パルスを伝搬させ、
伝搬に伴い時間波形が変化しないことを確認する。

ソリトン次数:  N^2 = γ P0 T0^2 / |β2|
N=1 の条件:    P0 = |β2| / (γ T0^2)

分散 (パルスを広げる) と 非線形 SPM (チャープでスペクトルを広げ、異常分散下で圧縮する)
がちょうど釣り合い、形が崩れない。比較のため「分散のみ」の場合(広がる)も示す。
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
BETA2 = -22.0          # 異常分散 (ソリトンに必須) [ps^2/km]
GAMMA = 2.1            # [1/(W·km)]
FWHM0 = 10.0           # 強度FWHM [ps]


def main() -> None:
    N = 2 ** 13
    Tspan = 400.0
    t = (np.arange(N) - N / 2) * (Tspan / N)
    dt = Tspan / N

    t0 = fiber.t0_from_fwhm_sech(FWHM0)
    P0 = abs(BETA2) / (GAMMA * t0 ** 2)             # N=1 条件のピーク強度
    Nsol = np.sqrt(GAMMA * P0 * t0 ** 2 / abs(BETA2))
    LD = fiber.dispersion_length(t0, BETA2)
    z0 = np.pi / 2 * LD                             # ソリトン周期
    print(f"β2={BETA2} ps^2/km, γ={GAMMA} 1/(W·km), FWHM={FWHM0} ps")
    print(f"T0(sech) = {t0:.3f} ps")
    print(f"N=1 のピーク強度 P0 = |β2|/(γ T0^2) = {P0:.4f} W")
    print(f"確認: ソリトン次数 N = {Nsol:.4f}")
    print(f"分散長 L_D = {LD:.3f} km, ソリトン周期 z0 = (π/2)L_D = {z0:.3f} km\n")

    A0 = fiber.sech_pulse(t, P0, FWHM0)
    z_total = 5 * z0                                # 5周期ぶん伝搬
    z_pts = np.linspace(0, z_total, 31)

    fwhm_sol, fwhm_disp = [], []
    for z in z_pts:
        # ソリトン (分散+非線形)
        A_sol = fiber.propagate_ssfm(A0, z, dz=z0 / 200, dt=dt,
                                     beta2=BETA2, gamma=GAMMA)
        fwhm_sol.append(fiber.fwhm_of(t, np.abs(A_sol) ** 2))
        # 比較: 分散のみ (非線形なし)
        A_disp = fiber.dispersion_step(A0, BETA2, z, fiber.omega_grid(N, dt))
        fwhm_disp.append(fiber.fwhm_of(t, np.abs(A_disp) ** 2))
    fwhm_sol = np.array(fwhm_sol)
    fwhm_disp = np.array(fwhm_disp)

    print(f"{'z/z0':>6} {'FWHM soliton [ps]':>18} {'FWHM 分散のみ [ps]':>20}")
    for i, z in enumerate(z_pts):
        if abs((z / z0) - round(z / z0)) < 1e-6:
            print(f"{z/z0:6.0f} {fwhm_sol[i]:18.2f} {fwhm_disp[i]:20.2f}")

    sol_var = (fwhm_sol.max() - fwhm_sol.min()) / fwhm_sol.mean()
    print(f"\nソリトンの FWHM 変動 = {sol_var*100:.2f} % (ほぼ一定なら 0% に近い)")

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (左) ソリトン波形 (複数 z 重ね描き) → ほぼ重なる
    for k in range(6):
        z = k * z0
        A = fiber.propagate_ssfm(A0, z, dz=z0 / 200, dt=dt, beta2=BETA2, gamma=GAMMA)
        ax1.plot(t, np.abs(A) ** 2, lw=1.2, label=f"z = {k}·z0")
    ax1.set_xlim(-40, 40)
    ax1.set_xlabel("time [ps]"); ax1.set_ylabel("power |A|^2 [W]")
    ax1.set_title("N=1 soliton: waveform stays invariant")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) FWHM の z 依存性: ソリトン(一定) vs 分散のみ(広がる)
    ax2.plot(z_pts / z0, fwhm_sol, "C0o-", ms=4, label="soliton (CD + NL)")
    ax2.plot(z_pts / z0, fwhm_disp, "C3s--", ms=4, label="dispersion only")
    ax2.set_xlabel("distance z / z0"); ax2.set_ylabel("pulse FWHM [ps]")
    ax2.set_title("Soliton keeps width; dispersion alone broadens")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "soliton.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
