"""問題4-1: 波長分散によるガウスパルスの広がり

標準単一モード光ファイバ (β2 = -22 ps^2/km) を、波長分散のみ考えて伝搬させる。
強度ピーク 1 W、強度 FWHM (パルス幅) 10 ps のガウスパルスを入力し、
パルス幅の伝搬距離 z 依存性を求める。さらに解析解と比較する。

解析解 (チャープ無しガウスパルス):
    T_FWHM(z) = T_FWHM(0) · sqrt( 1 + (z/L_D)^2 ),   L_D = T0^2 / |β2|
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
BETA2 = -22.0          # 二次分散 [ps^2/km]
P_PEAK = 1.0           # ピーク強度 [W]
FWHM0 = 10.0           # 入力パルス幅 (強度FWHM) [ps]


def main() -> None:
    # 時間グリッド (十分広く、細かく)
    N = 2 ** 14
    Tspan = 800.0       # [ps]
    t = (np.arange(N) - N / 2) * (Tspan / N)
    dt = Tspan / N

    t0 = fiber.t0_from_fwhm_gauss(FWHM0)
    LD = fiber.dispersion_length(t0, BETA2)
    print(f"β2 = {BETA2} ps^2/km, 入力 FWHM = {FWHM0} ps")
    print(f"T0 = {t0:.3f} ps, 分散長 L_D = T0^2/|β2| = {LD:.3f} km\n")

    A0 = fiber.gaussian_pulse(t, P_PEAK, FWHM0)
    omega = fiber.omega_grid(N, dt)

    z_list = np.linspace(0, 10, 41)     # 0〜10 km
    fwhm_sim = []
    for z in z_list:
        A = fiber.dispersion_step(A0, BETA2, z, omega)
        fwhm_sim.append(fiber.fwhm_of(t, np.abs(A) ** 2))
    fwhm_sim = np.array(fwhm_sim)
    fwhm_theory = FWHM0 * np.sqrt(1 + (z_list / LD) ** 2)

    # 代表点で数値を表示
    print(f"{'z [km]':>8} {'FWHM(sim) [ps]':>16} {'FWHM(theory) [ps]':>18}")
    for z, fs in zip(z_list, fwhm_sim):
        if abs(z - round(z)) < 1e-9 and int(round(z)) in (0, 1, 2, 5, 10):
            ft = FWHM0 * np.sqrt(1 + (z / LD) ** 2)
            print(f"{z:8.1f} {fs:16.2f} {ft:18.2f}")

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (左) いくつかの z での強度波形
    for z in [0, 1, 2, 5, 10]:
        A = fiber.dispersion_step(A0, BETA2, z, omega)
        ax1.plot(t, np.abs(A) ** 2, label=f"z = {z} km")
    ax1.set_xlim(-120, 120)
    ax1.set_xlabel("time [ps]"); ax1.set_ylabel("power |A|^2 [W]")
    ax1.set_title("Pulse broadening by chromatic dispersion")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) FWHM の z 依存性: 数値 vs 解析解
    ax2.plot(z_list, fwhm_theory, "r-", lw=2, label="analytic")
    ax2.plot(z_list, fwhm_sim, "ko", ms=4, mfc="none", label="simulation")
    ax2.set_xlabel("distance z [km]"); ax2.set_ylabel("pulse FWHM [ps]")
    ax2.set_title("Pulse width vs distance")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "dispersion.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    max_err = np.max(np.abs(fwhm_sim - fwhm_theory) / fwhm_theory)
    print(f"数値 vs 解析解 最大相対誤差 = {max_err*100:.2f} %")


if __name__ == "__main__":
    main()
