"""問題4-1: 波長分散によるガウスパルスの広がり

【この演習のねらい】
光ファイバには「波長 (周波数) によって光の進む速さが少し違う」という波長分散 (群速度
分散 GVD) がある。パルスは多数の周波数成分の重ね合わせなので、伝搬とともに速い成分と
遅い成分が時間的にずれ、パルスが時間軸方向に広がっていく。本問では分散だけを考え、
パルス幅 (FWHM) が距離 z とともにどう広がるかを数値計算し、解析解と比較する。

標準単一モード光ファイバ (β2 = -22 ps^2/km) を、波長分散のみ考えて伝搬させる。
強度ピーク 1 W、強度 FWHM (パルス幅) 10 ps のガウスパルスを入力し、
パルス幅の伝搬距離 z 依存性を求める。さらに解析解と比較する。

解析解 (チャープ無しガウスパルス):
    T_FWHM(z) = T_FWHM(0) · sqrt( 1 + (z/L_D)^2 ),   L_D = T0^2 / |β2|
ここで T0 はパルスの特性幅、L_D は分散長 (広がりが目立ち始める目安距離)。
z ≪ L_D ではほとんど広がらず、z ≫ L_D では幅が z にほぼ比例して増える。
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
BETA2 = -22.0          # 二次分散 β2 [ps^2/km] (標準 SMF の代表値。負 = 異常分散)
P_PEAK = 1.0           # ピーク強度 [W]
FWHM0 = 10.0           # 入力パルス幅 (強度FWHM) [ps]


def main() -> None:
    """ガウスパルスを分散のみで伝搬させ、パルス幅の z 依存性を解析解と比較する。"""
    # 時間グリッド (十分広く、細かく)。
    # N: サンプル数 (2のべき乗にすると FFT が速い)。Tspan: 観測時間窓 [ps]。
    # パルスが広がっても窓からはみ出さないよう Tspan は十分広く取る。
    N = 2 ** 14
    Tspan = 800.0       # [ps]
    t = (np.arange(N) - N / 2) * (Tspan / N)   # 0 を中心に左右対称な時間軸 [ps]
    dt = Tspan / N                             # サンプリング間隔 [ps]

    t0 = fiber.t0_from_fwhm_gauss(FWHM0)       # 入力 FWHM → ガウスの特性幅 T0 [ps] に変換
    LD = fiber.dispersion_length(t0, BETA2)    # 分散長 L_D = T0^2/|β2| [km] を計算
    print(f"β2 = {BETA2} ps^2/km, 入力 FWHM = {FWHM0} ps")
    # 標準出力: 特性幅 T0 と分散長 L_D。L_D は「分散でパルスが目立って広がる目安距離」
    print(f"T0 = {t0:.3f} ps, 分散長 L_D = T0^2/|β2| = {LD:.3f} km\n")

    A0 = fiber.gaussian_pulse(t, P_PEAK, FWHM0)  # 入力ガウスパルスの複素振幅 A(z=0, t)
    omega = fiber.omega_grid(N, dt)            # 分散ステップで使う角周波数グリッド [rad/ps]

    z_list = np.linspace(0, 10, 41)     # 0〜10 km を 41 点で評価
    fwhm_sim = []
    for z in z_list:
        # 距離 z だけ分散を作用させる (周波数領域で ω² の位相回転 → IFFT)。非線形・損失は無し
        A = fiber.dispersion_step(A0, BETA2, z, omega)
        # 伝搬後の強度波形 |A|² の半値全幅 (FWHM) を測る
        fwhm_sim.append(fiber.fwhm_of(t, np.abs(A) ** 2))
    fwhm_sim = np.array(fwhm_sim)
    # 解析解: T_FWHM(z) = T_FWHM(0)·sqrt(1 + (z/L_D)²)
    fwhm_theory = FWHM0 * np.sqrt(1 + (z_list / LD) ** 2)

    # 代表点 (z = 0,1,2,5,10 km) で数値と解析解を並べて表示
    print(f"{'z [km]':>8} {'FWHM(sim) [ps]':>16} {'FWHM(theory) [ps]':>18}")
    for z, fs in zip(z_list, fwhm_sim):
        if abs(z - round(z)) < 1e-9 and int(round(z)) in (0, 1, 2, 5, 10):  # 整数 km のみ抜粋
            ft = FWHM0 * np.sqrt(1 + (z / LD) ** 2)   # その z での解析解
            print(f"{z:8.1f} {fs:16.2f} {ft:18.2f}")  # z / 数値FWHM / 解析FWHM

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (左) いくつかの z での強度波形 → z が大きいほど低く広がる
    for z in [0, 1, 2, 5, 10]:
        A = fiber.dispersion_step(A0, BETA2, z, omega)   # 各 z での分散後波形
        ax1.plot(t, np.abs(A) ** 2, label=f"z = {z} km")
    ax1.set_xlim(-120, 120)
    # 横軸: 時間 [ps]、縦軸: 瞬時パワー |A|² [W]
    ax1.set_xlabel("time [ps]"); ax1.set_ylabel("power |A|^2 [W]")
    ax1.set_title("Pulse broadening by chromatic dispersion")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) FWHM の z 依存性: 数値 (点) vs 解析解 (線) が重なれば正しい
    ax2.plot(z_list, fwhm_theory, "r-", lw=2, label="analytic")
    ax2.plot(z_list, fwhm_sim, "ko", ms=4, mfc="none", label="simulation")
    # 横軸: 伝搬距離 z [km]、縦軸: パルス幅 FWHM [ps]
    ax2.set_xlabel("distance z [km]"); ax2.set_ylabel("pulse FWHM [ps]")
    ax2.set_title("Pulse width vs distance")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "dispersion.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    # 数値と解析解の一致度を相対誤差の最大値で評価 (小さいほど計算が正しい)
    max_err = np.max(np.abs(fwhm_sim - fwhm_theory) / fwhm_theory)
    print(f"数値 vs 解析解 最大相対誤差 = {max_err*100:.2f} %")


if __name__ == "__main__":
    main()
