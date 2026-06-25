"""問題4-3: 基本ソリトン (N=1) の伝搬

【この演習のねらい】
問20では分散がパルスを広げ、問21では非線形 SPM が位相を回してチャープを生むことを見た。
異常分散 (β2 < 0) の下では、この2つの効果がちょうど打ち消し合うように働く条件があり、
パルスが形を崩さずに伝わる。これが光ソリトンである。本問では分散と非線形の両方を考え、
ソリトン次数 N=1 となる sech パルスを伝搬させ、時間波形が伝搬距離によらず一定に保たれる
ことを確認する。比較のため「分散のみ」の場合 (どんどん広がる) も並べて示す。

ソリトン次数:  N^2 = γ P0 T0^2 / |β2|   (分散と非線形のバランスを表す無次元量)
N=1 の条件:    P0 = |β2| / (γ T0^2)     (この強度に合わせると基本ソリトンになる)

分散 (パルスを広げる) と 非線形 SPM (チャープでスペクトルを広げ、異常分散下で圧縮する)
がちょうど釣り合い、形が崩れない。比較のため「分散のみ」の場合(広がる)も示す。
ソリトン周期 z0 = (π/2)·L_D は波形が元に戻る特性距離 (N=1 では常に一定だが基準として使う)。
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
BETA2 = -22.0          # 異常分散 β2 [ps^2/km] (β2<0 がソリトン成立に必須)
GAMMA = 2.1            # 非線形係数 γ [1/(W·km)]
FWHM0 = 10.0           # 入力 sech パルスの強度FWHM [ps]


def main() -> None:
    """N=1 ソリトンと「分散のみ」を伝搬させ、波形が保たれるか崩れるかを比較する。"""
    # 時間グリッド。N: サンプル数 (FFT 用 2 のべき乗)、Tspan: 時間窓 [ps]
    N = 2 ** 13
    Tspan = 400.0
    t = (np.arange(N) - N / 2) * (Tspan / N)   # 0 中心の時間軸 [ps]
    dt = Tspan / N                             # サンプリング間隔 [ps]

    t0 = fiber.t0_from_fwhm_sech(FWHM0)        # 入力 FWHM → sech の特性幅 T0 [ps] に変換
    P0 = abs(BETA2) / (GAMMA * t0 ** 2)             # N=1 条件のピーク強度 P0 = |β2|/(γ T0²)
    Nsol = np.sqrt(GAMMA * P0 * t0 ** 2 / abs(BETA2))  # ソリトン次数 N の検算 (≈1 になるはず)
    LD = fiber.dispersion_length(t0, BETA2)    # 分散長 L_D = T0²/|β2| [km]
    z0 = np.pi / 2 * LD                             # ソリトン周期 z0 = (π/2)·L_D [km]
    print(f"β2={BETA2} ps^2/km, γ={GAMMA} 1/(W·km), FWHM={FWHM0} ps")
    print(f"T0(sech) = {t0:.3f} ps")
    # 標準出力: N=1 にするためのピーク強度と、その検算 (N≒1) を表示
    print(f"N=1 のピーク強度 P0 = |β2|/(γ T0^2) = {P0:.4f} W")
    print(f"確認: ソリトン次数 N = {Nsol:.4f}")
    print(f"分散長 L_D = {LD:.3f} km, ソリトン周期 z0 = (π/2)L_D = {z0:.3f} km\n")

    A0 = fiber.sech_pulse(t, P0, FWHM0)        # 入力 sech パルス (N=1 になる強度 P0)
    z_total = 5 * z0                                # 5周期ぶん伝搬
    z_pts = np.linspace(0, z_total, 31)        # 評価する伝搬距離 (0〜5周期を 31 点)

    fwhm_sol, fwhm_disp = [], []
    for z in z_pts:
        # ソリトン (分散+非線形): Split-Step Fourier 法で両効果を交互適用しながら距離 z 伝搬。
        # dz=z0/200 は 1 ステップの距離 (ソリトン周期を 200 分割する細かさ)。
        A_sol = fiber.propagate_ssfm(A0, z, dz=z0 / 200, dt=dt,
                                     beta2=BETA2, gamma=GAMMA)
        fwhm_sol.append(fiber.fwhm_of(t, np.abs(A_sol) ** 2))   # ソリトンの幅 (一定のはず)
        # 比較: 分散のみ (非線形なし)。同じ入力を分散だけで伝搬させると広がる
        A_disp = fiber.dispersion_step(A0, BETA2, z, fiber.omega_grid(N, dt))
        fwhm_disp.append(fiber.fwhm_of(t, np.abs(A_disp) ** 2))  # 分散のみの幅 (増えるはず)
    fwhm_sol = np.array(fwhm_sol)
    fwhm_disp = np.array(fwhm_disp)

    # 代表点 (z/z0 が整数の周期) でソリトンと分散のみの幅を並べて表示
    print(f"{'z/z0':>6} {'FWHM soliton [ps]':>18} {'FWHM 分散のみ [ps]':>20}")
    for i, z in enumerate(z_pts):
        if abs((z / z0) - round(z / z0)) < 1e-6:   # z が z0 の整数倍のときだけ抜粋
            print(f"{z/z0:6.0f} {fwhm_sol[i]:18.2f} {fwhm_disp[i]:20.2f}")  # 周期数 / ソリトン幅 / 分散のみ幅

    # ソリトンの幅の変動率 (最大-最小)/平均。理想ソリトンならほぼ 0% になる
    sol_var = (fwhm_sol.max() - fwhm_sol.min()) / fwhm_sol.mean()
    print(f"\nソリトンの FWHM 変動 = {sol_var*100:.2f} % (ほぼ一定なら 0% に近い)")

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (左) ソリトン波形 (複数 z 重ね描き) → 形が保たれるので各曲線がほぼ重なる
    for k in range(6):
        z = k * z0                                  # 0,1,...,5 周期ぶんの距離
        A = fiber.propagate_ssfm(A0, z, dz=z0 / 200, dt=dt, beta2=BETA2, gamma=GAMMA)
        ax1.plot(t, np.abs(A) ** 2, lw=1.2, label=f"z = {k}·z0")
    ax1.set_xlim(-40, 40)
    # 横軸: 時間 [ps]、縦軸: 瞬時パワー |A|² [W]
    ax1.set_xlabel("time [ps]"); ax1.set_ylabel("power |A|^2 [W]")
    ax1.set_title("N=1 soliton: waveform stays invariant")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) FWHM の z 依存性: ソリトン (ほぼ水平 = 一定) vs 分散のみ (右肩上がり = 広がる)
    ax2.plot(z_pts / z0, fwhm_sol, "C0o-", ms=4, label="soliton (CD + NL)")
    ax2.plot(z_pts / z0, fwhm_disp, "C3s--", ms=4, label="dispersion only")
    # 横軸: 規格化距離 z/z0 (ソリトン周期単位)、縦軸: パルス幅 FWHM [ps]
    ax2.set_xlabel("distance z / z0"); ax2.set_ylabel("pulse FWHM [ps]")
    ax2.set_title("Soliton keeps width; dispersion alone broadens")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "soliton.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
