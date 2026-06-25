"""問題4-2: 非線形効果 (自己位相変調 SPM) による位相変化

【この演習のねらい】
光ファイバの屈折率は光の強度でわずかに変わる (光カー効果)。強い光ほど屈折率が上がって
位相が余計に進むため、パルス自身の強度プロファイルに応じて位相が変化する。これを自己
位相変調 (SPM, Self-Phase Modulation) と呼ぶ。本問では分散・損失を無視して SPM だけを
考え、パルスピークの位相シフトが距離 z とともにどう増えるかを数値計算し解析解と比較する。
SPM は振幅 |A| は変えず位相だけを回すので、強度波形は変わらないがスペクトルは広がる。

標準単一モード光ファイバ (非線形係数 γ = 2.1 W^-1 km^-1) を、非線形効果のみ考えて伝搬させる。
強度ピーク 1 W、強度 FWHM 10 ps のガウスパルスを入力し、ピーク位置における位相変化量の
z 依存性を求める。さらに解析解と比較する。

解析解 (分散・損失を無視):
    |A| は変化せず、位相だけが φ(z,T) = γ|A(0,T)|² z 増える (自己位相変調)。
    ピーク (T=0, |A|²=P0) では  φ_max(z) = γ P0 z  (z に比例)。
非線形長 L_NL = 1/(γ P0) は「ピーク位相が 1 rad 進む距離」で、非線形効果の強さの目安。
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
GAMMA = 2.1            # 非線形係数 γ [1/(W·km)] (標準 SMF の代表値)
P_PEAK = 1.0           # ピーク強度 [W]
FWHM0 = 10.0           # 入力パルス幅 (強度FWHM) [ps]


def main() -> None:
    """ガウスパルスを SPM のみで伝搬させ、ピーク位相シフトの z 依存性を解析解と比較する。"""
    # 時間グリッド。N: サンプル数 (FFT 用に 2 のべき乗)、Tspan: 時間窓 [ps]
    N = 2 ** 14
    Tspan = 400.0
    t = (np.arange(N) - N / 2) * (Tspan / N)   # 0 中心の時間軸 [ps]
    dt = Tspan / N                             # サンプリング間隔 [ps]

    A0 = fiber.gaussian_pulse(t, P_PEAK, FWHM0)  # 入力ガウスパルス A(z=0, t)
    i_peak = np.argmax(np.abs(A0))             # 振幅ピーク (= T=0 付近) のインデックス
    L_NL = 1.0 / (GAMMA * P_PEAK)              # 非線形長 L_NL = 1/(γ P0) [km] を計算
    print(f"γ = {GAMMA} 1/(W·km), ピーク強度 = {P_PEAK} W")
    # 標準出力: 非線形長。ピーク位相が 1 rad 進む距離の目安
    print(f"非線形長 L_NL = 1/(γ P0) = {L_NL:.4f} km\n")

    z_list = np.linspace(0, 5, 51)            # 0〜5 km を 51 点で評価
    phi_sim = []
    for z in z_list:
        A = fiber.nonlinear_step(A0, GAMMA, z)        # 非線形のみ (分散なし)。位相だけ回る
        # ピーク位置の位相 (基準 A0 に対する増分)。
        # A·conj(A0) の偏角を取ると、入力に対する位相シフトだけが残る。
        phi = np.angle(A[i_peak] * np.conj(A0[i_peak]))
        phi_sim.append(phi)
    # angle は ±π に折り返るので、unwrap で 2π の飛びを繋いで連続な位相にする
    phi_sim = np.unwrap(np.array(phi_sim))
    phi_theory = GAMMA * P_PEAK * z_list      # 解析解 φ_max(z) = γ·P0·z (z に比例)

    # 代表点 (z = 0,1,2,5 km) で数値と解析解を並べて表示
    print(f"{'z [km]':>8} {'φ_peak(sim) [rad]':>18} {'φ_peak(theory) [rad]':>22}")
    for z in (0, 1, 2, 5):
        idx = int(np.argmin(np.abs(z_list - z)))     # 連続unwrap済み配列から取り出す
        print(f"{z:8.1f} {phi_sim[idx]:18.3f} {phi_theory[idx]:22.3f}")  # z / 数値φ / 解析φ

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # (左) ピーク位相シフトの z 依存性: 解析解 (直線) と数値 (点) が一致するはず
    ax1.plot(z_list, phi_theory, "r-", lw=2, label="analytic  γ·P0·z")
    ax1.plot(z_list, phi_sim, "ko", ms=4, mfc="none", label="simulation")
    # 横軸: 距離 z [km]、縦軸: ピーク位相シフト [rad]
    ax1.set_xlabel("distance z [km]"); ax1.set_ylabel("peak phase shift [rad]")
    ax1.set_title("SPM peak phase shift vs distance")
    ax1.legend(); ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # (右) SPM によるスペクトル広がり: 位相が回るだけでもスペクトルは広がる
    def spectrum(A):
        # FFT で周波数領域に移し、強度スペクトル |·|² を最大値で正規化して返す
        S = np.abs(np.fft.fftshift(np.fft.fft(A))) ** 2
        return S / S.max()
    f = np.fft.fftshift(np.fft.fftfreq(N, d=dt))    # 周波数軸 [1/ps = THz]
    # z=0 のスペクトル (基準)。dB 表示。+1e-12 は log(0) を避けるための微小オフセット
    ax2.plot(f * 1e3, 10 * np.log10(spectrum(A0) + 1e-12), label="z = 0 km")
    for z in (2, 5):
        A = fiber.nonlinear_step(A0, GAMMA, z)       # SPM 後の波形
        # f*1e3 で THz→GHz に変換して横軸に。z が大きいほどスペクトルが広がる
        ax2.plot(f * 1e3, 10 * np.log10(spectrum(A) + 1e-12), label=f"z = {z} km")
    ax2.set_xlim(-200, 200); ax2.set_ylim(-40, 2)
    # 横軸: 周波数オフセット [GHz]、縦軸: 正規化スペクトル [dB]
    ax2.set_xlabel("frequency offset [GHz]"); ax2.set_ylabel("normalized spectrum [dB]")
    ax2.set_title("SPM-induced spectral broadening")
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")

    fig.tight_layout()
    out = os.path.join(HERE, "nonlinear.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")

    # 数値と解析解の一致度を絶対誤差の最大値で評価 (小さいほど計算が正しい)
    max_err = np.max(np.abs(phi_sim - phi_theory))
    print(f"数値 vs 解析解 最大絶対誤差 = {max_err:.2e} rad")


if __name__ == "__main__":
    main()
