"""問題3-1: レーザ位相雑音の生成と線幅の確認

スペクトル線幅 df = 100 kHz のレーザ位相雑音を生成する。位相雑音は AWGN の累積
(ランダムウォーク = ウィーナー過程) であり、その AWGN の分散は

    σ_PN^2 = 2π·df / fs

で与えられる (証明は explanation.md)。fs = 32 Gsample/s とする。

確認すること:
  - 生成した位相揺らぎの時間波形。
  - それを位相成分とする複素電界 e[n] = exp(jθ[n]) のスペクトル (フーリエ変換) が
    ローレンツ型になり、その半値全幅 (FWHM) の期待値が線幅 df = 100 kHz になること。
  - 位相の最初のサンプルは 0 ではなく -π〜π の一様乱数とし、AWGN を累積させる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
DF = 100e3          # 線幅 (FWHM) [Hz]
FS = 32e9           # サンプリングレート [Hz]
SEED = 7


def fit_lorentzian_fwhm(psd: np.ndarray, freq: np.ndarray, fmax: float = 300e3):
    """ローレンツ型 PSD に当てはめて半値全幅 (FWHM) を求める [Hz]。

    ローレンツ型 S(f) = A·γ/(γ^2+f^2) は 1/S が f^2 の1次式になる
    (1/S = γ/A + f^2/(Aγ))。これを最小二乗で直線フィットすれば、
    1点ごとの統計揺らぎに強いロバストな線幅推定ができる。FWHM = 2γ。
    """
    psd = psd / psd.max()
    m = np.abs(freq) < fmax
    f = freq[m]
    S = psd[m]
    x = f ** 2
    y = 1.0 / S
    w = S ** 2                                  # S が大きい中心部を重視
    b1, b0 = np.polyfit(x, y, 1, w=w)           # y = b1·x + b0
    gamma = np.sqrt(b0 / b1)                     # HWHM
    return 2.0 * gamma, gamma


def main() -> None:
    rng = np.random.default_rng(SEED)
    sigma2 = 2 * np.pi * DF / FS
    print(f"df = {DF/1e3:.0f} kHz, fs = {FS/1e9:.0f} Gsample/s")
    print(f"位相増分の分散 σ_PN^2 = 2π·df/fs = {sigma2:.3e} rad^2 "
          f"(σ_PN = {np.sqrt(sigma2):.3e} rad)\n")

    # ---- (1) 位相揺らぎの時間波形 ----
    n_wave = 2_000_000                       # 約 62.5 µs ぶん
    theta = comm.laser_phase_noise(n_wave, DF, FS, rng)
    t_us = np.arange(n_wave) / FS * 1e6      # [µs]

    # ---- (2) 複素電界スペクトル (周期グラム平均でローレンツ型を測る) ----
    L = 2 ** 21                              # セグメント長 (周波数分解能 fs/L ≈ 15.3 kHz)
    K = 48                                   # 平均セグメント数 (多いほど滑らか)
    freq = np.fft.fftshift(np.fft.fftfreq(L, d=1 / FS))
    psd = np.zeros(L)
    for _ in range(K):
        th = comm.laser_phase_noise(L, DF, FS, rng)
        e = np.exp(1j * th)
        psd += np.abs(np.fft.fftshift(np.fft.fft(e))) ** 2
    psd /= K

    fwhm, gamma = fit_lorentzian_fwhm(psd, freq)
    print(f"複素電界スペクトルの FWHM (ローレンツ当てはめ) = {fwhm/1e3:.1f} kHz  "
          f"(期待値 = {DF/1e3:.0f} kHz)")

    # 当てはめたローレンツ曲線: S(f) ∝ γ / (γ^2 + f^2)
    lor = gamma / (gamma ** 2 + freq ** 2)
    lor *= psd.max() / lor.max()

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(t_us, theta, lw=0.6)
    ax1.set_title(f"Laser phase noise (random walk), df={DF/1e3:.0f} kHz")
    ax1.set_xlabel("time [µs]")
    ax1.set_ylabel("phase θ(t) [rad]")
    ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # スペクトルは中心 ±1 MHz を拡大、縦は対数で
    psd_db = 10 * np.log10(psd / psd.max() + 1e-12)
    lor_db = 10 * np.log10(lor / lor.max() + 1e-12)
    ax2.plot(freq / 1e3, psd_db, lw=0.8, label="simulated PSD")
    ax2.plot(freq / 1e3, lor_db, "r--", lw=1.2,
             label=f"fitted Lorentzian FWHM={fwhm/1e3:.0f} kHz")
    ax2.axhline(10 * np.log10(0.5), color="gray", ls=":", lw=1, label="-3 dB (half power)")
    ax2.set_xlim(-500, 500)
    ax2.set_ylim(-30, 2)
    ax2.set_title(f"Optical field spectrum (FWHM measured = {fwhm/1e3:.0f} kHz)")
    ax2.set_xlabel("frequency [kHz]")
    ax2.set_ylabel("normalized PSD [dB]")
    ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(HERE, "phase_noise.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
