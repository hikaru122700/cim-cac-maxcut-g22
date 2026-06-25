"""問題3-1: レーザ位相雑音の生成と線幅の確認

【背景】
現実のレーザ光は、出力する光の位相が時間とともにランダムに揺らいでいる。これを
「レーザ位相雑音」と呼ぶ。コヒーレント光通信では信号の位相に情報を載せるため、
この位相揺らぎが受信品質を直接劣化させる。位相揺らぎの「速さ」を表す指標が
スペクトル線幅 df [Hz] で、df が広いほどレーザの質が低く位相が速くばらつく。

【位相雑音のモデル: ランダムウォーク (ウィーナー過程)】
位相 θ[n] は「各サンプルで独立なガウス雑音 (AWGN) を少しずつ足していった累積」
としてモデル化される。その 1 ステップ分の AWGN の分散は

    σ_PN^2 = 2π·df / fs

で与えられる (導出は explanation.md)。fs はサンプリングレートで、本問では
fs = 32 Gsample/s とする。df が大きい/サンプル間隔 1/fs が長いほど 1 歩の
揺らぎが大きくなる、という直感に対応する。

【この演習で確認すること】
  1. 生成した位相揺らぎ θ(t) の時間波形 (ゆっくりとしたランダムウォーク)。
  2. その位相を持つ複素電界 e[n] = exp(j·θ[n]) のパワースペクトル (FFT の絶対値2乗)
     が「ローレンツ型」になり、その半値全幅 (FWHM) が線幅 df = 100 kHz に一致すること。
     → 線幅 df の定義そのものが「光電界スペクトルのローレンツ型 FWHM」である。
  3. 位相の最初のサンプルは 0 ではなく -π〜π の一様乱数とし、以後 AWGN を累積させる
     (絶対位相に意味はなく、初期位相はランダムでよいため)。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 共通ライブラリ comm.py を import するため、隣の _common ディレクトリを検索パスに追加する
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))   # 図の保存先となるこのファイルのあるフォルダ
DF = 100e3          # レーザのスペクトル線幅 (ローレンツ型 FWHM) [Hz]
FS = 32e9           # サンプリングレート [Hz] (= 32 Gsample/s)
SEED = 7            # 乱数シード (毎回同じ位相雑音を再現するため固定)


def fit_lorentzian_fwhm(psd: np.ndarray, freq: np.ndarray, fmax: float = 300e3):
    """ローレンツ型 PSD に当てはめて半値全幅 (FWHM) を求める [Hz]。

    ローレンツ型 S(f) = A·γ/(γ^2+f^2) は 1/S が f^2 の1次式になる
    (1/S = γ/A + f^2/(Aγ))。これを最小二乗で直線フィットすれば、
    1点ごとの統計揺らぎに強いロバストな線幅推定ができる。FWHM = 2γ。

    Args:
        psd:  測定されたパワースペクトル密度 (各周波数の値)。
        freq: psd と対応する周波数軸 [Hz]。
        fmax: フィットに使う中心付近の周波数範囲 [Hz] (裾の雑音を除外する)。

    Returns:
        (fwhm, gamma): 半値全幅 FWHM=2γ [Hz] と半値半幅 HWHM=γ [Hz]。
    """
    psd = psd / psd.max()                       # ピークを 1 に正規化 (フィットを安定させる)
    m = np.abs(freq) < fmax                     # 中心 ±fmax の範囲だけを使う (裾の雑音を捨てる)
    f = freq[m]
    S = psd[m]
    # ローレンツ型は 1/S が f^2 の1次式 → x=f^2, y=1/S とおくと直線になる
    x = f ** 2
    y = 1.0 / S
    w = S ** 2                                  # 重み: S が大きい中心部 (信頼度が高い) を重視
    b1, b0 = np.polyfit(x, y, 1, w=w)           # 重み付き最小二乗で y = b1·x + b0 を当てはめ
    gamma = np.sqrt(b0 / b1)                     # 切片/傾き の関係から HWHM=γ を逆算
    return 2.0 * gamma, gamma                    # FWHM = 2γ と γ を返す


def main() -> None:
    rng = np.random.default_rng(SEED)            # 再現可能な乱数生成器を1つ用意
    # 位相雑音の 1 ステップ分の分散 σ_PN^2 = 2π·df/fs を計算して表示する
    sigma2 = 2 * np.pi * DF / FS
    print(f"df = {DF/1e3:.0f} kHz, fs = {FS/1e9:.0f} Gsample/s")
    print(f"位相増分の分散 σ_PN^2 = 2π·df/fs = {sigma2:.3e} rad^2 "
          f"(σ_PN = {np.sqrt(sigma2):.3e} rad)\n")   # σ_PN = 1サンプルあたりの位相揺らぎ [rad]

    # ---- (1) 位相揺らぎの時間波形 ----
    n_wave = 2_000_000                       # サンプル数 (fs=32Gで 約 62.5 µs ぶん)
    # comm.laser_phase_noise: 線幅 df のランダムウォーク位相 θ[n] を生成する
    #   θ[0]=一様乱数(-π,π), θ[n]=θ[n-1]+N(0, 2π·df/fs) の累積和
    theta = comm.laser_phase_noise(n_wave, DF, FS, rng)
    t_us = np.arange(n_wave) / FS * 1e6      # サンプル番号を実時間 [µs] に変換 (横軸用)

    # ---- (2) 複素電界スペクトル (周期グラム平均でローレンツ型を測る) ----
    # 1 回の FFT だけだとスペクトルがギザギザなので、K 本のセグメントの
    # パワースペクトルを平均して滑らかにする (Bartlett/Welch 法の考え方)。
    L = 2 ** 21                              # 1 セグメントの長さ (周波数分解能 fs/L ≈ 15.3 kHz)
    K = 48                                   # 平均するセグメント数 (多いほど滑らかになる)
    freq = np.fft.fftshift(np.fft.fftfreq(L, d=1 / FS))   # FFT に対応する周波数軸 [Hz] (中心が0)
    psd = np.zeros(L)
    for _ in range(K):
        th = comm.laser_phase_noise(L, DF, FS, rng)       # セグメントごとに独立な位相雑音
        e = np.exp(1j * th)                               # 振幅一定・位相だけ揺らぐ複素電界
        psd += np.abs(np.fft.fftshift(np.fft.fft(e))) ** 2   # |FFT|^2 = パワースペクトルを加算
    psd /= K                                              # K 本で平均してパワースペクトルにする

    # 平均スペクトルにローレンツ型を当てはめて FWHM を推定 (= 線幅 df の実測値)
    fwhm, gamma = fit_lorentzian_fwhm(psd, freq)
    print(f"複素電界スペクトルの FWHM (ローレンツ当てはめ) = {fwhm/1e3:.1f} kHz  "
          f"(期待値 = {DF/1e3:.0f} kHz)")   # 実測 FWHM が設定線幅 df に一致するはず

    # 図に重ねる用に、当てはめたローレンツ曲線 S(f) ∝ γ / (γ^2 + f^2) を作る
    lor = gamma / (gamma ** 2 + freq ** 2)
    lor *= psd.max() / lor.max()             # 測定スペクトルとピーク高さを合わせる

    # ---- 図 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))   # 左: 時間波形, 右: スペクトル

    # 左図: 位相 θ(t) の時間波形。横軸=時間[µs], 縦軸=位相[rad]。
    # ゆっくり上下にドリフトするランダムウォークの様子が見える。
    ax1.plot(t_us, theta, lw=0.6)
    ax1.set_title(f"Laser phase noise (random walk), df={DF/1e3:.0f} kHz")
    ax1.set_xlabel("time [µs]")
    ax1.set_ylabel("phase θ(t) [rad]")
    ax1.grid(True, alpha=0.3); ax1.tick_params(direction="in")

    # 右図: 複素電界スペクトル。中心 ±1 MHz を拡大し、縦軸はピーク基準の dB 表示にする。
    psd_db = 10 * np.log10(psd / psd.max() + 1e-12)   # 測定スペクトルを dB に (1e-12 は log(0)回避)
    lor_db = 10 * np.log10(lor / lor.max() + 1e-12)   # 当てはめたローレンツ曲線を dB に
    ax2.plot(freq / 1e3, psd_db, lw=0.8, label="simulated PSD")
    ax2.plot(freq / 1e3, lor_db, "r--", lw=1.2,
             label=f"fitted Lorentzian FWHM={fwhm/1e3:.0f} kHz")
    # -3 dB (パワー半分) の水平線。スペクトルがこの線を切る幅が FWHM に対応する。
    ax2.axhline(10 * np.log10(0.5), color="gray", ls=":", lw=1, label="-3 dB (half power)")
    ax2.set_xlim(-500, 500)
    ax2.set_ylim(-30, 2)
    ax2.set_title(f"Optical field spectrum (FWHM measured = {fwhm/1e3:.0f} kHz)")
    ax2.set_xlabel("frequency [kHz]")
    ax2.set_ylabel("normalized PSD [dB]")
    ax2.grid(True, alpha=0.3); ax2.tick_params(direction="in")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(HERE, "phase_noise.png")   # このファイルと同じフォルダに保存
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
