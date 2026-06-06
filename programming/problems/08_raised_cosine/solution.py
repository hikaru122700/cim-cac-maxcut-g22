"""問題2-2: レイズドコサイン (raised cosine) スペクトル整形

符号速度 (シンボルレート) Rs = 50 Gbaud、ロールオフ係数 α = 1 の raised cosine で
スペクトル整形した QAM サンプル列を生成する。

ポイント:
  - α=1 RC の占有帯域は (1+α)Rs = 100 GHz (複素ベースバンド両側) なので、
    サンプリング定理より fs >= 100 Gsample/s、すなわち 2 sample/symbol 以上が必要。
    (1 sample/symbol = 50 Gsample/s では足りない。)
  - 整形の入力に使う系列は「QAMシンボルを sps 倍にアップサンプル (シンボル間に 0 を挿入)
    した列」。これを RC フィルタに通す。
  - RC は Nyquist パルス: シンボル中心では h=1、他のシンボル位置では h=0 (ISIなし)。
    よってシンボル中心でサンプルすると元のシンボルがそのまま得られる。
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
RS_GBD = 50.0            # シンボルレート [Gbaud]
BETA = 1.0              # ロールオフ係数 α
SPS = 2                 # 1シンボルあたりサンプル数 (最小: fs=(1+α)Rs=100Gsample/s)
SPAN = 12               # RCフィルタ長 [シンボル]
QAM_ORDERS = [4, 16, 64, 256]


def rc_pulse(t: np.ndarray, beta: float) -> np.ndarray:
    """連続時間 raised cosine パルス h(t) (t はシンボル時間単位)。表示用の滑らかな曲線に使う。"""
    sinc = np.sinc(t)
    denom = 1.0 - (2 * beta * t) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.cos(np.pi * beta * t) / denom
    if beta > 0:
        cos[np.isclose(np.abs(2 * beta * t), 1.0)] = np.pi / 4
    return sinc * cos


def pulse_shape(sym: np.ndarray, sps: int, beta: float, span: int):
    """シンボル列を RC 整形する。返り値 (shaped, delay, h)。

    delay は群遅延 (サンプル数)。shaped[delay + n*sps] が n 番目のシンボル中心。
    """
    h = comm.rc_filter(beta, sps, span)             # ピーク1の対称FIR
    up = np.zeros(len(sym) * sps, dtype=complex)
    up[::sps] = sym                                 # アップサンプル (0挿入)
    shaped = np.convolve(up, h)
    delay = (len(h) - 1) // 2
    return shaped, delay, h


def main() -> None:
    fs = (1 + BETA) * RS_GBD
    print(f"シンボルレート Rs = {RS_GBD} Gbaud, ロールオフ α = {BETA}")
    print(f"必要サンプリング周波数 fs >= (1+α)Rs = {fs:.0f} Gsample/s "
          f"= {SPS} sample/symbol 以上")
    print(f"(1 sample/symbol = {RS_GBD} Gsample/s ではサンプリング定理を満たさない)\n")

    rng = np.random.default_rng(0)

    # ============ (1) 時間波形 (16QAM, 最初の10シンボル) ============
    M = 16
    k = int(np.log2(M))
    prbs = comm.generate_prbs(15)
    sym = comm.bits_to_symbols(np.tile(prbs, k), M)[:40]   # 先頭40シンボルで十分
    shaped, delay, h = pulse_shape(sym, SPS, BETA, SPAN)

    # シンボル中心でサンプル抽出 -> 元のシンボルに一致するか確認
    centers = delay + np.arange(len(sym)) * SPS
    sampled = shaped[centers]
    err = np.max(np.abs(sampled - sym))
    print(f"ISI確認 (16QAM): シンボル中心サンプルと元シンボルの最大誤差 = {err:.2e}")

    # 連続波形 (アナログ再構成) を高分解能で計算 (表示用)
    n_show = 10
    osf = 32
    t_fine = np.linspace(0, n_show, n_show * osf + 1)
    x_fine = np.zeros_like(t_fine, dtype=complex)
    for n in range(len(sym)):
        x_fine += sym[n] * rc_pulse(t_fine - n, BETA)

    fig, (axI, axQ) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, comp, name in ((axI, np.real, "In-phase (I)"), (axQ, np.imag, "Quadrature (Q)")):
        ax.plot(t_fine, comp(x_fine), "C0-", lw=1.2, label="shaped waveform (analog)")
        # 2 sample/symbol のサンプル点
        t_samp = (np.arange(len(shaped)) - delay) / SPS
        m = (t_samp >= 0) & (t_samp <= n_show)
        ax.plot(t_samp[m], comp(shaped[m]), "C1.", ms=8, label="samples (2 sps)")
        # シンボル中心
        ax.plot(np.arange(n_show + 1), comp(sym[:n_show + 1]), "rs", ms=7,
                mfc="none", label="symbol centers (ISI-free)")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
    axQ.set_xlabel("time [symbol]")
    axI.set_title(f"{M}QAM raised-cosine shaped waveform (α={BETA}, {SPS} sps)")
    axI.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out1 = os.path.join(HERE, "waveform.png")
    fig.savefig(out1, dpi=120)
    print(f"時間波形を保存しました: {out1}")

    # ============ (2) ISI-free サンプル抽出後のコンステレーション (4方式) ============
    fig2, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        k = int(np.log2(M))
        s = comm.bits_to_symbols(np.tile(prbs, k), M)
        sh, d, _ = pulse_shape(s, SPS, BETA, SPAN)
        c = d + np.arange(len(s)) * SPS
        samp = sh[c]
        ax.scatter(samp.real, samp.imag, s=8, alpha=0.3)
        ax.set_title(f"{M}QAM (ISI-free sampling)")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    fig2.suptitle("Constellations after ISI-free sampling of RC-shaped signal", fontsize=12)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "constellation_rc.png")
    fig2.savefig(out2, dpi=120)
    print(f"コンステレーションを保存しました: {out2}")


if __name__ == "__main__":
    main()
