"""問題2-6: スペクトル整形 QAM サンプル列の時間遅延

50 Gbaud (シンボル周期 T = 20 ps)、α=1 raised cosine 整形の QAM サンプル列を、
τ = 5, 10, 15, 20 ps だけ遅延させたサンプル列を生成し、時間波形を図示する
(最初の10シンボル時間ぶん)。

τ=5,10,15,20 ps は T=20 ps の 1/4, 1/2, 3/4, 1 倍に相当する。
10 ps は 2 sample/symbol のちょうど 1 サンプル、20 ps は 1 シンボル(2サンプル)ぶんの遅延。
5,15 ps は「半サンプル」の端数遅延なので、帯域制限信号の補間(連続波形を遅らせた時刻で再標本化)で実現する。
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
RS_GBD = 50.0
T_PS = 1000.0 / RS_GBD          # シンボル周期 [ps] = 20 ps
BETA = 1.0
SPS = 2
TS_PS = T_PS / SPS             # サンプル周期 [ps] = 10 ps
DELAYS_PS = [0, 5, 10, 15, 20]
N_SHOW = 10                    # 表示シンボル数
M = 16                         # 代表として16QAM


def rc_pulse(t: np.ndarray, beta: float) -> np.ndarray:
    """連続 raised cosine パルス (t はシンボル時間単位)。"""
    sinc = np.sinc(t)
    denom = 1.0 - (2 * beta * t) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.cos(np.pi * beta * t) / denom
    if beta > 0:
        cos[np.isclose(np.abs(2 * beta * t), 1.0)] = np.pi / 4
    return sinc * cos


def analog_waveform(sym: np.ndarray, t_ps: np.ndarray) -> np.ndarray:
    """シンボル列の連続波形 x(t) = Σ s[n] h(t/T - n) を時刻 t_ps [ps] で評価。"""
    x = np.zeros_like(t_ps, dtype=complex)
    for n in range(len(sym)):
        x += sym[n] * rc_pulse(t_ps / T_PS - n, BETA)
    return x


def main() -> None:
    k = int(np.log2(M))
    prbs = comm.generate_prbs(15)
    sym = comm.bits_to_symbols(np.tile(prbs, k), M)[:N_SHOW + 6]  # 余裕を持って

    print(f"Rs={RS_GBD} Gbaud -> T={T_PS:.0f} ps, {SPS} sps -> Ts={TS_PS:.0f} ps")
    print("遅延 τ とサンプル換算:")
    for tau in DELAYS_PS:
        print(f"  τ={tau:2d} ps = {tau / T_PS:.2f} シンボル = {tau / TS_PS:.1f} サンプル")

    # 表示用の連続波形時刻 (0〜10シンボル)
    t_fine = np.linspace(0, N_SHOW * T_PS, N_SHOW * 64 + 1)
    # 2-sps サンプル時刻
    m_idx = np.arange(0, N_SHOW * SPS + 1)
    t_samp = m_idx * TS_PS

    fig, axes = plt.subplots(len(DELAYS_PS), 1, figsize=(10, 11), sharex=True)
    for ax, tau in zip(axes, DELAYS_PS):
        # 遅延後の連続波形 x(t-τ) と、遅延後サンプル列 x(mTs - τ)
        x_cont = analog_waveform(sym, t_fine - tau)
        x_samp = analog_waveform(sym, t_samp - tau)        # ← 遅延サンプル列(成果物)
        # 参照: 遅延なし
        x0 = analog_waveform(sym, t_fine)
        ax.plot(t_fine, x0.real, color="0.7", lw=1.0, label="τ=0 (reference)")
        ax.plot(t_fine, x_cont.real, "C0-", lw=1.4, label=f"delayed τ={tau} ps")
        ax.plot(t_samp, x_samp.real, "C1.", ms=7, label="delayed samples (2 sps)")
        ax.set_ylabel(f"I  (τ={tau} ps)")
        ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(0, N_SHOW * T_PS)
        if tau == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1].set_xlabel("time [ps]")
    axes[0].set_title(f"{M}QAM RC-shaped waveform delayed by τ = 0,5,10,15,20 ps "
                      f"(T={T_PS:.0f} ps, I component)")
    fig.tight_layout()
    out = os.path.join(HERE, "delay.png")
    fig.savefig(out, dpi=120)
    print(f"\n遅延波形を保存しました: {out}")

    # τ=20 ps (=1シンボル) のとき、サンプル列が元のサンプル列を1シンボル(2サンプル)
    # シフトしたものに一致することを確認
    x0_samp = analog_waveform(sym, t_samp)
    x20_samp = analog_waveform(sym, t_samp - 20.0)
    shifted = np.concatenate([[np.nan, np.nan], x0_samp[:-2]])
    valid = ~np.isnan(shifted)
    err = np.nanmax(np.abs(x20_samp[valid] - shifted[valid]))
    print(f"τ=20 ps の遅延サンプル列 vs 元サンプルを2サンプルずらした列: 最大誤差={err:.2e}")


if __name__ == "__main__":
    main()
