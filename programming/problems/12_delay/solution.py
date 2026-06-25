"""問題2-6: スペクトル整形 QAM サンプル列の時間遅延

50 Gbaud (シンボル周期 T = 20 ps)、α=1 raised cosine 整形の QAM サンプル列を、
τ = 5, 10, 15, 20 ps だけ遅延させたサンプル列を生成し、時間波形を図示する
(最初の10シンボル時間ぶん)。

遅延の作り方(本問の中心テーマ = fractional delay / 端数遅延):
  整形後の波形は x(t) = Σ s[n]·h(t/T - n) という「連続関数」として書ける
  (h は raised cosine パルス)。τ だけ遅らせた信号は x(t-τ) なので、
  これをサンプル時刻 mTs で評価し直す = x(mTs - τ) とすれば、任意の遅延を
  実現できる。整数サンプルの遅延も端数(半サンプル)の遅延も同じ式で扱える。

τ とサンプル数の対応 (T=20 ps, Ts=10 ps = 2 sps):
  τ=5,10,15,20 ps は T=20 ps の 1/4, 1/2, 3/4, 1 倍。
  - 10 ps … ちょうど 1 サンプルぶんの遅延 (整数サンプル)
  - 20 ps … 1 シンボル = 2 サンプルぶんの遅延 (整数サンプル)
  - 5,15 ps … 「半サンプル」の端数遅延。離散サンプルのシフトでは作れないので、
              上記の連続波形を遅らせた時刻で再標本化する補間で実現する。

最後に τ=20 ps (=1シンボル) の遅延サンプル列が、元のサンプル列をちょうど
2サンプルずらした列と一致することを数値的に確認する(整数サンプル遅延の検算)。
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
RS_GBD = 50.0                  # シンボルレート [Gbaud] (1秒あたり 50G シンボル)
T_PS = 1000.0 / RS_GBD          # シンボル周期 [ps] = 1/Rs = 20 ps
BETA = 1.0                      # raised cosine ロールオフ係数 α
SPS = 2                         # 1シンボルあたりサンプル数
TS_PS = T_PS / SPS             # サンプル周期 [ps] = T/sps = 10 ps
DELAYS_PS = [0, 5, 10, 15, 20] # 試す遅延 τ [ps] (0 は参照, 5/15 は端数, 10/20 は整数サンプル)
N_SHOW = 10                    # 表示シンボル数 (波形を描く範囲)
M = 16                         # 代表として16QAM


def rc_pulse(t: np.ndarray, beta: float) -> np.ndarray:
    """連続 raised cosine パルス h(t) を返す (t はシンボル時間単位 t/T)。

    comm.rc_filter と同じ閉形式 h = sinc(t)·cos(πβt)/(1-(2βt)^2) だが、
    こちらは離散FIRではなく「任意の連続時刻 t」で評価するための関数。
    遅延 x(t-τ) を任意時刻で計算するために自前で用意している。
    """
    sinc = np.sinc(t)                           # ナイキストの sinc 項 (numpyは正規化sinc)
    denom = 1.0 - (2 * beta * t) ** 2            # コサイン項の分母 (t=±1/(2β) で 0 になる)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.cos(np.pi * beta * t) / denom   # 0/0 は一旦 nan を許容
    if beta > 0:
        # 特異点 t=±1/(2β) (= 2βt の絶対値が 1) は極限値 π/4 で埋める
        cos[np.isclose(np.abs(2 * beta * t), 1.0)] = np.pi / 4
    return sinc * cos


def analog_waveform(sym: np.ndarray, t_ps: np.ndarray) -> np.ndarray:
    """シンボル列の連続波形 x(t) = Σ s[n]·h(t/T - n) を時刻 t_ps [ps] で評価する。

    各シンボル s[n] を時刻 nT に置いた RC パルスの重ね合わせ。t_ps を好きな
    時刻配列で渡せるので、t_ps の代わりに (t_ps - τ) を渡せば遅延波形 x(t-τ) が
    得られる(これが本問の fractional delay の核心)。
    """
    x = np.zeros_like(t_ps, dtype=complex)
    for n in range(len(sym)):
        # n 番目のシンボルの寄与: s[n] × パルスを nT だけずらしたもの
        x += sym[n] * rc_pulse(t_ps / T_PS - n, BETA)
    return x


def main() -> None:
    k = int(np.log2(M))                          # シンボルあたりビット数
    prbs = comm.generate_prbs(15)                # PRBS (試験用の擬似ランダムビット列)
    # PRBS を k 回タイルして十分なビット数を作り QAM変調。表示分+余裕6シンボルだけ使う
    sym = comm.bits_to_symbols(np.tile(prbs, k), M)[:N_SHOW + 6]  # 余裕を持って

    print(f"Rs={RS_GBD} Gbaud -> T={T_PS:.0f} ps, {SPS} sps -> Ts={TS_PS:.0f} ps")
    print("遅延 τ とサンプル換算:")
    for tau in DELAYS_PS:
        # τ がシンボル周期T・サンプル周期Tsの何倍に当たるかを表示 (整数/端数の確認)
        print(f"  τ={tau:2d} ps = {tau / T_PS:.2f} シンボル = {tau / TS_PS:.1f} サンプル")

    # 連続波形を滑らかに描くための細かい時刻軸 (0〜10シンボル, 1シンボルあたり64点)
    t_fine = np.linspace(0, N_SHOW * T_PS, N_SHOW * 64 + 1)
    # 実際のサンプリング時刻 (2 sps なので Ts=10 ps 間隔)
    m_idx = np.arange(0, N_SHOW * SPS + 1)        # サンプル番号 m = 0,1,2,...
    t_samp = m_idx * TS_PS                         # サンプル時刻 mTs [ps]

    # 各遅延を 1 段ずつ縦に並べて描画 (横軸=時刻 を共有)
    fig, axes = plt.subplots(len(DELAYS_PS), 1, figsize=(10, 11), sharex=True)
    for ax, tau in zip(axes, DELAYS_PS):
        # 時刻を τ だけ前にずらして評価 = 波形を τ だけ右(遅れ方向)へ平行移動
        x_cont = analog_waveform(sym, t_fine - tau)        # 遅延後の連続波形 x(t-τ)
        x_samp = analog_waveform(sym, t_samp - tau)        # ← 遅延後サンプル列 x(mTs-τ)(成果物)
        # 参照: 遅延なしの連続波形 x(t)
        x0 = analog_waveform(sym, t_fine)
        ax.plot(t_fine, x0.real, color="0.7", lw=1.0, label="τ=0 (reference)")  # 灰: 遅延なし
        ax.plot(t_fine, x_cont.real, "C0-", lw=1.4, label=f"delayed τ={tau} ps")  # 青線: 遅延波形
        ax.plot(t_samp, x_samp.real, "C1.", ms=7, label="delayed samples (2 sps)")  # 橙点: 遅延サンプル
        ax.set_ylabel(f"I  (τ={tau} ps)")        # 縦軸は実部(I成分)。段ごとに遅延量を併記
        ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(0, N_SHOW * T_PS)
        if tau == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)  # 凡例は最上段(τ=0)にだけ表示
    axes[-1].set_xlabel("time [ps]")             # 共有の横軸ラベル(最下段)
    axes[0].set_title(f"{M}QAM RC-shaped waveform delayed by τ = 0,5,10,15,20 ps "
                      f"(T={T_PS:.0f} ps, I component)")
    fig.tight_layout()
    out = os.path.join(HERE, "delay.png")
    fig.savefig(out, dpi=120)
    print(f"\n遅延波形を保存しました: {out}")  # 出力PNGのパスを表示

    # --- 検算: τ=20 ps (=1シンボル=2サンプル) は整数サンプル遅延なので、
    #     遅延サンプル列は「元のサンプル列を 2 サンプルずらした列」と一致するはず ---
    x0_samp = analog_waveform(sym, t_samp)               # 遅延なしのサンプル列
    x20_samp = analog_waveform(sym, t_samp - 20.0)       # τ=20 ps 遅延のサンプル列
    # 元サンプル列を先頭に nan を 2 個詰めて 2 サンプル右シフト (先頭2点は比較対象外)
    shifted = np.concatenate([[np.nan, np.nan], x0_samp[:-2]])
    valid = ~np.isnan(shifted)                           # nan を除いた有効インデックス
    # 遅延サンプル列とシフト列の最大差。理論上 0 (端数遅延でないので補間誤差なし)
    err = np.nanmax(np.abs(x20_samp[valid] - shifted[valid]))
    print(f"τ=20 ps の遅延サンプル列 vs 元サンプルを2サンプルずらした列: 最大誤差={err:.2e}")


if __name__ == "__main__":
    main()
