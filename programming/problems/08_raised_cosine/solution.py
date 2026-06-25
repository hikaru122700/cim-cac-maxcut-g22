"""問題2-2: レイズドコサイン (raised cosine) スペクトル整形

この演習でやること:
  符号速度 (シンボルレート) Rs = 50 Gbaud、ロールオフ係数 α = 1 の raised cosine
  (レイズドコサイン) フィルタで「スペクトル整形」した QAM サンプル列を生成し、
  (1) 整形後の時間波形と、(2) シンボル中心で抜き出した点のコンステレーションを図示する。

なぜスペクトル整形するか:
  QAM シンボルをそのまま方形パルスで送ると周波数帯域が無限に広がってしまう。
  帯域を有限に抑えつつ、受信側でサンプルしたときに隣のシンボルが干渉しない
  (= 符号間干渉 ISI が出ない) パルス形が望ましい。RC はその両立を実現する
  代表的な「ナイキストパルス」。

各パラメータの意味:
  - ロールオフ係数 β (= α): スペクトルの裾の広がり (0〜1)。大きいほど帯域は広いが
    時間波形の裾の収束が速い。β=1 は最も帯域を使う設定で、占有帯域は (1+β)Rs。
  - sps (samples per symbol): 1シンボルを何個のサンプルで表すか (= オーバーサンプル率)。
  - span: フィルタの長さをシンボル数で表したもの (FIR の打ち切り範囲)。

サンプリング定理の確認 (ポイント):
  - α=1 RC の占有帯域は (1+α)Rs = 100 GHz (複素ベースバンド両側) なので、
    サンプリング定理より fs >= 100 Gsample/s、すなわち 2 sample/symbol 以上が必要。
    (1 sample/symbol = 50 Gsample/s では足りない。)
  - 整形の入力に使う系列は「QAMシンボルを sps 倍にアップサンプル (シンボル間に 0 を挿入)
    した列」。これを RC フィルタに通す。
  - RC は Nyquist パルス: シンボル中心では h=1、他のシンボル位置では h=0 (ISIなし)。
    よってシンボル中心でサンプルすると元のシンボルがそのまま得られる
    (本スクリプトでは「シンボル中心サンプルと元シンボルの最大誤差」で実際に確認する)。
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


HERE = os.path.dirname(os.path.abspath(__file__))   # このスクリプトのあるフォルダ (図の保存先)
RS_GBD = 50.0            # シンボルレート [Gbaud]
BETA = 1.0              # ロールオフ係数 α (β=1: 最も帯域を使う設定)
SPS = 2                 # 1シンボルあたりサンプル数 (最小: fs=(1+α)Rs=100Gsample/s)
SPAN = 12               # RCフィルタ長 [シンボル] (FIR の打ち切り範囲)
QAM_ORDERS = [4, 16, 64, 256]   # コンステレーションを描く QAM 多値数


def rc_pulse(t: np.ndarray, beta: float) -> np.ndarray:
    """連続時間 raised cosine パルス h(t) (t はシンボル時間単位)。表示用の滑らかな曲線に使う。

    comm.rc_filter は離散サンプル位置でのインパルス応答を返すのに対し、こちらは
    任意の連続時刻 t に対して h(t) を直接計算する (波形をアナログ的に再構成して
    滑らかに描くため)。式は comm.rc_filter と同じ RC の閉形式。

    Args:
        t: 時刻配列 (シンボル時間 T を単位とする無次元時間)。
        beta: ロールオフ係数 β。

    Returns:
        各 t での RC パルス値 (t と同 shape)。
    """
    sinc = np.sinc(t)                            # ナイキストの sinc 項 (中心1, 整数時刻で0)
    denom = 1.0 - (2 * beta * t) ** 2            # コサイン項の分母
    with np.errstate(divide="ignore", invalid="ignore"):
        # 分母が 0 になる点は一旦 inf/nan を許容 (後で極限値で埋める)
        cos = np.cos(np.pi * beta * t) / denom
    if beta > 0:
        # 特異点 t=±1/(2β) (分母=0) は 0/0 になるので極限値 π/4 に置換
        cos[np.isclose(np.abs(2 * beta * t), 1.0)] = np.pi / 4
    return sinc * cos                            # h(t) = sinc項 × コサイン項


def pulse_shape(sym: np.ndarray, sps: int, beta: float, span: int):
    """シンボル列を RC 整形する。返り値 (shaped, delay, h)。

    (comm.pulse_shape とほぼ同じだが、確認用にフィルタ係数 h も一緒に返す版。)
    処理は「アップサンプル (シンボル間に 0 を挿入) -> RC フィルタと畳み込み」。

    Args:
        sym: 入力シンボル列。
        sps: 1シンボルあたりサンプル数 (アップサンプル率)。
        beta: ロールオフ係数。
        span: フィルタ長 [シンボル]。

    Returns:
        (shaped, delay, h):
          shaped … RC 整形後のサンプル列
          delay  … 群遅延 (サンプル数)。shaped[delay + n*sps] が n 番目のシンボル中心
          h      … 使用した RC フィルタ係数 (確認・表示用)
    """
    h = comm.rc_filter(beta, sps, span)             # RCフィルタ係数を取得 (ピーク1の対称FIR)
    up = np.zeros(len(sym) * sps, dtype=complex)    # アップサンプル用のゼロ配列
    up[::sps] = sym                                 # sps 間隔にシンボルを配置 (間は 0)
    shaped = np.convolve(up, h)                     # フィルタと畳み込んで波形整形
    delay = (len(h) - 1) // 2                       # 対称FIRの群遅延 (中心タップまでの長さ)
    return shaped, delay, h


def main() -> None:
    # 占有帯域 = (1+β)Rs。これが必要サンプリング周波数 fs の下限になる
    fs = (1 + BETA) * RS_GBD
    print(f"シンボルレート Rs = {RS_GBD} Gbaud, ロールオフ α = {BETA}")
    print(f"必要サンプリング周波数 fs >= (1+α)Rs = {fs:.0f} Gsample/s "
          f"= {SPS} sample/symbol 以上")
    print(f"(1 sample/symbol = {RS_GBD} Gsample/s ではサンプリング定理を満たさない)\n")

    rng = np.random.default_rng(0)   # (この演習では未使用だが再現性のため固定)

    # ============ (1) 時間波形 (16QAM, 最初の10シンボル) ============
    M = 16
    k = int(np.log2(M))                                # 1シンボルあたりビット数
    prbs = comm.generate_prbs(15)                      # 試験用の擬似ランダムビット列 (PRBS15)
    # PRBS を k 回タイルしてビット数をシンボル境界にそろえ、QAM変調 -> 先頭40シンボルを採用
    sym = comm.bits_to_symbols(np.tile(prbs, k), M)[:40]   # 先頭40シンボルで十分
    shaped, delay, h = pulse_shape(sym, SPS, BETA, SPAN)   # RC整形 (波形・群遅延・係数を取得)

    # シンボル中心でサンプル抽出 -> 元のシンボルに一致するか確認 (ISIが無いことの検証)
    centers = delay + np.arange(len(sym)) * SPS        # 各シンボル中心のサンプル位置
    sampled = shaped[centers]                          # その位置のサンプル値
    err = np.max(np.abs(sampled - sym))                # 元シンボルとの最大ずれ (≈0 なら ISI なし)
    print(f"ISI確認 (16QAM): シンボル中心サンプルと元シンボルの最大誤差 = {err:.2e}")

    # 連続波形 (アナログ再構成) を高分解能で計算 (表示用)
    # 各シンボル sym[n] に連続RCパルス rc_pulse(t-n) を掛けて足し合わせる (Σ s[n]·h(t-n))
    n_show = 10                                        # 表示するシンボル数
    osf = 32                                           # 1シンボルあたりの描画分解能 (細かく刻む)
    t_fine = np.linspace(0, n_show, n_show * osf + 1)  # 滑らかな曲線用の連続時刻軸
    x_fine = np.zeros_like(t_fine, dtype=complex)
    for n in range(len(sym)):
        x_fine += sym[n] * rc_pulse(t_fine - n, BETA)  # n番目のシンボルのRCパルスを重ね合わせ

    # I (実部=同相) と Q (虚部=直交) を上下2段で描く (sharex で時間軸を共有)
    fig, (axI, axQ) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, comp, name in ((axI, np.real, "In-phase (I)"), (axQ, np.imag, "Quadrature (Q)")):
        # (a) 連続再構成した整形波形 (アナログ波形のイメージ)
        ax.plot(t_fine, comp(x_fine), "C0-", lw=1.2, label="shaped waveform (analog)")
        # (b) 実際にサンプルした 2 sample/symbol の離散点。時間軸はシンボル単位に換算
        t_samp = (np.arange(len(shaped)) - delay) / SPS   # サンプル位置を [シンボル] 単位へ
        m = (t_samp >= 0) & (t_samp <= n_show)             # 表示範囲内のサンプルだけ抽出
        ax.plot(t_samp[m], comp(shaped[m]), "C1.", ms=8, label="samples (2 sps)")
        # (c) シンボル中心 (ISI-free タイミング)。元シンボル値と一致するはず
        ax.plot(np.arange(n_show + 1), comp(sym[:n_show + 1]), "rs", ms=7,
                mfc="none", label="symbol centers (ISI-free)")
        ax.set_ylabel(name)                      # 縦軸: I 振幅 / Q 振幅
        ax.grid(True, alpha=0.3)
        ax.tick_params(direction="in")
    axQ.set_xlabel("time [symbol]")              # 横軸: シンボル単位の時間
    axI.set_title(f"{M}QAM raised-cosine shaped waveform (α={BETA}, {SPS} sps)")
    axI.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out1 = os.path.join(HERE, "waveform.png")
    fig.savefig(out1, dpi=120)
    print(f"時間波形を保存しました: {out1}")

    # ============ (2) ISI-free サンプル抽出後のコンステレーション (4方式) ============
    # RC整形 -> シンボル中心で抜き出すと、ISI が無ければ元の格子状コンステレーションに戻る。
    # それを 4つの QAM 方式について 2x2 の図で確認する。
    fig2, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, M in zip(axes.ravel(), QAM_ORDERS):
        k = int(np.log2(M))
        s = comm.bits_to_symbols(np.tile(prbs, k), M)  # その方式のシンボル列を生成
        sh, d, _ = pulse_shape(s, SPS, BETA, SPAN)     # RC整形 (波形と群遅延)
        c = d + np.arange(len(s)) * SPS                # 各シンボル中心のサンプル位置
        samp = sh[c]                                   # ISI-free タイミングで抽出した受信点
        # 抽出点を複素平面 (横=I, 縦=Q) に散布。理想格子点に集まるはず
        ax.scatter(samp.real, samp.imag, s=8, alpha=0.3)
        ax.set_title(f"{M}QAM (ISI-free sampling)")
        ax.set_xlabel("I"); ax.set_ylabel("Q")        # 横軸=同相成分, 縦軸=直交成分
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
        ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6)
    fig2.suptitle("Constellations after ISI-free sampling of RC-shaped signal", fontsize=12)
    fig2.tight_layout()
    out2 = os.path.join(HERE, "constellation_rc.png")
    fig2.savefig(out2, dpi=120)
    print(f"コンステレーションを保存しました: {out2}")


if __name__ == "__main__":
    main()
