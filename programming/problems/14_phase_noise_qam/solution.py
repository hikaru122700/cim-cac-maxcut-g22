"""問題3-2: レーザ位相雑音を付加した QAM のコンステレーション

【目的】
問3-1 で作ったレーザ位相雑音を、実際の QAM 信号に掛けるとコンステレーション
(IQ 平面上のシンボル配置) がどう崩れるかを観察する。

【設定】
  - 1 sample/symbol (1 sps)、符号速度 32 Gbaud (= fs = 32 Gsample/s)。
    1 シンボル = 1 サンプルなので、シンボル間隔の時間は 1/Rs。
  - 変調方式は QPSK(4QAM)、16QAM、64QAM の 3 種、各シンボル数 10**6。
  - 線幅 df = 10 kHz のレーザ位相雑音を付加する。これはコヒーレント受信における
    送受信レーザの位相揺らぎ (絶対位相のドリフト) をモデル化したもの:
        rx[n] = tx[n] · exp( j·θ[n] )
    すなわち各シンボルを θ[n] だけ「回転」させる (振幅 |tx[n]| は変えない)。
  - 受信コンステレーションを図示する。

【期待される結果】
位相雑音 θ[n] はゆっくりしたランダムウォークなので、隣り合うシンボルの回転角は
ほぼ同じだが、時間が経つと少しずつ累積してずれていく。結果として各シンボル点は
原点を中心に「円弧状」に広がる (振幅は保たれ、位相だけが回るため)。
df が大きいほど円弧の広がりも大きくなる。
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


HERE = os.path.dirname(os.path.abspath(__file__))   # 図の保存先 (このファイルのフォルダ)
QAM_ORDERS = [4, 16, 64]                              # 比較する QAM 多値数
LABELS = {4: "QPSK (4QAM)", 16: "16QAM", 64: "64QAM"}  # 図タイトル用の表示名
RS = 32e9               # 符号速度 (シンボルレート) [baud]。1 sps なので fs と等しい
DF = 10e3              # レーザのスペクトル線幅 [Hz]
N_SYM = 10 ** 6        # 生成するシンボル数
SEED = 8               # 乱数シード (位相雑音を再現するため固定)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS を元に M-QAM の送信シンボル列を n_sym 個だけ作る。"""
    prbs = comm.generate_prbs(15)        # M系列 PRBS を 1 周期生成 (擬似ランダムな試験ビット列)
    k = int(np.log2(M))                  # 1 シンボルあたりのビット数 (例: 16QAM なら 4)
    # PRBS を k 回つないでから QAM シンボルへ変調する
    #   comm.bits_to_symbols: ビット列 -> 平均電力1 に規格化した複素 QAM シンボル列
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))   # 必要数に届くまで繰り返す回数
    return np.tile(block, reps)[:n_sym]       # 繰り返して n_sym 個に切り揃える


def main() -> None:
    rng = np.random.default_rng(SEED)            # 再現可能な乱数生成器
    # 1 シンボルあたりの位相増分の分散 σ_PN^2 = 2π·df/Rs (問3-1 と同じ式, fs→Rs)
    sigma2 = 2 * np.pi * DF / RS
    print(f"符号速度 Rs = fs = {RS/1e9:.0f} Gbaud, 線幅 df = {DF/1e3:.0f} kHz, "
          f"シンボル数 = {N_SYM}")
    print(f"位相増分の分散 σ_PN^2 = {sigma2:.2e} rad^2")
    # ランダムウォークは N 歩で分散が N 倍 → 標準偏差は √N 倍。系列全体での
    # 位相のばらつきの目安を表示する (どれだけ大きく回りうるかの感覚をつかむ)。
    print(f"系列全体での位相のばらつき ≈ √(N)·σ_PN = "
          f"{np.sqrt(N_SYM*sigma2):.2f} rad\n")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))   # 3 つの変調方式を横に並べる
    for ax, M in zip(axes, QAM_ORDERS):
        tx = make_symbols(M, N_SYM)                          # 送信 M-QAM シンボル列
        theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)   # レーザ位相雑音 θ[n] (ランダムウォーク)
        rx = tx * np.exp(1j * theta)                          # 各シンボルを θ[n] だけ位相回転 = 受信信号

        # 位相 θ がどこまで回ったか (最小〜最大) を表示。系列が長いほど範囲が広がる。
        print(f"{LABELS[M]:14s}: 位相 θ の範囲 = [{theta.min():.2f}, {theta.max():.2f}] rad")

        # 受信シンボルの先頭 6000 点だけを IQ 平面に散布図表示 (全点だと重すぎるため間引く)
        ax.scatter(rx[:6000].real, rx[:6000].imag, s=4, alpha=0.15)   # 横=I (実部), 縦=Q (虚部)
        ax.set_title(f"{LABELS[M]} + phase noise (df={DF/1e3:.0f} kHz)")
        ax.set_xlabel("I"); ax.set_ylabel("Q")
        ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")  # IQ は等倍表示
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"Constellations with laser phase noise (1 sps, N={N_SYM:,})", fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "constellation_pn.png")   # このファイルと同じフォルダに保存
    fig.savefig(out, dpi=120)
    print(f"\nコンステレーション図を保存しました: {out}")


if __name__ == "__main__":
    main()
