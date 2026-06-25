"""問題3-3: 単一タップ MIMO + LMS による適応等化

【目的】
問3-2 で作った「レーザ位相雑音が付加された受信 QAM サンプル」を、単一タップの
適応等化器 (1 個の複素係数 m) で補償し、崩れたコンステレーションを元に戻す。
係数 m は LMS (Least Mean Squares, 最小二乗) アルゴリズムで逐次更新する。

【単一タップ LMS の更新式 (問題指定)】
入力サンプル u (= 受信シンボル) に対し、

    v = m * u                       # 等化器出力 (m で補正したシンボル)
    m = m + mu*(d - v)*conj(u)      # LMS 更新

  - d: 参照符号 (本問では送信シンボル tx を既知とする「データ援用 / トレーニング」)
  - e = d - v: 誤差。出力 v が参照 d にどれだけ近いか。
  - mu (μ): ステップサイズ。大きいほど速く追従するが雑音に弱い (収束と追従のトレードオフ)。
  - conj(u): 勾配の向きを与える項。誤差を減らす方向へ m を少しだけ動かす。

【なぜ単一タップで効くのか】
位相雑音は「振幅一定で位相だけがゆっくり回る」歪みなので、補償も 1 個の複素係数
(回転 + ゲイン) で表せる。位相雑音 θ[n] に対し、m はその時々の逆回転 e^{-jθ[n]} に
追従していき、結果として受信点が元の QAM 格子に戻る。
(より一般のマルチタップ MIMO は、波形が時間方向に広がる歪み = ISI も補償できる。)
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
LABELS = {4: "QPSK", 16: "16QAM", 64: "64QAM"}       # 図タイトル用の表示名
RS = 32e9                # 符号速度 [baud] (1 sps なので fs と等しい)
DF = 10e3                # レーザのスペクトル線幅 [Hz]
N_SYM = 200_000          # シンボル数 (等化の収束・追従が見える程度)
MU = 0.01                # LMS ステップサイズ μ (収束速度と安定性のトレードオフ)
WARMUP = 5000            # 収束前の過渡区間。BER 評価や図から除外する
SEED = 9                 # 乱数シード (位相雑音を再現するため固定)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS を元に M-QAM の送信シンボル列を n_sym 個だけ作る (問3-2 と同じ)。"""
    prbs = comm.generate_prbs(15)        # M系列 PRBS を 1 周期生成 (擬似ランダムな試験ビット列)
    k = int(np.log2(M))                  # 1 シンボルあたりのビット数
    # comm.bits_to_symbols: ビット列 -> 平均電力1 に規格化した複素 QAM シンボル列
    block = comm.bits_to_symbols(np.tile(prbs, k), M)
    reps = int(np.ceil(n_sym / len(block)))   # 必要数に届くまで繰り返す回数
    return np.tile(block, reps)[:n_sym]       # 繰り返して n_sym 個に切り揃える


def lms_single_tap(rx: np.ndarray, ref: np.ndarray, mu: float):
    """単一タップ LMS 等化。返り値 (等化出力 v, タップ履歴 m)。

    Args:
        rx:  受信シンボル列 (位相雑音付き)。各サンプルが入力 u になる。
        ref: 参照シンボル列 (データ援用なので送信シンボル tx を渡す)。各サンプルが d。
        mu:  ステップサイズ μ。

    Returns:
        (v_out, m_hist): 等化後の出力列 v[n] と、各時刻のタップ係数 m[n] の履歴。
    """
    n = len(rx)
    v_out = np.empty(n, dtype=complex)   # 等化器出力を貯める配列
    m_hist = np.empty(n, dtype=complex)  # タップ係数 m の時間変化を記録する配列
    m = 1.0 + 0.0j                       # タップ初期値 (まだ何も学習していないので恒等 = 1)
    for i in range(n):
        u = rx[i]                        # 入力サンプル (位相雑音で回された受信シンボル)
        v = m * u                        # 等化器出力: 現在のタップ m で補正
        v_out[i] = v
        d = ref[i]                       # 参照符号 (既知の送信シンボル = 教師信号)
        # LMS 更新: 誤差 (d - v) を減らす方向へ m を mu 分だけ動かす。
        # conj(u) は勾配の向き。これにより m は逆回転 e^{-jθ} へ追従していく。
        m = m + mu * (d - v) * np.conj(u)
        m_hist[i] = m                    # 更新後のタップを記録
    return v_out, m_hist


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"符号速度 {RS/1e9:.0f} Gbaud, 線幅 {DF/1e3:.0f} kHz, μ={MU}, "
          f"シンボル数 {N_SYM}\n")

    # 上段=等化前, 下段=等化後 を、3 つの変調方式について列ごとに並べる
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for col, M in enumerate(QAM_ORDERS):
        tx = make_symbols(M, N_SYM)                        # 送信 M-QAM シンボル
        theta = comm.laser_phase_noise(N_SYM, DF, RS, rng) # レーザ位相雑音 θ[n]
        rx = tx * np.exp(1j * theta)                       # 位相雑音付き受信信号 (等化器への入力)

        v, m_hist = lms_single_tap(rx, tx, MU)             # 単一タップ LMS 等化を実行

        # --- 等化後の BER (ビット誤り率) を、収束後の区間 (WARMUP 以降) だけで評価 ---
        rx_bits = comm.symbols_to_bits(v[WARMUP:], M)      # 等化出力を硬判定 + 復調してビット列に
        tx_bits = comm.symbols_to_bits(tx[WARMUP:], M)     # 送信シンボルも同じ手順でビット列に
        nbit = min(len(rx_bits), len(tx_bits))             # 比較できる長さに揃える
        # comm.count_bit_errors: 不一致ビット数。それを総ビット数で割って BER に。
        ber = comm.count_bit_errors(tx_bits[:nbit], rx_bits[:nbit]) / nbit
        print(f"{LABELS[M]:6s}: 等化後 BER (収束後) = {ber:.2e}")   # うまく追従できていれば極小

        # (上段) 等化前のコンステレーション (位相雑音で円弧状に広がっている)
        axt = axes[0, col]
        axt.scatter(rx[WARMUP:WARMUP + 6000].real, rx[WARMUP:WARMUP + 6000].imag,
                    s=4, alpha=0.15, color="C3")          # 収束後の 6000 点だけ表示
        axt.set_title(f"{LABELS[M]} before EQ (phase noise)")
        # (下段) 等化後のコンステレーション (タップが逆回転に追従し格子に戻る)
        axb = axes[1, col]
        axb.scatter(v[WARMUP:WARMUP + 6000].real, v[WARMUP:WARMUP + 6000].imag,
                    s=4, alpha=0.15, color="C0")
        axb.set_title(f"{LABELS[M]} after single-tap LMS")
        for ax in (axt, axb):                              # 上下とも IQ 平面・等倍で軸を揃える
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7)

    fig.suptitle(f"Single-tap MIMO LMS equalization (μ={MU})", fontsize=14)
    fig.tight_layout()
    out = os.path.join(HERE, "lms_equalization.png")   # 等化前後の比較図を保存
    fig.savefig(out, dpi=120)
    print(f"\n等化前後のコンステレーションを保存しました: {out}")

    # --- タップ位相が位相雑音の逆回転を追っていることを確認 (16QAM) ---
    # タップ m[n] の偏角 ∠m[n] が、位相雑音 θ[n] の符号反転 −θ[n] に重なれば、
    # 「m は逆回転 e^{-jθ} を学習している」ことが目で確認できる。
    fig2, ax = plt.subplots(figsize=(9, 4))
    M = 16
    tx = make_symbols(M, N_SYM)
    theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)   # 新たに位相雑音を生成
    rx = tx * np.exp(1j * theta)                          # 位相雑音付き受信
    _, m_hist = lms_single_tap(rx, tx, MU)               # タップ履歴だけ取得
    n0 = 30000                                            # 先頭 30000 シンボルを表示
    # np.angle で複素タップの偏角を取り出し、np.unwrap で ±π の飛びをつないで滑らかにする
    ax.plot(np.arange(n0), np.unwrap(np.angle(m_hist[:n0])), "C0", lw=1,
            label="tap phase  ∠m[n]")                    # 実際に学習されたタップ位相
    ax.plot(np.arange(n0), -theta[:n0], "r--", lw=1, label="−θ[n] (ideal inverse)")  # 理想の逆回転
    ax.set_xlabel("symbol index"); ax.set_ylabel("phase [rad]")  # 横=シンボル番号, 縦=位相[rad]
    ax.set_title("Single tap tracks the inverse of laser phase noise (16QAM)")
    ax.legend(); ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
    fig2.tight_layout()
    out2 = os.path.join(HERE, "tap_tracking.png")        # タップ追従の図を保存
    fig2.savefig(out2, dpi=120)
    print(f"タップ追従の図を保存しました: {out2}")


if __name__ == "__main__":
    main()
