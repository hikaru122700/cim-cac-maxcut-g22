"""問題3-6: 偏波回転+AWGN+位相雑音を受けた両偏波 QAM の 2x2 MIMO 等化と BER

【全体の流れ】
送信 (両偏波 QAM) → 伝送路で劣化 → 受信側で 2x2 MIMO 等化 → BER 測定、という
コヒーレント光通信の受信処理を一通り再現する問題。

1 sample/symbol の両偏波 QPSK・16QAM を作り、伝送路で次の3つの劣化を順に与える:
  1. 偏波回転 (SU(2))           … x偏波・y偏波が混ざる
  2. 複素AWGN                   … 受信機の熱雑音 (SNR で強さを指定)
  3. レーザ位相雑音 (線幅10kHz) … コヒーレント受信時の搬送波位相のふらつき
SNR は QPSK:10 dB、16QAM:15 dB。

受信側ではこれを 2x2 MIMO + LMS (適応等化) で逆変換し、等化後の BER を測る。
MIMO 等化器は「混ざった2偏波を元に戻し (逆混合)、位相回転も追従して補正する」役割。
LMS は既知の送信シンボルを教師にして、フィルタ係数を1サンプルずつ最適化する手法。
タップ数 1 と 10 を比較し、解析解 (理論 BER) と突き合わせる。

【なぜ単一タップで足りるのか】
1 sample/symbol では伝送路が「瞬時的な偏波混合 + 共通位相回転」だけで、過去のシンボルが
今のシンボルに漏れ込む符号間干渉 (ISI) が無い。よって時間方向のフィルタ (多タップ) は不要で、
各時刻の 2×2 行列1つ (= 単一タップ MIMO) で偏波分離・位相追従ができ、BER は解析解に達する。
タップ数を増やしても (余分なタップが雑音を拾うぶん) わずかに劣化する程度にとどまる。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

# 共通ライブラリ comm.py を import できるよう、隣の _common フォルダを検索パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))   # このファイルのあるフォルダ (図の保存先)
CASES = [(4, 10.0, "QPSK"), (16, 15.0, "16QAM")]    # (多値数M, SNR[dB], 表示名) の検証ケース
RS = 32e9             # シンボルレート [baud]。1 sps なのでサンプリングレートにも相当
DF = 10e3            # レーザ線幅 [Hz] (位相雑音の強さを決める)
N_SYM = 200_000      # 偏波あたりのシンボル数
DELAY = 1000         # y偏波に与える遅延 [サンプル] (無相関化のため)
WARMUP = 20_000      # LMS 収束前の過渡区間。BER 集計から除外する助走サンプル数
TAPS = [1, 10]       # 比較する MIMO 等化器のタップ数
MU = {1: 2e-3, 10: 1e-3}  # タップ数ごとの LMS ステップサイズ (大きいほど速いが不安定)
SEED = 12            # 乱数シード (偏波回転・雑音・位相雑音の再現性確保)


def make_symbols(M: int, n_sym: int) -> np.ndarray:
    """PRBS から長さ n_sym の規格化 M-QAM シンボル列を作る (問3-4と同じ)。"""
    prbs = comm.generate_prbs(15)                   # M系列 (擬似ランダムビット列) を生成
    k = int(np.log2(M))                             # 1シンボルあたりビット数
    block = comm.bits_to_symbols(np.tile(prbs, k), M)  # ビット列を QAM シンボルへ変調
    reps = int(np.ceil(n_sym / len(block)))         # 必要数に届くまでの繰り返し回数
    return np.tile(block, reps)[:n_sym]             # 繰り返して先頭 n_sym だけ切り出す


def channel(M: int, snr_db: float, rng):
    """送信両偏波信号と、伝送路通過後の受信両偏波信号を返す。

    送信信号 s に対し「偏波回転 → AWGN → 位相雑音」を順に適用して受信信号 rx を作る。
    戻り値の s は等化の教師信号 (参照) として、rx は MIMO 等化器の入力として使う。
    """
    # --- 送信両偏波信号 (問3-4と同じ構成) ---
    x = make_symbols(M, N_SYM)                              # x偏波
    y = np.roll(x, DELAY)                                   # y偏波 = x の遅延版
    s = np.vstack([x, y])                                   # 送信 (2,N)

    # --- 劣化1: 偏波回転 (2偏波が混ざる) ---
    U = unitary_group.rvs(2, random_state=SEED)            # ランダム 2×2 ユニタリ行列
    U = U / np.sqrt(np.linalg.det(U))                       # det=1 に正規化 → SU(2) 偏波回転
    r = U @ s                                               # 偏波混合後の信号 (2,N)
    # --- 劣化2: 複素AWGN (受信機熱雑音)。signal_power=1 を基準に SNR[dB] 分の雑音を付加 ---
    r = comm.add_awgn(r, snr_db, signal_power=1.0, rng=rng)  # 複素AWGN
    # --- 劣化3: レーザ位相雑音。ランダムウォーク的な共通位相 θ[n] を生成 ---
    theta = comm.laser_phase_noise(N_SYM, DF, RS, rng)      # 共通レーザ位相雑音 [rad]
    # 両偏波に同じ位相回転 exp(jθ) を掛ける (同一レーザ由来なので両偏波で共通の位相)
    rx = r * np.exp(1j * theta)
    return s, rx


def ber_after_eq(s, v, valid, M):
    """等化後シンボル v と送信シンボル s を比較し、両偏波合計の BER を計算する。

    WARMUP (LMS 収束前) から等化の有効区間末尾までを使い、各偏波で
    シンボル→ビットに復調してから送信ビットと突き合わせ、誤りビット数を数える。
    """
    # 集計区間: WARMUP (助走) を飛ばし、等化器の有効範囲 valid の末尾まで
    sl = slice(WARMUP, valid.stop)
    errs = tot = 0
    for p in range(2):                                      # x偏波(0), y偏波(1) それぞれ
        tb = comm.symbols_to_bits(s[p, sl], M)             # 送信シンボル→送信ビット (教師)
        rb = comm.symbols_to_bits(v[p, sl], M)             # 等化後シンボル→受信ビット (硬判定込み)
        nn = min(len(tb), len(rb))                          # 比較長を短い方に合わせる
        # 不一致ビット数を加算し、比較した総ビット数も積算
        errs += comm.count_bit_errors(tb[:nn], rb[:nn]); tot += nn
    return errs / tot                                      # BER = 誤りビット数 / 総ビット数


def main() -> None:
    rng = np.random.default_rng(SEED)    # 雑音・位相雑音用の乱数生成器 (シード固定で再現可能)
    print(f"1 sps, 線幅 {DF/1e3:.0f} kHz, シンボル数 {N_SYM}/偏波\n")
    # 結果テーブルの見出し: 方式 / SNR / タップ数 / 実測BER / 理論BER(解析解)
    print(f"{'方式':>6} {'SNR':>5} {'タップ':>6} {'等化後BER':>12} {'解析解':>12}")

    # 図: 2行 (QPSK / 16QAM) × 3列 (等化前, タップ1後, タップ10後) のコンステレーション
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (M, snr, name) in enumerate(CASES):
        s, rx = channel(M, snr, rng)                       # 送信信号 s と受信信号 rx を生成
        # ber_theory_qam: グレイ符号方形QAMのBER解析解。実測との比較基準 (検算用)
        ber_th = float(comm.ber_theory_qam(M, snr))

        # --- 1列目: 等化前 (x偏波) コンステレーション ---
        # 偏波混合・雑音・位相回転で点が散らばり、QAM 格子が崩れている様子を示す
        ax0 = axes[row, 0]
        ax0.scatter(rx[0, WARMUP:WARMUP + 5000].real, rx[0, WARMUP:WARMUP + 5000].imag,
                    s=4, alpha=0.15, color="C3")
        ax0.set_title(f"{name} x-pol before EQ")

        # --- 2列目以降: タップ数ごとに MIMO 等化 → BER 測定 → 等化後コンステレーション ---
        for col, L in enumerate(TAPS, start=1):
            # mimo_lms: 2x2 バタフライFIR を LMS で適応学習し、偏波分離+位相補正した出力 v を返す
            #   v=等化後シンボル, valid=端の過渡を除いた有効サンプル範囲(slice)
            v, valid = comm.mimo_lms(rx, s, L, MU[L])
            ber = ber_after_eq(s, v, valid, M)             # 等化後の両偏波合計 BER
            # 実測 BER と理論 BER を1行に出力 (両者が近ければ等化が理想的に働いた証拠)
            print(f"{name:>6} {snr:4.0f}dB {L:6d} {ber:12.3e} {ber_th:12.3e}")
            ax = axes[row, col]
            # 等化後はきれいな QAM 格子に戻る (偏波分離・位相追従が成功)
            ax.scatter(v[0, WARMUP:WARMUP + 5000].real, v[0, WARMUP:WARMUP + 5000].imag,
                       s=4, alpha=0.15, color="C0")
            ax.set_title(f"{name} x-pol after MIMO (L={L}, BER={ber:.1e})")

        # この行の全サブプロットの軸設定 (横軸=I, 縦軸=Q, 等倍, 内向き目盛り)
        for ax in axes[row]:
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

    fig.suptitle("2x2 MIMO LMS equalization at 1 sample/symbol", fontsize=14)
    fig.tight_layout()
    out = os.path.join(HERE, "mimo_ber.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
