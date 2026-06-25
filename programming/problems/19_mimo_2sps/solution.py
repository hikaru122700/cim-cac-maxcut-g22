"""問題3-7: 2 sample/symbol での 2x2 MIMO 等化 (単一タップ vs 10タップ)

【この演習のねらい】
コヒーレント受信では、2偏波 (X/Y) が伝送路で混ざり合う「偏波回転」や、受信機の
サンプリングタイミングのずれ (タイミングオフセット) を、ディジタル信号処理 (DSP) の
適応等化器 (MIMO 等化器) で補償する。本問では「1シンボルあたり 2 サンプル
(2 sample/symbol, 2 sps)」で受信し、サンプリング位相が symbol 中心から半シンボル
(T/2 = 1サンプル) ずれている状況を考える。このとき等化器のタップ数が結果を大きく
左右することを示す。

問3-6 と同じ伝送路を、2 sample/symbol・パルス整形ロールオフ係数 α (BETA) = 0 で行う。
α=0 は理想的な帯域効率だが、波形が sinc 関数になり時間軸の裾が長く尾を引くため、
タイミングがずれると隣接シンボルが漏れ込む符号間干渉 (ISI) が起きやすい。

  - 単一タップ (L=1) MIMO: タップ1個では各偏波1サンプルしか見られず、タイミングずれを
    補えない。α=0 の長い sinc 裾による符号間干渉 (ISI) で BER (ビット誤り率) が大きく劣化。
  - 10タップ (L=10) 分数間隔 MIMO: シンボル周期より細かい (= 分数間隔, FSE) タップで
    複数サンプルを参照でき、整合フィルタ + タイミング補償 + 偏波分離をまとめて学習する。
    結果として BER が解析解 (理論限界) に近づく。

これにより「2 sps では多タップ MIMO が必要」であることを示す。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import unitary_group

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
import comm  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))
# 評価する変調方式: (多値数 M, SNR [dB], 表示名)。M=4=QPSK, M=16=16QAM。
# 16QAM は QPSK より雑音に弱いので、同程度の BER になるよう高い SNR を与えている。
CASES = [(4, 10.0, "QPSK"), (16, 15.0, "16QAM")]
RS = 32e9              # シンボルレート [baud] (= 32 Gシンボル/秒)
DF = 10e3             # レーザ線幅 [Hz]。位相雑音 (搬送波のふらつき) の強さを決める
N_SYM = 120_000       # 送信シンボル数 (等化器を十分収束させるため多めに取る)
SPS = 2               # 1シンボルあたりサンプル数 (2 sample/symbol)
BETA = 0.0              # α=0 (ロールオフ係数。sinc パルスになり裾が長く ISI が出やすい)
SPAN = 16             # パルス整形 FIR フィルタ長 [シンボル]
DELAY = 1000          # X/Y 偏波に与える相対遅延 [シンボル] (両偏波を別系列にする工夫)
T_OFFSET = 1           # サンプリング位相のずれ [サンプル] (=T/2。半シンボルずらす)
WARMUP = 20_000       # 等化器が収束するまでの過渡区間 (BER 集計から除外する)
SEED = 13             # 乱数シード (PRBS・偏波回転・雑音を再現可能にする)
# 比較する等化器設定: (タップ数 L, ステップサイズ μ)。
# L=1 は単一タップ (失敗例)、L=10 は分数間隔の多タップ (成功例)。
# μ は LMS の学習率で、タップが多いほど発散しにくいよう小さめにする。
CONFIGS = [(1, 1e-3), (10, 5e-4)]    # (タップ数, μ)


def make_symbols(M, n):
    """送信シンボル列 (長さ n) を作る。

    PRBS (擬似ランダムビット列) を生成し、M-QAM のシンボルに変調してから、
    必要な長さ n になるまで繰り返し並べる。
    """
    prbs = comm.generate_prbs(15)                 # M系列の擬似ランダムビット列 (1周期)
    k = int(np.log2(M))                            # 1シンボルあたりビット数
    # PRBS を k 倍に並べてから QAM シンボルへ変調 (bits_to_symbols)
    b = comm.bits_to_symbols(np.tile(prbs, k), M)
    # 1周期ぶんのシンボル列を必要数まで繰り返して長さ n に切り出す
    return np.tile(b, int(np.ceil(n / len(b))))[:n]


def channel(M, snr_db, rng):
    """送信〜受信までの伝送路をシミュレートする。

    送信シンボル → パルス整形 (2 sps) → 偏波回転 → AWGN → レーザ位相雑音 の順で
    劣化させ、受信サンプル列 rx を返す。等化器の入力 (rx) と参照 (送信シンボル s) を
    作るのが目的。
    """
    x = make_symbols(M, N_SYM)                     # X 偏波の送信シンボル列
    y = np.roll(x, DELAY)                          # Y 偏波は x を DELAY だけずらして別系列にする
    s = np.vstack([x, y])                          # 2偏波をまとめた送信シンボル (2, N_SYM)
    # 2 sps パルス整形 (両偏波): シンボル列を帯域制限波形へ。delay は FIR の群遅延 [サンプル]
    shx, delay = comm.pulse_shape(s[0], SPS, BETA, SPAN)
    shy, _ = comm.pulse_shape(s[1], SPS, BETA, SPAN)
    sh = np.vstack([shx, shy])                     # 整形後の両偏波波形 (2, Nsamp)
    # 偏波回転: 2x2 のユニタリ行列 U で X/Y 偏波を混ぜる (伝送路の偏波変動を模擬)
    U = unitary_group.rvs(2, random_state=SEED)   # ランダムな 2x2 ユニタリ行列
    U = U / np.sqrt(np.linalg.det(U))             # 行列式を 1 に正規化 (純粋な回転にする)
    r = U @ sh                                     # 偏波混合を適用
    # AWGN (白色ガウス雑音) を指定 SNR で付加。signal_power=1.0 は信号電力を 1 と明示
    r = comm.add_awgn(r, snr_db, signal_power=1.0, rng=rng)
    # レーザ位相雑音 (搬送波位相のランダムウォーク) を生成し、両偏波共通に掛ける
    theta = comm.laser_phase_noise(r.shape[1], DF, RS, rng)
    rx = r * np.exp(1j * theta)                    # 共通位相雑音 e^{jθ} を乗算
    return s, rx, delay


def ber_of(s, v, kmax, M):
    """等化出力 v と送信シンボル s を突き合わせて BER (ビット誤り率) を計算する。

    WARMUP〜kmax の収束済み区間だけを使い、両偏波 (p=0,1) のビット誤りを合算する。
    """
    sl = slice(WARMUP, kmax)                       # 収束後かつ有効計算範囲のシンボルだけ評価
    errs = tot = 0
    for p in range(2):                             # X 偏波 (p=0)・Y 偏波 (p=1) の両方
        tb = comm.symbols_to_bits(s[p, sl], M)     # 送信側 (正解) のビット列
        rb = comm.symbols_to_bits(v[p, sl], M)     # 等化出力を復調したビット列
        nn = min(len(tb), len(rb))                 # 長さを揃えて比較
        errs += comm.count_bit_errors(tb[:nn], rb[:nn]); tot += nn  # 誤り数と総ビット数を累積
    return errs / tot                              # BER = 誤りビット数 / 総ビット数


def main() -> None:
    """QPSK と 16QAM について、単一タップ vs 10タップ MIMO の等化結果を比較する。

    各変調方式について「等化前」「L=1 等化後」「L=10 等化後」のコンステレーション
    (IQ 平面の散布図) を描き、BER を標準出力に表示する。
    """
    rng = np.random.default_rng(SEED)              # 全処理で共有する乱数生成器
    # 実験条件を標準出力に表示 (何の条件で測ったか分かるように)
    print(f"2 sps, α={BETA}, サンプリング位相ずれ {T_OFFSET} サンプル(=T/2), "
          f"線幅 {DF/1e3:.0f} kHz\n")
    # 表のヘッダ: 方式 / SNR / タップ数 / 等化後の実測BER / 理論BER (解析解)
    print(f"{'方式':>6} {'SNR':>6} {'タップ':>6} {'等化後BER':>12} {'解析解':>12}")

    # 2行 (QPSK / 16QAM) × 3列 (等化前 / L=1 / L=10) のサブプロット
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, (M, snr, name) in enumerate(CASES):
        s, rx, delay = channel(M, snr, rng)        # 伝送路を通して受信サンプル rx を得る
        ber_th = float(comm.ber_theory_qam(M, snr))  # この M・SNR での理論 BER (比較の基準)

        # 等化前: ずれた位相で symbol レート抽出した点 (x偏波)。
        # delay (群遅延) + T_OFFSET (タイミングずれ) を起点に SPS 間隔で間引く。
        ax0 = axes[row, 0]
        idx = delay + T_OFFSET + np.arange(N_SYM) * SPS  # シンボル中心からずれたサンプル位置
        idx = idx[idx < rx.shape[1]]               # 信号配列の範囲内に収まるものだけ残す
        pre = rx[0, idx]                            # X 偏波を symbol レートで抜き出した受信点
        # コンステレーションを描画 (収束区間から 4000 点だけ。偏波混合+ISI で潰れているはず)
        ax0.scatter(pre[WARMUP:WARMUP + 4000].real, pre[WARMUP:WARMUP + 4000].imag,
                    s=4, alpha=0.15, color="C3")
        ax0.set_title(f"{name} before EQ (timing-offset sampling)")

        for col, (L, mu) in enumerate(CONFIGS, start=1):
            # 入力窓の開始位置: 群遅延を基準に、タップ中心が来るよう L//2 戻し、
            # さらにタイミングずれ T_OFFSET を加える。
            base = delay - L // 2 + T_OFFSET
            # 分数間隔 2x2 MIMO LMS 等化を実行。s を参照信号として LMS でタップを学習し、
            # シンボルレートの等化出力 v と有効最大シンボル番号 kmax を返す。
            v, kmax = comm.mimo_lms_fse(rx, s, L, mu, SPS, base)
            ber = ber_of(s, v, kmax, M)            # 等化出力の BER を計算
            # 1行: 方式 / SNR / タップ数 / 実測BER / 理論BER。L=1 は劣化, L=10 は理論に接近するはず
            print(f"{name:>6} {snr:5.0f}dB {L:6d} {ber:12.3e} {ber_th:12.3e}")
            ax = axes[row, col]
            # 等化後 X 偏波のコンステレーション。L=10 では各シンボル点がきれいに分離する
            ax.scatter(v[0, WARMUP:WARMUP + 4000].real, v[0, WARMUP:WARMUP + 4000].imag,
                       s=4, alpha=0.15, color="C0")
            ax.set_title(f"{name} after MIMO L={L} (BER={ber:.1e})")

        # 各サブプロット共通の体裁。横軸 I (同相成分)・縦軸 Q (直交成分) の IQ 平面
        for ax in axes[row]:
            ax.set_xlabel("I"); ax.set_ylabel("Q"); ax.set_aspect("equal")
            ax.grid(True, alpha=0.3); ax.tick_params(direction="in")
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)

    fig.suptitle("2 sps MIMO: single tap fails, 10 taps recovers (α=0, T/2 timing offset)",
                 fontsize=13)
    fig.tight_layout()
    out = os.path.join(HERE, "mimo_2sps.png")
    fig.savefig(out, dpi=120)
    print(f"\n図を保存しました: {out}")


if __name__ == "__main__":
    main()
