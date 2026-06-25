"""
問題1: PRBS (Pseudo Random Bit Sequence) の生成と強度スペクトル

【この問題で学ぶこと】
  通信路の評価 (BER 測定など) で「試験信号」として広く使われる PRBS を、
  自分で実装して作り、その周波数特性 (強度スペクトル) を確認する。
  PRBS は「決まった手順で生成するのに統計的にはランダムに見える」ビット列で、
  本物のランダム信号を毎回送るよりも再現性・取り回しの点で都合がよい。

【何をするか】
  15ビットのシフトレジスタ (LFSR) を用いて、周期 2**15 - 1 = 32767 ビットの
  PRBS を 1 周期分生成し、その強度(パワー)スペクトルを図示する。

【LFSR (線形帰還シフトレジスタ) のレジスタ動作】
  - 各クロックでレジスタを右に 1 ビットシフト
  - 最高次ビット b14 を出力
  - 最低次ビット b0 に b13 XOR b14 を入力
  これは生成多項式 x**15 + x**14 + 1 の Fibonacci 型 LFSR に相当し、
  この多項式は原始多項式なので最大長 (M系列, 周期 2**15 - 1) の系列が得られる。
  「最大長」= 全 0 を除く 2**15 - 1 通りの状態をちょうど一巡してから初期状態に戻る。
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 画面が無い環境でも画像保存できるようにする
import matplotlib.pyplot as plt


N_BITS = 15                # シフトレジスタの段数 (= 生成多項式の次数)
PERIOD = 2 ** N_BITS - 1  # M系列の周期 = 32767 (全 0 状態を除く全状態数)


def generate_prbs15(initial_state: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    """PRBS を 1 周期分生成する。

    LFSR (線形帰還シフトレジスタ) を PERIOD 回まわし、毎クロックの出力ビットを
    集めて 1 周期分の系列を作る。最終レジスタ状態も返すので、1 周期回すと
    初期状態へ戻る (= 周期性) ことの確認に使える。

    Args:
        initial_state: レジスタ b0..b14 の初期値 (0/1 の長さ15リスト)。
                       省略時は全ビット 1 で初期化する (0 以外なら何でもよい)。
                       全 0 にすると帰還しても 0 のままで系列が動かないため禁止。

    Returns:
        (出力ビット列 (長さ PERIOD の ndarray), シフト後の最終レジスタ状態)
    """
    if initial_state is None:
        reg = [1] * N_BITS  # b0..b14
    else:
        if len(initial_state) != N_BITS:
            raise ValueError("初期状態は 15 ビットで指定してください")
        if all(b == 0 for b in initial_state):
            raise ValueError("初期状態を全 0 にすると系列が生成されません")
        reg = list(initial_state)

    output = np.empty(PERIOD, dtype=np.int8)   # 出力ビット列の入れ物 (長さ 1 周期分)
    for i in range(PERIOD):
        output[i] = reg[14]              # 最高次ビット b14 を出力 (このクロックの送出ビット)
        feedback = reg[13] ^ reg[14]     # 帰還ビット = b13 XOR b14 (生成多項式のタップ)
        reg = [feedback] + reg[:14]      # 右に 1 ビットシフトし b0 に feedback を入れる
    return output, reg


def line_spectrum(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1 ビット = 1 サンプル(インパルス)として強度スペクトルを計算する。

    各ビットを瞬時値として FFT する。NRZ を双極性 (0->-1, 1->+1) に変換してから
    取ると直流成分が消える。最大長 PRBS では全スペクトル線がほぼ等高(白色的)になる。
    (パルス形状を考えない理想化されたモデル。次の nrz_spectrum と対比する。)

    Returns:
        (正規化周波数 [0, 0.5], パワー)
    """
    # 0/1 を ±1 の双極性信号へ。平均が 0 になるので直流 (周波数 0) 成分が消える
    signal = 2.0 * bits.astype(float) - 1.0
    spectrum = np.fft.rfft(signal)                    # 実信号用 FFT (正の周波数のみ返す)
    power = (np.abs(spectrum) ** 2) / len(signal)     # 強度 = |複素振幅|^2 を長さで正規化
    freq = np.fft.rfftfreq(len(signal))               # クロック周波数を 1 とした正規化周波数
    return freq, power


def nrz_spectrum(bits: np.ndarray, samples_per_bit: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """NRZ 矩形パルスとして送信したときの強度スペクトルを計算する。

    各ビットを samples_per_bit 個のサンプルで保持(矩形パルス)してから FFT する。
    矩形パルスのフーリエ変換により、包絡線が sinc^2 形になり、クロック周波数の
    整数倍でヌル(0)になる。光ファイバ上を伝送する波形のスペクトルに対応する。

    Returns:
        (クロック周波数を 1 とした周波数, パワー)
    """
    # 0/1 を ±1 の双極性信号へ (line_spectrum と同じ前処理)
    signal = 2.0 * bits.astype(float) - 1.0
    waveform = np.repeat(signal, samples_per_bit)     # 各ビットを保持して矩形パルスに整形
    spectrum = np.fft.rfft(waveform)                  # 矩形パルス波形を FFT
    power = (np.abs(spectrum) ** 2) / len(waveform)   # 強度スペクトル
    # サンプリング周波数 = samples_per_bit * クロック周波数。クロック単位に直す。
    freq = np.fft.rfftfreq(len(waveform)) * samples_per_bit
    return freq, power


def main() -> None:
    # PRBS を 1 周期分生成 (bits: 0/1 系列, final_reg: 1 周期後のレジスタ状態)
    bits, final_reg = generate_prbs15()

    # --- PRBS の基本性質を確認 ---
    # M系列の「平衡性」: 1 周期内で 1 の数が 0 の数より 1 だけ多くなる
    ones = int(bits.sum())                       # 系列中の 1 の個数
    zeros = PERIOD - ones                         # 系列中の 0 の個数
    print(f"周期           : {PERIOD} ビット")
    # 理論値 2**14 = 16384。1 の数がこれと一致すれば平衡性 OK
    print(f"1 の個数       : {ones}  (理論値 2**14 = {2 ** (N_BITS - 1)})")
    print(f"0 の個数       : {zeros}")
    print(f"先頭 32 ビット : {''.join(map(str, bits[:32]))}")

    # 1 周期回すとレジスタが初期値 (全 1) に戻ることを確認 (= 周期 PERIOD の裏付け)
    returns_to_init = final_reg == [1] * N_BITS
    print(f"1 周期後に初期状態へ復帰: {returns_to_init}")

    # --- 強度スペクトル (2 つのモデルで計算) ---
    f_line, p_line = line_spectrum(bits)        # インパルス: 白色的な平坦スペクトル
    f_nrz, p_nrz = nrz_spectrum(bits)           # NRZ 矩形パルス: sinc^2 包絡線

    # --- 図示 (3 段)  ※日本語フォント非依存にするためラベルは英語 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # (上) 時間波形の一部: 生成した PRBS の最初の n_show ビットを階段状に表示
    #      横軸 = クロック (時間)、縦軸 = ビット値 (0 か 1)
    n_show = 100
    ax1.step(range(n_show), bits[:n_show], where="post")
    ax1.set_title(f"PRBS15 bit sequence (first {n_show} bits)")
    ax1.set_xlabel("clock")
    ax1.set_ylabel("bit")
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # (中) インパルス強度スペクトル: 最大長 PRBS は全成分がほぼ等高 = 白色的
    #      横軸 = クロックを 1 とした正規化周波数、縦軸 = パワー
    ax2.plot(f_line, p_line, linewidth=0.8)
    ax2.set_title("Power spectrum (impulse model): flat / white-noise-like")
    ax2.set_xlabel("normalized frequency (clock frequency = 1)")
    ax2.set_ylabel("power")
    ax2.set_ylim(0, 2)
    ax2.grid(True, alpha=0.3)

    # (下) NRZ 強度スペクトル: sinc^2 包絡線 (クロック整数倍でヌル)
    #      横軸 = クロック周波数を 1 とした周波数、縦軸 = パワー
    #      実測スペクトルが矩形パルス由来の sinc^2 包絡線に乗ることを確認する
    ax3.plot(f_nrz, p_nrz, linewidth=0.4, label="PRBS15 (NRZ)")
    envelope = (np.sinc(f_nrz) ** 2) * np.max(p_nrz)   # 理論包絡線 sinc^2 (ピークに合わせてスケール)
    ax3.plot(f_nrz, envelope, "r--", linewidth=1.2, label="sinc^2 envelope")
    ax3.set_title("Power spectrum (NRZ rectangular pulse): sinc^2 envelope")
    ax3.set_xlabel("frequency / clock frequency")
    ax3.set_ylabel("power")
    ax3.set_xlim(0, 4)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    # この solution.py と同じフォルダに spectrum.png として図を保存する
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum.png")
    fig.savefig(out_path, dpi=120)
    print(f"スペクトル図を保存しました: {out_path}")


if __name__ == "__main__":
    main()
