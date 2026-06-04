"""
問題1: PRBS (Pseudo Random Bit Sequence) の生成と強度スペクトル

15ビットのシフトレジスタ (LFSR) を用いて、周期 2**15 - 1 = 32767 ビットの
PRBS を 1 周期分生成し、その強度(パワー)スペクトルを図示する。

レジスタ動作:
  - 各クロックでレジスタを右に 1 ビットシフト
  - 最高次ビット b14 を出力
  - 最低次ビット b0 に b13 XOR b14 を入力
これは生成多項式 x**15 + x**14 + 1 の Fibonacci 型 LFSR に相当し、
この多項式は原始多項式なので最大長 (周期 2**15 - 1) の系列が得られる。
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # 画面が無い環境でも画像保存できるようにする
import matplotlib.pyplot as plt


N_BITS = 15
PERIOD = 2 ** N_BITS - 1  # 32767


def generate_prbs15(initial_state: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    """PRBS を 1 周期分生成する。

    Args:
        initial_state: レジスタ b0..b14 の初期値 (0/1 の長さ15リスト)。
                       省略時は全ビット 1 で初期化する (0 以外なら何でもよい)。

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

    output = np.empty(PERIOD, dtype=np.int8)
    for i in range(PERIOD):
        output[i] = reg[14]              # 最高次ビット b14 を出力
        feedback = reg[13] ^ reg[14]     # b13 XOR b14
        reg = [feedback] + reg[:14]      # 右に 1 ビットシフトし b0 に feedback を入れる
    return output, reg


def line_spectrum(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """1 ビット = 1 サンプル(インパルス)として強度スペクトルを計算する。

    各ビットを瞬時値として FFT する。NRZ を双極性 (0->-1, 1->+1) に変換してから
    取ると直流成分が消える。最大長 PRBS では全スペクトル線がほぼ等高(白色的)になる。

    Returns:
        (正規化周波数 [0, 0.5], パワー)
    """
    signal = 2.0 * bits.astype(float) - 1.0
    spectrum = np.fft.rfft(signal)
    power = (np.abs(spectrum) ** 2) / len(signal)
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
    signal = 2.0 * bits.astype(float) - 1.0
    waveform = np.repeat(signal, samples_per_bit)     # 矩形パルスに整形
    spectrum = np.fft.rfft(waveform)
    power = (np.abs(spectrum) ** 2) / len(waveform)
    # サンプリング周波数 = samples_per_bit * クロック周波数。クロック単位に直す。
    freq = np.fft.rfftfreq(len(waveform)) * samples_per_bit
    return freq, power


def main() -> None:
    bits, final_reg = generate_prbs15()

    # --- PRBS の基本性質を確認 ---
    ones = int(bits.sum())
    zeros = PERIOD - ones
    print(f"周期           : {PERIOD} ビット")
    print(f"1 の個数       : {ones}  (理論値 2**14 = {2 ** (N_BITS - 1)})")
    print(f"0 の個数       : {zeros}")
    print(f"先頭 32 ビット : {''.join(map(str, bits[:32]))}")

    # 1 周期回すとレジスタが初期値 (全 1) に戻ることを確認
    returns_to_init = final_reg == [1] * N_BITS
    print(f"1 周期後に初期状態へ復帰: {returns_to_init}")

    # --- 強度スペクトル ---
    f_line, p_line = line_spectrum(bits)        # インパルス: 白色的な平坦スペクトル
    f_nrz, p_nrz = nrz_spectrum(bits)           # NRZ 矩形パルス: sinc^2 包絡線

    # --- 図示 (3 段)  ※日本語フォント非依存にするためラベルは英語 ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # (上) 時間波形の一部
    n_show = 100
    ax1.step(range(n_show), bits[:n_show], where="post")
    ax1.set_title(f"PRBS15 bit sequence (first {n_show} bits)")
    ax1.set_xlabel("clock")
    ax1.set_ylabel("bit")
    ax1.set_ylim(-0.2, 1.2)
    ax1.grid(True, alpha=0.3)

    # (中) インパルス強度スペクトル: 最大長 PRBS は全成分がほぼ等高 = 白色的
    ax2.plot(f_line, p_line, linewidth=0.8)
    ax2.set_title("Power spectrum (impulse model): flat / white-noise-like")
    ax2.set_xlabel("normalized frequency (clock frequency = 1)")
    ax2.set_ylabel("power")
    ax2.set_ylim(0, 2)
    ax2.grid(True, alpha=0.3)

    # (下) NRZ 強度スペクトル: sinc^2 包絡線 (クロック整数倍でヌル)
    ax3.plot(f_nrz, p_nrz, linewidth=0.4, label="PRBS15 (NRZ)")
    envelope = (np.sinc(f_nrz) ** 2) * np.max(p_nrz)   # 理論包絡線 sinc^2
    ax3.plot(f_nrz, envelope, "r--", linewidth=1.2, label="sinc^2 envelope")
    ax3.set_title("Power spectrum (NRZ rectangular pulse): sinc^2 envelope")
    ax3.set_xlabel("frequency / clock frequency")
    ax3.set_ylabel("power")
    ax3.set_xlim(0, 4)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spectrum.png")
    fig.savefig(out_path, dpi=120)
    print(f"スペクトル図を保存しました: {out_path}")


if __name__ == "__main__":
    main()
