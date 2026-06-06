"""問題1-5: 受信シンボル列のビット系列への変換 (QAMデマッピング)

判定後のシンボルを、マッピングの逆操作でビットに戻す。
  振幅 -> レベル番号(2進) -> グレイ符号 -> ビット
方形QAMなので I 軸・Q 軸を独立に処理し、前半ビット=I軸・後半ビット=Q軸として連結する。

ここでは
  (a) 無雑音なら「マッピング -> デマッピング」が完全に元のビットに戻ること
  (b) 16QAM のグレイ符号ラベル付きコンステレーション
を確認する。
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
QAM_ORDERS = [4, 16, 64, 256]


def main() -> None:
    rng = np.random.default_rng(0)

    # --- (a) 無雑音での往復確認 (マッピングの逆がデマッピングであることの検証) ---
    print("=== 無雑音での往復確認 (マッピング -> デマッピング) ===")
    for M in QAM_ORDERS:
        k = int(np.log2(M))
        bits = rng.integers(0, 2, size=20000 * k).astype(np.int8)
        sym = comm.bits_to_symbols(bits, M)        # マッピング
        rx_bits = comm.symbols_to_bits(sym, M)     # デマッピング
        ok = np.array_equal(bits, rx_bits)
        print(f"{M:3d}QAM: 入力{len(bits):6d}bit -> {len(sym):5d}シンボル "
              f"-> {len(rx_bits):6d}bit  完全復元={ok}")

    # --- (b) 16QAM のグレイ符号ラベル付きコンステレーション図 ---
    M = 16
    k, L, kk = comm.qam_params(M)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # 全 M 個のシンボルについて、(ビット列 -> シンボル) を1つずつ描く
    for g in range(M):
        bits = np.array([(g >> (k - 1 - i)) & 1 for i in range(k)], dtype=np.int8)
        s = comm.bits_to_symbols(bits, M)[0]
        label = "".join(map(str, bits))
        ax.scatter(s.real, s.imag, s=60, color="C0")
        ax.annotate(label, (s.real, s.imag), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, family="monospace")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title("16QAM Gray-coded bit labels (I-bits | Q-bits)")
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    fig.tight_layout()
    out = os.path.join(HERE, "gray_labels.png")
    fig.savefig(out, dpi=120)
    print(f"\nグレイ符号ラベル図を保存しました: {out}")

    # --- 隣接シンボルが1ビットだけ異なることの確認 ---
    print("\n=== 16QAM: 横方向に隣接するシンボルのビット差 (グレイ性の確認) ===")
    # I軸 -3,-1,+1,+3 で Q を固定したときのラベル変化
    for q_bits in (np.array([0, 0]), ):
        prev = None
        for i_val in range(L):
            i_bits = comm._int_to_bits(comm._gray_encode(np.array([i_val])), kk)[0]
            full = np.concatenate([i_bits, q_bits]).astype(np.int8)
            label = "".join(map(str, full))
            if prev is not None:
                diff = int(np.count_nonzero(full != prev))
                print(f"  {''.join(map(str,prev))} -> {label} : 変化ビット数 {diff}")
            prev = full


if __name__ == "__main__":
    main()
