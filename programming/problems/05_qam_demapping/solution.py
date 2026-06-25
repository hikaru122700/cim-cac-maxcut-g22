"""問題1-5: 受信シンボル列のビット系列への変換 (QAMデマッピング)

【この問題で学ぶこと】
判定後のシンボルを、送信側マッピング (bits_to_symbols) の逆操作でビットに戻す
=デマッピング (復調) を理解する。これで「ビット -> シンボル -> ビット」の往復が
完成し、受信したシンボルから元の情報ビットを取り出せるようになる。

【デマッピングの流れ】
  振幅 -> レベル番号(2進) -> グレイ符号 -> ビット
方形QAMなので I 軸 (実部)・Q 軸 (虚部) を独立に処理し、前半ビット=I軸・
後半ビット=Q軸として連結する (送信側のビット配置と対応させる)。
この計算は共通ライブラリ comm.symbols_to_bits が行う (内部で硬判定も実施)。

【グレイ符号を使う理由】
隣り合うシンボルどうしが必ず 1 ビットだけ異なるように符号を割り当てておくと、
雑音で隣のシンボルへ誤判定しても誤るビットが 1 つで済む。これにより SER が同じでも
BER を小さく抑えられる。本スクリプトの (b)(c) でこの「グレイ性」を目で確認する。

【このスクリプトで確認すること】
  (a) 無雑音なら「マッピング -> デマッピング」が完全に元のビットに戻ること (可逆性)
  (b) 16QAM のグレイ符号ラベル付きコンステレーション図 (どの点がどのビット列か)
  (c) 横方向に隣接するシンボルどうしが 1 ビットしか違わないこと (グレイ性の数値確認)
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


HERE = os.path.dirname(os.path.abspath(__file__))  # このスクリプトがあるフォルダ (図の保存先)
QAM_ORDERS = [4, 16, 64, 256]   # 往復確認に使う QAM 多値数


def main() -> None:
    rng = np.random.default_rng(0)                 # 乱数生成器 (ランダムなビット列の生成用)

    # --- (a) 無雑音での往復確認 (マッピングの逆がデマッピングであることの検証) ---
    # 雑音を加えなければ、ビット -> シンボル -> ビット が完全に元に戻るはず (可逆性の確認)
    print("=== 無雑音での往復確認 (マッピング -> デマッピング) ===")
    for M in QAM_ORDERS:
        k = int(np.log2(M))                        # 1 シンボルあたりビット数
        # ランダムな 0/1 ビット列を生成 (長さは k の倍数になるよう 20000*k 個)
        bits = rng.integers(0, 2, size=20000 * k).astype(np.int8)
        sym = comm.bits_to_symbols(bits, M)        # マッピング: ビット -> 複素QAMシンボル
        rx_bits = comm.symbols_to_bits(sym, M)     # デマッピング: シンボル -> ビット (内部で硬判定)
        ok = np.array_equal(bits, rx_bits)         # 入力ビットと完全一致すれば往復成功
        print(f"{M:3d}QAM: 入力{len(bits):6d}bit -> {len(sym):5d}シンボル "
              f"-> {len(rx_bits):6d}bit  完全復元={ok}")

    # --- (b) 16QAM のグレイ符号ラベル付きコンステレーション図 ---
    # 各シンボル点の真上に「そのシンボルを表すビット列」を書き込み、配置とラベルの
    # 対応 (どの格子点がどの 4 ビットか) を可視化する。
    M = 16
    k, L, kk = comm.qam_params(M)                  # k=4 (ビット/シンボル), L=4 (軸レベル数), kk=2 (軸ビット数)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    # 全 M 個のシンボルについて、(ビット列 -> シンボル) を1つずつ描く
    for g in range(M):
        # 整数 g (0..15) を MSB first で k ビットに展開 (この問題で描く対象のビット列)
        bits = np.array([(g >> (k - 1 - i)) & 1 for i in range(k)], dtype=np.int8)
        s = comm.bits_to_symbols(bits, M)[0]       # そのビット列に対応する複素シンボル点
        label = "".join(map(str, bits))            # ラベル文字列 (例 "0110")
        ax.scatter(s.real, s.imag, s=60, color="C0")  # シンボル点をプロット
        # 点の少し上にビット列ラベルを注記
        ax.annotate(label, (s.real, s.imag), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, family="monospace")
    ax.axhline(0, color="gray", lw=0.5)            # I=0 軸線
    ax.axvline(0, color="gray", lw=0.5)            # Q=0 軸線
    ax.set_title("16QAM Gray-coded bit labels (I-bits | Q-bits)")
    ax.set_xlabel("In-phase (I)")                  # 横軸 = 同相成分 (実部, 前半 kk ビット)
    ax.set_ylabel("Quadrature (Q)")                # 縦軸 = 直交成分 (虚部, 後半 kk ビット)
    ax.set_aspect("equal")                         # I/Q 等スケール
    ax.grid(True, alpha=0.3)
    ax.tick_params(direction="in")
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    fig.tight_layout()
    out = os.path.join(HERE, "gray_labels.png")
    fig.savefig(out, dpi=120)                       # ラベル付きコンステレーション図を保存
    print(f"\nグレイ符号ラベル図を保存しました: {out}")

    # --- (c) 隣接シンボルが1ビットだけ異なることの確認 (グレイ性の数値検証) ---
    # I 軸方向に隣り合うシンボルのビット列を順に並べ、変化ビット数が常に 1 であることを示す。
    print("\n=== 16QAM: 横方向に隣接するシンボルのビット差 (グレイ性の確認) ===")
    # Q 軸ビットを "00" に固定し、I 軸レベルを 0,1,2,3 (= 振幅 -3,-1,+1,+3) と動かす
    for q_bits in (np.array([0, 0]), ):
        prev = None                                 # 直前のシンボルのビット列 (差分計算用)
        for i_val in range(L):
            # I 軸のレベル番号 i_val をグレイ符号化し、kk ビットのビット列に展開 (送信側と同じ並び)
            i_bits = comm._int_to_bits(comm._gray_encode(np.array([i_val])), kk)[0]
            full = np.concatenate([i_bits, q_bits]).astype(np.int8)  # I軸ビット + Q軸ビット
            label = "".join(map(str, full))
            if prev is not None:
                # 隣接する 2 シンボルで異なるビットの数を数える (グレイ符号なら必ず 1)
                diff = int(np.count_nonzero(full != prev))
                print(f"  {''.join(map(str,prev))} -> {label} : 変化ビット数 {diff}")
            prev = full


if __name__ == "__main__":
    main()
