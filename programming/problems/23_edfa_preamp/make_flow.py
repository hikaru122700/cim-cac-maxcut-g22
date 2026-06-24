"""問題4-4 (EDFA 前置増幅) の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("単位電力 QPSK シンボル列を作る", note="PRBS -> bits_to_symbols(M=4)"),
        Step("入力パワー Pin の候補を並べる", note="-50 .. -38 dBm"),
        Step("入力光を作る A_in = √Pin · s", note="|A_in|^2 = Pin [W]"),
        Step("EDFA で増幅し ASE を加える", note="fiber.edfa: √G 倍 + 複素AWGN"),
        Step("√(G·Pin) で正規化して QPSK 判定", note="symbols_to_bits"),
        Step("送受信ビットを比べ BER を測る", note="count_bit_errors / nb"),
        Step("Es/N0 から理論 BER を計算", note="Q(√(Es/N0))"),
        Branch("全ての Pin を試したか?", yes="ループ終了", no="次の Pin へ"),
        Step("BER vs Pin を片対数で描画", note="edfa_preamp_ber.png"),
        Step("BER=1e-3 の受信感度を補間で求める", note="np.interp"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="EDFA 前置増幅 QPSK の BER vs Pin の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
