"""問題1-4の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("QAM 多値数 M を 1 つ取り出す", note="M = 4, 16, 64, 256"),
        Step("送信シンボル列を作る", note="make_symbols(M, N)"),
        Step("AWGN を付加して受信列にする", note="comm.add_awgn, SNR=20 dB"),
        Step("各受信点を最近傍シンボルへ硬判定", note="comm.decide(rx, M)"),
        Step("送信と判定を比べて SER を計算", note="~np.isclose(dec, tx) の割合"),
        Step("受信点と参照シンボルを散布図に描く", note="赤 x = qam_constellation(M)"),
        Branch("4 つの M を全て処理したか?", yes="図を保存", no="次の M へ戻る"),
        Step("2x2 の判定図を保存", note="decision.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="QAM 受信サンプル判定の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
