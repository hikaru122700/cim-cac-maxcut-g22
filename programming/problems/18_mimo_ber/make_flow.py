"""問題3-6の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("両偏波 QAM シンボルを生成", note="make_symbols (PRBS15 由来)"),
        Step("偏波回転 U を掛ける", note="SU(2) で 2 偏波を混合"),
        Step("複素 AWGN を付加", note="QPSK 10dB / 16QAM 15dB"),
        Step("共通レーザ位相雑音を乗算", note="線幅 10 kHz, exp(jθ)"),
        Step("2x2 MIMO LMS で適応等化", note="comm.mimo_lms(rx, s, L, μ)"),
        Branch("収束後の区間か?", yes="BER 集計に使う", no="ウォームアップとして除外"),
        Step("2 偏波合算で等化後 BER を計算", note="解析解と比較"),
        Branch("次のタップ数 L があるか?", yes="L を変えて再等化", no="次の方式へ"),
        Step("等化前後コンステレーションを保存", note="mimo_ber.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="2x2 MIMO 等化と BER の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
