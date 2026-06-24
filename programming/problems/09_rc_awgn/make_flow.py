"""問題2-3 の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("PRBS から QAM シンボルを生成", note="make_symbols, 各 M で 50 万シンボル"),
        Step("raised cosine で整形", note="comm.pulse_shape: 0挿入アップサンプル + 畳み込み"),
        Step("複素 AWGN を付加", note="comm.add_awgn(signal_power=1.0) で N0=0.01"),
        Step("ISI-free タイミングでダウンサンプル", note="idx = delay + n*sps"),
        Step("実測 SNR を計算", note="10log10(1/<|s_hat - s|^2>)"),
        Branch("4 つの M を処理したか?", yes="次へ", no="次の M へ戻る"),
        Step("4 枚のコンステレーションを描画して保存", note="constellation_rc_awgn.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="RC整形 + 複素AWGN の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
