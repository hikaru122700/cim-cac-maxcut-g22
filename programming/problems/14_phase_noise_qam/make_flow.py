"""問題3-2の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("パラメータと乱数器を用意", note="Rs=32 Gbaud, df=10 kHz, N=10^6"),
        Step("位相増分の分散を計算", note="σ_PN^2 = 2π·df/fs"),
        Step("QAM シンボル列を生成", note="make_symbols (PRBS -> QAM)"),
        Step("レーザ位相雑音 θ[n] を生成", note="comm.laser_phase_noise (累積和)"),
        Step("各シンボルを θ[n] だけ回転", note="rx = tx · exp(j·θ)"),
        Branch("QPSK / 16QAM / 64QAM 全部やったか?", yes="図の保存へ", no="次の M へ戻る"),
        Step("先頭 6000 点を散布図に描画", note="I-Q 平面"),
        Step("3 枚並べて保存", note="constellation_pn.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="位相雑音付き QAM コンステレーションの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
