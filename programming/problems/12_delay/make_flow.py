"""問題2-6の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("PRBS から QAM シンボル列を作る", note="16QAM, comm.bits_to_symbols"),
        Step("時間軸を用意", note="T=20 ps, Ts=10 ps (2 sps)"),
        Step("各遅延 τ=0,5,10,15,20 ps でループ"),
        Step("連続波形 x(t-τ) を評価", note="RC パルスの重ね合わせ"),
        Step("サンプル時刻 mTs-τ で再標本化", note="遅延サンプル列 x_τ[m]"),
        Branch("τ は整数サンプルか?", yes="ずらすだけ (10,20 ps)", no="端数=波形を補間 (5,15 ps)"),
        Step("基準・遅延波形・サンプルを重ね描き"),
        Step("5 段の図を保存", note="delay.png"),
        Step("τ=20 ps が 2 サンプルずらしと一致を検証"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="QAM サンプル列の時間遅延の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
