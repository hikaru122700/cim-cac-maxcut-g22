"""問題4-6の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("QPSK 送信シンボル列を作る", note="PRBS15 -> bits_to_symbols"),
        Step("角周波数グリッド ω を用意", note="dt=1/Rs を ps に換算"),
        Step("入力パワー Pin で振幅をスケール", note="A = sqrt(Pin)*s"),
        Step("1 スパン: 分散 -> 損失 -> EDFA(+ASE)", note="N スパン繰り返す"),
        Branch("N スパン回したか?", yes="補償へ", no="次のスパンへ"),
        Step("総分散を一括で逆補償", note="dispersion_step(z=-N*L)"),
        Step("最近傍判定でビットに戻す", note="symbols_to_bits"),
        Step("BER を数え理論値と比較", note="Pin を掃引"),
        Step("BER 曲線と前後コンステレーションを保存", note="*.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="損失+分散の多中継と分散補償の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
