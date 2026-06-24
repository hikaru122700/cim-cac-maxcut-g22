"""問題3-5 の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("16QAM シンボル列を生成", note="make_symbols (PRBS15 由来)"),
        Step("y 偏波を DELAY だけ巡回シフト", note="np.roll(x, 1000)"),
        Step("2 偏波ベクトルに積む", note="s = vstack([x, y]) -> (2, N)"),
        Step("ランダムユニタリ U を生成", note="unitary_group.rvs(2)"),
        Step("det の平方根で割り SU(2) 化", note="U_su = U / sqrt(det U)"),
        Step("偏波回転を一括適用", note="r = U @ s -> (2, N)"),
        Step("電力保存を数値検証", note="p_in == p_out == 2.0"),
        Step("回転前後のコンステレーションを描画", note="pol_rotation.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="偏波回転 SU(2) 付加の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
