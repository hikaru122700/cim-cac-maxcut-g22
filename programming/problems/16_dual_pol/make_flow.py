"""問題3-4の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("PRBS15 を生成", note="comm.generate_prbs(15)"),
        Step("ビットを 16QAM シンボルに変換", note="comm.bits_to_symbols"),
        Step("N=10^6 シンボルになるまで繰り返し並べる", note="np.tile で長さを揃える"),
        Step("これを x偏波とする", note="x = make_symbols(...)"),
        Step("x を 1000 サンプル遅延して y偏波を作る", note="y = np.roll(x, 1000)"),
        Step("x と y を縦に積んで (2, N) 配列にする", note="dual = np.vstack([x, y])"),
        Step("平均電力と x·y* 相関を確認", note="相関 ≈ 0 で無相関"),
        Step("両偏波のコンステレーションを描画して保存", note="dual_pol.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="両偏波 QAM シンボル列の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
