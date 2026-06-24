"""問題3-7の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("送信シンボルを生成", note="2偏波分 (x と遅延コピー y)"),
        Step("2 sps でパルス整形", note="α=0 の sinc, pulse_shape"),
        Step("偏波回転 U を掛ける", note="2x2 ユニタリ SU(2)"),
        Step("AWGN と位相雑音を付加", note="add_awgn, laser_phase_noise"),
        Step("サンプリング位相を T/2 ずらす", note="base に T_OFFSET=1 を仕込む"),
        Branch("タップ数 L は?", yes="L=1 単一タップ", no="L=10 分数間隔 FSE"),
        Step("2x2 MIMO LMS で等化", note="mimo_lms_fse, sps ごとにスライド"),
        Step("BER を計算し解析解と比較", note="ber_of, ber_theory_qam"),
        Step("コンステレーションを描画して保存", note="mimo_2sps.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="2 sps MIMO 等化の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
