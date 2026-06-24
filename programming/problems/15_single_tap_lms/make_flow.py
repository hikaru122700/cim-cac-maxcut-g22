"""問題3-3の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("送信 QAM シンボル列を作る", note="PRBS15 を QAM マップ"),
        Step("レーザ位相雑音 θ[n] を生成", note="線幅 10 kHz のランダムウォーク"),
        Step("受信信号 u[n] = s[n]·e^{jθ[n]} を作る"),
        Step("タップを初期化", note="m = 1"),
        Step("等化器出力 v = m·u を計算"),
        Step("タップ更新 m += μ(d−v)·conj(u)", note="d は参照符号"),
        Branch("全シンボル処理したか?", yes="ループ終了", no="次のシンボルへ"),
        Step("収束後の区間で BER を評価", note="先頭 5000 を除外"),
        Step("等化前後とタップ追従を描画", note="lms_equalization.png / tap_tracking.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="単一タップ LMS 適応等化の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
