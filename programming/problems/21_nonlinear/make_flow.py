"""問題4-2 (非線形効果 SPM) の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("時間グリッドを作る", note="N=2^14, Tspan=400 ps"),
        Step("入力ガウスパルス A0 を生成", note="fiber.gaussian_pulse(P0=1W, FWHM=10ps)"),
        Step("ピーク位置 i_peak と L_NL を求める", note="L_NL = 1/(γ P0)"),
        Step("距離 z を 0..5 km で 51 点とる"),
        Step("各 z で A = A0·exp(jγ|A0|²z) を計算", note="fiber.nonlinear_step"),
        Step("ピーク位相 angle(A·conj(A0)) を測る", note="np.unwrap で連続化"),
        Step("解析解 φ = γ P0 z と比較", note="最大誤差 ~1e-16 rad"),
        Step("位相曲線とスペクトル広がりを描画して保存", note="nonlinear.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="SPM ピーク位相変化の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
