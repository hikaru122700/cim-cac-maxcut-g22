"""問題4-1の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("時間グリッドを用意", note="N=2^14, 窓 ±400 ps"),
        Step("入力ガウスパルス A0 を生成", note="fiber.gaussian_pulse"),
        Step("T0 と分散長 L_D を計算", note="L_D = T0^2/|β2|"),
        Step("角周波数グリッド ω を作る", note="fiber.omega_grid"),
        Step("距離 z を分散ステップで伝搬", note="fft → ×exp(jβ2ω²z/2) → ifft"),
        Step("各 z で強度 FWHM を測定", note="fiber.fwhm_of"),
        Branch("z リストを全部回したか?", yes="次へ", no="次の z へ戻る"),
        Step("解析解 FWHM=√(1+(z/L_D)²) と比較"),
        Step("波形と FWHM(z) を描画して保存", note="dispersion.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="波長分散によるパルス広がりの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
