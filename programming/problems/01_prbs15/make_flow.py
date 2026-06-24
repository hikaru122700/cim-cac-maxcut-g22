"""問題1の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("レジスタを初期化", note="全ビット 1 (全 0 は禁止)"),
        Step("b14 を 1 ビット出力"),
        Step("帰還ビット = b13 XOR b14 を計算"),
        Step("右に 1 シフトし b0 へ帰還を入れる", note="reg = [feedback] + reg[:14]"),
        Branch("32767 回まわしたか?", yes="ループ終了", no="出力へ戻る"),
        Step("双極性に変換 0->-1, 1->+1", note="直流成分を消す"),
        Step("FFT してパワー |X|^2 を計算", note="np.fft.rfft"),
        Step("時間波形・平坦・sinc^2 を描画して保存", note="spectrum.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="PRBS15 強度スペクトルの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
