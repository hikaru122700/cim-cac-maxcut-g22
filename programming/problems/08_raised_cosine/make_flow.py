"""問題2-2の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("サンプリング周波数を決める", note="fs >= (1+α)Rs = 2 sps"),
        Step("PRBS から QAM シンボル列を作る", note="bits_to_symbols"),
        Step("sps 倍にアップサンプル", note="シンボル間に 0 を挿入"),
        Step("RC フィルタで畳み込み整形", note="np.convolve(up, h)"),
        Step("シンボル中心インデックスを計算", note="delay + n*sps"),
        Branch("中心サンプル == 元シンボル?", yes="ISI なし (誤差 0)", no="整形/遅延を見直す"),
        Step("時間波形を描画して保存", note="waveform.png"),
        Step("コンステレーションを描画して保存", note="constellation_rc.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="レイズドコサイン整形の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
