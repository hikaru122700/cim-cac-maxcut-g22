"""問題1-3 の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("PRBS15 を 1 周期生成", note="comm.generate_prbs(15)"),
        Step("k=log2(M) 本に複製しビット列を作る", note="np.tile(prbs, k)"),
        Step("QAM シンボルへマッピング", note="comm.bits_to_symbols"),
        Step("シンボルを繰り返して 10^6 個に増やす", note="np.tile -> [:N_SYM]"),
        Step("雑音電力 N0 = Ps / 10^(SNR/10) を決める", note="SNR=20dB -> N0=0.01"),
        Step("複素 AWGN を付加 r = s + n", note="comm.add_awgn"),
        Step("実測 SNR を逆算して確認", note="10log10(Ps/<|n|^2>)"),
        Branch("M = 4,16,64,256 すべて処理したか?", yes="図の描画へ", no="次の M へ"),
        Step("4 つのコンステレーションを描画して保存", note="constellation_awgn.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="QAM + AWGN コンステレーションの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
