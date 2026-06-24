"""問題4-5の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("EDFA の利得と ASE を準備", note="G=スパン損失, S_ASE=ase_psd(G,NF)"),
        Step("QPSK シンボル列を生成", note="PRBS15 -> bits_to_symbols"),
        Step("入力パワー Pin を 1 つ選ぶ", note="-16〜+12 dBm を走査"),
        Step("振幅を Pin にスケール", note="A = sqrt(Pin)·s"),
        Branch("nspan 回まわしたか?", yes="次へ", no="損失 -> EDFA を 1 スパン分"),
        Step("単位電力へ戻して硬判定", note="r = A/sqrt(Pin)"),
        Step("実測 BER と理論 BER を計算", note="SNR_N = Pin/(N·S_ASE·Rs)"),
        Step("全スパン数を重ね描きして保存", note="multispan_loss_ber.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="多中継(損失+EDFA)QPSK BER の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
