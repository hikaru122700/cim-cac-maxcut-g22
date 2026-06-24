"""問題4-7の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("送信波形を生成", note="PRBS->QPSK->RC整形 (sps=2)"),
        Step("平均電力を Pin に正規化", note="A = sqrt(Pin) * w"),
        Step("1スパンを Split-Step 伝搬", note="propagate_ssfm (損失+分散+非線形)"),
        Step("EDFA 利得 sqrt(G) を掛ける", note="雑音は受信端でまとめて付加"),
        Branch("nspan 回まわしたか?", yes="受信処理へ", no="次のスパンへ戻る"),
        Step("総波長分散を完全補償", note="dispersion_step(-nspan*L)"),
        Step("ISI-free 抽出 + SPM平均位相を除去", note="downsample / exp(-j*ph)"),
        Step("受信端で ASE 雑音を付加", note="SNR = Pin/(N*S_ASE*Rs)"),
        Step("判定して BER を計算し描画", note="multispan_nl_ber.png 他"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="損失+分散+非線形 多中継伝送の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
