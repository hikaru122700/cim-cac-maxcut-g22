"""問題2-4の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("送信ビットを生成", note="rng.integers, N_SYM*k ビット"),
        Step("QAM マッピング", note="comm.bits_to_symbols"),
        Step("raised cosine で整形", note="comm.pulse_shape, 2 sps"),
        Step("複素 AWGN を付加", note="comm.add_awgn, SNR=20 dB"),
        Step("ISI-free でダウンサンプリング", note="idx = delay + n*sps"),
        Step("硬判定 + デマッピング", note="comm.symbols_to_bits"),
        Step("送信ビットと比較し BER を計算", note="count_bit_errors / N"),
        Branch("全 QAM 方式を回したか?", yes="図の作成へ", no="次の M へ戻る"),
        Step("実測 BER と解析解を棒グラフで保存", note="ber_rc.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="整形QAMのダウンサンプリングとBERの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
