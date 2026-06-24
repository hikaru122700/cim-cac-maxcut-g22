"""問題1-6の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("QAM 方式 M を 1 つ選ぶ", note="M = 4, 16, 64, 256"),
        Step("PRBS から送信シンボルを作る", note="bits_to_symbols"),
        Step("送信ビット tx_bits を逆算", note="symbols_to_bits(tx_sym)"),
        Step("AWGN を付加して伝送", note="SNR = 20 dB"),
        Step("硬判定 + デマッピングで rx_bits", note="symbols_to_bits(rx_sym)"),
        Step("不一致を数えて BER を計算", note="count_bit_errors / 総ビット"),
        Branch("4 方式すべて終わったか?", yes="計測終了", no="次の M へ戻る"),
        Step("処理時間を表示", note="time.time() の差"),
        Step("実測 BER と解析解を図示", note="ber.png (描画は計測外)"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="BER 測定と処理時間計測の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
