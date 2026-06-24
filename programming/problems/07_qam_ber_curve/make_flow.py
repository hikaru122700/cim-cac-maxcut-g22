"""問題2-1の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("QAM 方式と SNR の組を 1 つ選ぶ", note="M=4,16,64,256 / SNR=0..30dB"),
        Step("ランダムビットを生成", note="N_SYM*log2(M) ビット"),
        Step("QAM シンボルにマッピング", note="comm.bits_to_symbols"),
        Step("SNR 指定で AWGN を付加", note="comm.add_awgn"),
        Step("硬判定+デマッピングで誤りを計数", note="comm.symbols_to_bits"),
        Step("BER = 誤りビット数 / 全ビット数を記録"),
        Branch("全 SNR・全方式を測ったか?", yes="次へ", no="次の組へ戻る"),
        Step("解析解 ber_theory_qam を細かい SNR で計算"),
        Step("実測点と解析解を片対数で描画して保存", note="ber_vs_snr.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="BER-SNR 曲線の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
