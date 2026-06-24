"""問題2-5 の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("SNR 値を 1 つ選ぶ", note="0,2,...,30 dB を順に"),
        Step("ランダムビットを QAM シンボルに変換", note="comm.bits_to_symbols"),
        Step("raised cosine で整形", note="comm.pulse_shape (2 sps, α=1)"),
        Step("複素 AWGN を付加", note="comm.add_awgn"),
        Step("ISI-free タイミングで間引き", note="comm.downsample_isi_free"),
        Step("硬判定してビットに戻し誤りを数える", note="comm.symbols_to_bits"),
        Branch("全 SNR を測ったか?", yes="次の M / 描画へ", no="次の SNR へ戻る"),
        Step("実測点と解析解を重ねて描画して保存", note="ber_vs_snr_rc.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="整形QAM の BER-SNR 曲線の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
