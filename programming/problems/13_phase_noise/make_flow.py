"""問題3-1の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("線幅 df・サンプリングレート fs を設定", note="df=100 kHz, fs=32 GHz"),
        Step("位相増分の分散を計算", note="σ_PN^2 = 2π·df/fs"),
        Step("位相雑音 θ[n] を生成", note="初期値は一様、増分を累積 (cumsum)"),
        Step("複素電界 e[n] = exp(jθ[n]) を作る"),
        Step("FFT してパワー |FFT|^2 を加算", note="長さ L=2^21 のセグメント"),
        Branch("K=48 セグメント平均したか?", yes="平均して PSD 確定", no="次のセグメントを生成"),
        Step("ローレンツ型に当てはめ FWHM を測定", note="1/S を f^2 に直線フィット"),
        Step("時間波形とスペクトルを描画して保存", note="phase_noise.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="レーザ位相雑音と線幅確認の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
