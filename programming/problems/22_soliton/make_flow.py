"""問題4-3の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("時間グリッドを用意", note="N=2^13 点, 幅 400 ps"),
        Step("FWHM から T0 を換算", note="t0_from_fwhm_sech"),
        Step("N=1 のピーク強度 P0 を計算", note="P0 = |β2|/(γ T0^2)"),
        Step("初期 sech パルス A0 を作る", note="sech_pulse"),
        Branch("各伝搬距離 z について", yes="伝搬を計算", no="ループ終了"),
        Step("ソリトンを伝搬 (分散+非線形)", note="propagate_ssfm"),
        Step("分散のみでも伝搬 (比較用)", note="dispersion_step"),
        Step("両者の FWHM を測る", note="fwhm_of"),
        Step("波形と FWHM 推移を描画して保存", note="soliton.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="基本ソリトン (N=1) 伝搬の処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
