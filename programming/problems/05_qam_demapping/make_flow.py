"""問題1-5の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
ここでは「シンボル列 -> ビット列」のデマッピング (judge + demap) の流れを示す。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("受信シンボル列を受け取る", note="複素数 (規格化済み)"),
        Step("規格化を外し整数振幅へ戻す", note="x = sym * qam_norm(M)"),
        Step("最近傍レベル番号に丸める", note="idx = round((x+(L-1))/2)  これが硬判定"),
        Step("レベル番号をグレイ符号へ", note="g = idx ^ (idx>>1)"),
        Step("I軸・Q軸を MSB first でビット化", note="_int_to_bits"),
        Step("前半=I, 後半=Q として連結", note="np.concatenate -> ravel"),
        Branch("無雑音か?", yes="入力ビットと完全一致", no="シンボル誤り1個 = ビット誤り1個"),
        Step("ビット列を返す / ラベル図を保存", note="gray_labels.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="QAMデマッピングの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
