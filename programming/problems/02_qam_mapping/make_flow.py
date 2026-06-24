"""問題1-2の処理フロー図 (flow.png) を生成する。

共通ヘルパー _common/flowchart.py を使う。解説 explanation.md から参照する。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))

from flowchart import Branch, Step, draw_flowchart


def main() -> None:
    nodes = [
        Step("PRBS を 1 周期生成", note="comm.generate_prbs(15) 長さ 32767"),
        Step("各 QAM 方式 M=4,16,64,256 を順に処理"),
        Step("k = log2(M) を求める", note="1 シンボルのビット数 = 繰り返し数"),
        Step("PRBS を k 回繰り返す", note="np.tile(prbs, k) で k*32767 ビット"),
        Step("k ビットずつに区切る", note="前半 k/2 を I 軸, 後半を Q 軸"),
        Step("各軸をグレイ符号として振幅にマップ", note="グレイ->2進->振幅 a=2*idx-(L-1)"),
        Step("sqrt(2(M-1)/3) で割り規格化", note="平均電力を 1 にそろえる"),
        Branch("4 方式すべて処理したか?", yes="図の保存へ", no="次の M へ戻る"),
        Step("4 枚のコンステレーションを描画して保存", note="constellation.png"),
    ]
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow.png")
    draw_flowchart(nodes, out_path=out, title="QAM マッピングの処理フロー")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
