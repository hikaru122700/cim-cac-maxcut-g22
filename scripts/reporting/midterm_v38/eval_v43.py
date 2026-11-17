"""v43 の「手法:評価方法と条件」：左に評価指標・問題・条件、右に実験のアーキテクチャー図（条件は箱の中）。"""
from pptx.util import Inches

import build_v38 as b
from lib38 import BLACK, RED, WHITE, block_arrow, box, fill_text, line, text

GRID = "{5940675A-B579-460E-94D1-54222C63F5DA}"


def T0(rest=""):
    return [("T", "v"), ("0", "vs"), rest]


def evaluation(prs):
    s = b.new_slide(prs, "手法:評価方法と条件")
    # ---- 左：評価指標・問題・条件 ----
    text(s, 0.3, 0.95, 6.0, 0.4, ["評価指標：既知の最良値（BKS）との差"], 18)
    box(s, 0.7, 1.4, 4.6, 0.8, None, kind="rect", lw=1.5)
    b.picture(s, "s16_17.png", 1.05, 1.53, w=3.9)
    text(s, 0.3, 2.25, 6.0, 0.4, ["小さいほど良い（0% で既知の最良値）"], 16, "c")
    text(s, 0.3, 2.75, 6.0, 0.4, ["ベンチマーク問題（BKS は文献 [5]）"], 18)
    rows = [["問題", "頂点数", "辺数", "BKS"], ["G22", "2000", "19990", "13359"]]
    gf = s.shapes.add_table(2, 4, Inches(0.6), Inches(3.2), Inches(5.0), Inches(0.7))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = GRID
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = gf.table.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = b.LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 16, "c")
    text(s, 0.3, 4.05, 6.0, 0.4, ["条件"], 18)
    text(s, 0.3, 4.45, 6.0, 2.0, [
        ["・SA：温度 "] + T0(" = 2.0 → 0.001、5000 スイープ（3 者で同じ）"),
        "・CIM：2400 周回を 256 回、カット値の上位 16 個を使う",
        "・これを独立に 100 組（計 1600 本の SA）",
        "・SA の解：ランダム開始の SA で得た解から、CIM の解 1 つずつに同じカット値の解を選ぶ",
        "・そろえたカット値：13317〜13353（平均 13330）",
        "・検定：16 本の平均どうしを 100 組で比べる（Wilcoxon 検定）"], 14, space=2)

    # ---- 右：アーキテクチャー図 ----
    text(s, 6.5, 0.95, 6.6, 0.4, ["実験のアーキテクチャー"], 18, "c")
    box(s, 7.9, 1.4, 3.8, 0.55, ["G22（頂点 2000、辺 19990）"], 14)
    box(s, 6.55, 2.2, 6.55, 3.85, None, kind="round", lw=1.5, dash=True)
    text(s, 11.75, 1.7, 1.45, 0.5, ["× 100 組", "（独立）"], 14, "c")
    cols = [(6.7, [("CIM の解", "r")], ["CIM", "2400 周回 × 256 回"], ["カット値の", "上位 16 個"], RED),
            (8.85, ["SA の解"], ["ランダム開始の SA", "（5000 スイープ）"], ["CIM の解と同じ", "カット値の解 16 個"], BLACK),
            (11.0, ["ランダムな解"], ["ランダムな解", "16 個"], None, BLACK)]
    for x, title, b1, b2, col in cols:
        cx = x + 0.98
        line(s, 9.8, 1.95, cx, 2.6, lw=1.5, arrow=True)
        text(s, x, 2.55, 1.95, 0.35, title, 14, "c")
        box(s, x, 2.9, 1.95, 0.8, b1, 13, line=col, lw=2.0)
        line(s, cx, 3.7, cx, 3.9, lw=1.5, arrow=True)
        if b2:
            box(s, x, 3.9, 1.95, 0.75, b2, 13)
        else:
            line(s, cx, 3.9, cx, 4.65, lw=1.5)
        line(s, cx, 4.65, cx, 4.85, lw=1.5, arrow=True)
    box(s, 6.7, 4.85, 6.25, 0.4, [["SA（"] + T0(" = 2.0 → 0.001、5000 スイープ）を各 16 本")], 13)
    line(s, 9.82, 5.25, 9.82, 5.5, lw=1.5, arrow=True)
    box(s, 6.7, 5.5, 6.25, 0.4, ["組ごとに 16 本の BKS との差を平均"], 13)
    line(s, 9.82, 5.9, 9.82, 6.2, lw=1.5, arrow=True)
    box(s, 6.7, 6.2, 6.25, 0.4, ["100 組で比べる（Wilcoxon 検定）"], 13)
    block_arrow(s, 6.07, 6.64, 1.2, 0.22)
    text(s, 0.4, 6.86, 12.5, 0.45, [["開始のカット値は同じなので、差が出れば", ("カット値以外の違い", "r"), "による"]], 18, "c")
    return s
