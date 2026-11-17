"""卒論中間報告 v38：佐々木さん（2025/11/27 中間報告、CIM）の資料を土台に、同じ書式・図の流儀で作り直す。
背景の CIM・イジングの図は佐々木さんの図を使い、SA・GA・初期解・手法の図は同じ線画の流儀で描く。"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
from lib38 import (BLACK, BLUE, GRAY, RED, WHITE, ball, bezier, block_arrow, box, copy_shape, fill_text,  # noqa: E402
                   landscape, line, local_min, node, picture, polyline, renumber, spin, spin_row, text)
from pptx.dml.color import RGBColor  # noqa: E402

SRC = Path(__file__).parent.parent / "senpai" / "sasaki.pptx"
import os
DST = Path(os.environ.get("V38_OUT", "C:/Users/owner/Downloads/卒論中間報告_三輪光_v38.pptx"))
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
PINK = RGBColor(0xFF, 0xE5, 0xE5)


def new_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[11])  # パラグラフスライド
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx == 11:
            ph.text_frame.text = title
        else:
            ph._element.getparent().remove(ph._element)
    return s


def set_title(s, title):
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 11:
            r = ph.text_frame.paragraphs[0].runs
            r[0].text = title
            for x in r[1:]:
                x._r.getparent().remove(x._r)


# ---------------- 表紙・背景 ----------------

def cover(s):
    title = next(sh for sh in s.shapes if sh.name.startswith("タイトル"))
    runs = title.text_frame.paragraphs[0].runs
    runs[1].text = "コヒーレントイジングマシンの出力解を初期解とする組合せ最適化の効率化"
    runs[1].font.size = Pt(32)
    runs[2].text, runs[3].text, runs[4].text = "11", "月", "11"
    for r in runs[2:]:
        r.font.size = Pt(32)
    sub = next(sh for sh in s.shapes if sh.name.startswith("字幕"))
    sub.text_frame.paragraphs[0].runs[0].text = "三輪光"


def maxcut(prs):
    s = new_slide(prs, "背景:マックスカット問題")
    text(s, 0.4, 0.95, 12.5, 0.45, ["頂点を2つのグループ（白・黒）に分け、グループをまたぐ辺（切断される辺）の数を最大にする問題"], 18, "c")
    pts = {"A": (1.0, 2.1), "B": (2.9, 2.1), "C": (4.8, 2.1), "D": (1.0, 4.0), "E": (2.9, 4.0), "F": (4.8, 4.0)}
    black = {"B", "D", "F"}
    for a, b in ["AB", "BC", "AD", "BE", "CF", "DE", "EF", "AE", "BF"]:
        cut = (a in black) != (b in black)
        line(s, *pts[a], *pts[b], color=RED if cut else BLACK, lw=3.0 if cut else 1.5, dash=not cut)
    for k, (cx, cy) in pts.items():
        node(s, cx, cy, 0.55, k in black)
    text(s, 0.5, 4.5, 5.0, 0.8, [[("赤の実線", "r"), "：切断される辺（7本）"], "破線：同じグループ内の辺"], 16, "c")
    picture(s, "s03_16.png", 6.4, 1.75, w=4.4)
    text(s, 6.3, 2.75, 6.8, 0.8, [[("σ", "v"), ("i", "vs"), " = ±1：頂点 ", ("i", "v"), " のグループ、", ("w", "v"), ("ij", "vs"),
                                   "：辺の重み"]], 16)
    text(s, 6.3, 3.35, 6.8, 0.8, [["頂点数 ", ("N", "v"), " のグラフでは解の候補が 2", ("N", "vp"), " 通り"],
                                  [("（", ""), ("N", "v"), " = 2000 で約 10", ("602", "p"), " 通り）"]], 18)
    block_arrow(s, 9.2, 4.2, 1.0, 0.32)
    text(s, 6.3, 4.55, 6.8, 0.45, [[("大規模な問題を厳密に解くのは現実的でない（NP困難）", "r")]], 18, "c")
    labels = ["回路設計\n（配線の分割）", "通信\n（資源割当）", "創薬・材料\n（構造探索）", "物流\n（配送計画）"]
    text(s, 0.5, 5.25, 12.3, 0.4, ["回路設計・通信・創薬・物流など、多くの問題がマックスカット（イジングモデル）に書き換えられる"], 18, "c")
    for k, lab in enumerate(labels):
        box(s, 1.3 + k * 2.75, 5.72, 2.35, 0.72, lab.split("\n"), 16, kind="round")
    block_arrow(s, 6.07, 6.55, 1.2, 0.28)
    text(s, 1.5, 6.88, 10.3, 0.45, [["限られた時間で", ("良い近似解", "r"), "を得る手法が求められている"]], 18, "c")
    return s


def sim_model(prs, src7):
    s = new_slide(prs, "背景:本研究で用いるCIMのシミュレーションモデル")
    copy_shape(src7, 120, s, 0.2, 1.0, w=6.6)
    picture(s, "s06_15.png", 7.15, 0.95, w=5.95)
    text(s, 6.75, 4.35, 6.5, 0.45, ["パルスの振幅の時間変化（G22 の一部、赤：+1、青：−1）"], 16, "c")
    text(s, 0.3, 4.2, 6.0, 0.45, ["測定フィードバック型CIMの周回ごとの更新式"], 18, "c")
    picture(s, "s06_17.png", 0.6, 4.75, w=4.8)
    picture(s, "s06_18.png", 0.6, 5.35, w=4.95)
    text(s, 6.55, 4.85, 6.6, 1.6, [
        [("a", "v"), "：ループ損失　", ("bJ", "v"), "：測定フィードバックによる結合"],
        [("c n", "v"), ("0", "s"), "：雑音（周回ごとに引き直す乱数）"],
        [("α p", "v"), "(", ("n", "v"), ")：ポンプによる利得（周回ごとに増やす）"],
        [("β", "v"), "：利得飽和"]], 16)
    text(s, 0.4, 6.35, 12.5, 0.9, [
        ["最後に振幅の符号（+ / −）をスピンとして読み出し、カット値を計算する"],
        [("本研究では CIM を CPU 上で模擬し、", ""), ("出力解をそのまま次の手法へ渡す", "r")]], 18, "c")
    return s


def prior(prs):
    s = new_slide(prs, "背景:CIMの性能と課題（先行研究）")
    text(s, 0.4, 0.95, 12.5, 0.45, ["CIMの性能はグラフの種類によって異なる"], 18, "c")
    box(s, 0.7, 1.55, 5.7, 2.6, [[("密なグラフ", "B")], "（全頂点がつながる完全グラフなど）", "",
                                  ["CIM が SA より約 50〜1000 倍速い [1, 2]"], ["量子アニーリングよりも有利 [3]"]], 18,
        kind="round", lw=3.0)
    box(s, 6.9, 1.55, 5.7, 2.6, [[("疎なグラフ", "B")], "（辺が少ないグラフ：G-set など）", "",
                                  ["同じ時間で SA にわずかに劣る [4]"], ["3正則グラフでは量子アニーリングに劣る [3]"]], 18,
        kind="round", lw=3.0, line=RED)
    # 小さなグラフの絵（密・疎）
    import math
    for cx, dense in ((3.55, True), (9.75, False)):
        ps = [(cx + 0.55 * math.cos(2 * math.pi * k / 6), 4.95 + 0.55 * math.sin(2 * math.pi * k / 6)) for k in range(6)]
        pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)] if dense else [(0, 1), (2, 3), (4, 5), (1, 4)]
        for i, j in pairs:
            line(s, *ps[i], *ps[j], lw=1.0)
        for k, (px, py) in enumerate(ps):
            node(s, px, py, 0.2, k % 2 == 0)
    block_arrow(s, 6.07, 5.65, 1.2, 0.32)
    text(s, 0.5, 6.0, 12.3, 0.45, ["疎なグラフでは、CIM 単独で既存の手法（SA など）に勝つのは難しい"], 18, "c")
    block_arrow(s, 6.07, 6.5, 1.2, 0.32)
    text(s, 0.5, 6.85, 12.3, 0.45, [[("CIM を既存の手法と組み合わせて使う", "r"), "ことで、CIM を活かせないか？"]], 18, "c")
    return s


def sa(prs):
    s = new_slide(prs, "背景:発見的解法 シミュレーテッドアニーリング(SA)")
    text(s, 0.4, 0.95, 12.5, 0.45, ["1つの解からスピンを1個ずつ反転し、エネルギーの低い（カット値の大きい）解へ移っていく"], 18, "c")
    pos = landscape(s, 1.1, 1.75, 6.3, 3.6)
    u1, u2 = local_min(0.33), local_min(0.68)
    ball(s, pos, 0.05, color=GRAY)
    ball(s, pos, u1)
    ball(s, pos, u2)
    a, b = pos(u1, 0.3), pos(u2, 0.3)
    polyline(s, bezier(a, ((a[0] + b[0]) / 2, a[1] - 1.6), b), color=RED, lw=2.0, arrow=True, dash=True)
    text(s, 2.9, 1.25, 3.6, 0.7, [[("悪くなる反転も確率で受け入れ", "r")], [("→ 山を越えて深い谷へ", "r")]], 16, "c")
    text(s, pos(u1)[0] - 0.9, pos(u1)[1] + 0.05, 1.8, 0.4, ["局所解"], 16, "c")
    text(s, pos(u2)[0] + 0.15, pos(u2)[1] - 0.45, 1.2, 0.4, ["最良解"], 16)
    text(s, pos(0.05)[0] - 0.8, pos(0.05)[1] - 0.68, 1.6, 0.4, ["初期解"], 16, "c")
    # 右：手順とスピンの反転
    text(s, 7.75, 1.55, 5.4, 0.4, ["1回の操作：スピンを1個選んで反転を試す"], 16)
    bits = [1, 0, 0, 1, 1, 0, 1, 0, 1]
    spin_row(s, 8.1, 2.05, bits)
    box(s, 8.1 + 4 * 0.29 - 0.06, 1.98, 0.34, 0.55, kind="round", line=BLACK, lw=2.0)
    block_arrow(s, 9.25, 2.6, 0.5, 0.3, color=BLACK)
    flipped = bits[:]
    flipped[4] = 0
    spin_row(s, 8.1, 2.95, flipped)
    text(s, 9.85, 2.52, 1.4, 0.4, ["反転"], 16)
    text(s, 7.75, 3.6, 5.5, 1.9, [
        "・良くなる反転 → 受け入れる",
        ["・悪くなる反転 → 確率 e", ("−ΔH/T", "vp"), " で受け入れる"],
        ["・温度 ", ("T", "v"), " を徐々に下げる"],
        "　高温：大きく動く → 低温：谷に落ち着く"], 18, space=4)
    block_arrow(s, 6.07, 5.95, 1.2, 0.32)
    text(s, 0.5, 6.3, 12.3, 0.9, [["探索を長く続けるほど良い解が得られるが、", ("時間がかかる", "r")],
                                  ["探索は", ("初期解", "r"), "から始まる（通常はランダムな解）"]], 18, "c")
    return s


def ga(prs):
    s = new_slide(prs, "背景:発見的解法 遺伝的アルゴリズム(GA)")
    text(s, 0.4, 0.95, 12.5, 0.45, ["多数の解（個体）の集団を持ち、良い解どうしを組み合わせて新しい解を作る"], 18, "c")
    pop = [[1, 0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 0, 1], [0, 1, 1, 0, 1, 1, 0], [1, 1, 0, 1, 0, 0, 1]]
    cuts = ["13", "11", "14", "10"]
    text(s, 0.4, 1.5, 3.4, 0.4, ["① 集団（初期集団）"], 18, "c")
    for k, b in enumerate(pop):
        end = spin_row(s, 0.55, 2.05 + k * 0.62, b, w=0.2, h=0.38, gap=0.08)
        text(s, end + 0.05, 2.05 + k * 0.62, 1.35, 0.4, [f"カット {cuts[k]}"], 14)
    block_arrow(s, 3.85, 3.0, 0.5, 0.45, "right", BLACK)
    # 交叉
    text(s, 4.4, 1.5, 4.2, 0.4, ["② 交叉"], 18, "c")
    p1, p2 = pop[0], pop[1]
    text(s, 4.4, 2.05, 0.75, 0.4, ["親1"], 16)
    spin_row(s, 5.15, 2.05, p1, w=0.2, h=0.38, gap=0.08)
    text(s, 4.4, 2.67, 0.75, 0.4, ["親2"], 16)
    spin_row(s, 5.15, 2.67, p2, w=0.2, h=0.38, gap=0.08)
    diff = [k for k in range(7) if p1[k] != p2[k]]
    line(s, 6.1, 3.15, 6.1, 3.45, lw=2.0, arrow=True)
    text(s, 4.4, 3.5, 0.75, 0.4, ["子"], 16)
    spin_row(s, 5.15, 3.5, p1, w=0.2, h=0.38, gap=0.08, gray=diff)
    text(s, 4.4, 4.0, 4.2, 0.8, ["一致するスピンを受け継ぐ", "（灰色：親で異なる → 選び直す）"], 14, "c")
    block_arrow(s, 7.15, 3.0, 0.5, 0.45, "right", BLACK)
    # 改良
    text(s, 7.7, 1.5, 2.6, 0.4, ["③ 改良"], 18, "c")
    child = p1[:]
    child[2] = 0
    child[4] = 0
    spin_row(s, 7.95, 2.6, child, w=0.2, h=0.38, gap=0.08)
    text(s, 7.7, 3.15, 2.6, 0.8, ["局所探索・", "突然変異で良くする"], 14, "c")
    block_arrow(s, 10.3, 3.0, 0.5, 0.45, "right", BLACK)
    text(s, 10.85, 1.5, 2.3, 0.4, ["④ 入れ替え"], 18, "c")
    text(s, 10.85, 2.2, 2.3, 1.6, ["集団で最も悪い", "個体と入れ替える"], 16, "c")
    polyline(s, [(12.0, 3.0), (12.0, 4.85), (2.1, 4.85), (2.1, 4.55)], lw=2.0, arrow=True)
    text(s, 5.0, 4.9, 3.5, 0.4, ["世代をくり返す"], 16, "c")
    block_arrow(s, 6.07, 5.6, 1.2, 0.32)
    text(s, 0.5, 5.95, 12.3, 0.9, [["多様な解を組み合わせて探すため良い解が得られやすいが、", ("時間がかかる", "r")],
                                   ["探索は", ("初期集団", "r"), "から始まる（通常はランダムな解）"]], 18, "c")
    return s


def init_sol(prs):
    s = new_slide(prs, "背景:探索の初期解の与え方")
    text(s, 0.4, 0.95, 12.5, 0.45, ["エネルギーの山谷の「どこから探索を始めるか」で、到達できる解が変わる"], 18, "c")
    pos = landscape(s, 1.1, 1.7, 6.0, 3.5)
    u2 = local_min(0.68)
    ball(s, pos, 0.04, color=GRAY)
    ball(s, pos, 0.62)
    rx, ry = pos(0.04)
    text(s, rx - 0.6, ry - 0.68, 2.4, 0.4, ["ランダムな初期解"], 16)
    cx, cy = pos(0.62)
    text(s, cx - 2.2, cy - 0.55, 1.9, 0.4, [("CIMの解", "r")], 16, "r")
    text(s, pos(u2)[0] + 0.15, pos(u2)[1] - 0.45, 1.2, 0.4, ["最良解"], 16)
    # 探索の道のり：ランダムからは長く、CIM からは短い
    a, b = pos(0.04, 0.35), pos(u2, 0.35)
    polyline(s, bezier(a, ((a[0] + b[0]) / 2, a[1] - 0.7), b), color=GRAY, lw=2.0, arrow=True, dash=True)
    a = pos(0.62, 0.3)
    polyline(s, bezier(a, (a[0] + 0.15, a[1] + 0.1), pos(u2, 0.3)), color=RED, lw=2.5, arrow=True)
    box(s, 7.5, 1.55, 5.5, 1.6, [[("従来：ランダムな初期解", "B")], "・作るコストはほぼ 0",
                                  ["・悪い地点から始まり、", ("長い探索が必要", "r")]], 18, kind="round", align="l", lw=2.0)
    box(s, 7.5, 3.4, 5.5, 2.0, [[("本研究：CIMの出力解", "B")], ["・CIM は全スピンを並列に更新し、", ("良い地点", "r"), "を短時間で出力"],
                                 ["・ただし CIM を動かす", ("時間が余分にかかる", "r")]], 18, kind="round", align="l", lw=2.0,
        line=RED)
    block_arrow(s, 6.07, 5.75, 1.2, 0.32)
    text(s, 0.5, 6.15, 12.3, 0.9, [["CIM の解から探索を始めると、", ("CIM の時間を含めても", "r"), "より良い解にたどり着けるか？"]], 20, "c")
    return s


def purpose(prs):
    s = new_slide(prs, "本研究の目的")
    box(s, 1.09, 1.9, 11.15, 1.9, [[("コヒーレントイジングマシンの出力解をSA・GAの初期解とすることで、"
                                    "限られた計算時間内に得られる解の品質が改善するかを明らかにする。", "")]], 24,
        kind="round", lw=3.0)
    text(s, 1.6, 4.4, 10.5, 1.8, [
        [("・CIM の解を初期解にしたときの、SA 1本あたりの解の品質を明らかにする。", "r")],
        "・CIM の計算時間を含めた同じ総時間で、ランダムな初期解と比べる。",
        "・CIM の解が解空間のどこにあるか（最良解との距離）を明らかにする。",
        "・効果が SA の探索時間やグラフ（規模・密度）にどう依存するかを明らかにする。"], 18, space=6)
    return s


# ---------------- 手法 ----------------

def method(prs):
    s = new_slide(prs, "手法:CIMの解を初期解とするSA・GA")
    box(s, 4.15, 0.95, 5.0, 0.6, ["マックスカット問題（G-set）"], 18)
    lanes = [(0.55, "提案：CIM → SA", ["CIM を 256 回実行", "（2400 周回）"], ["カット値の上位", "16 個を選ぶ"], ["SA を 16 本", "（各解から 1 本）"], RED),
             (4.65, "提案：CIM → GA", ["CIM を 10 回実行", "（4800 周回）"], ["10 個で", "初期集団を作る"], ["GA", "（世代を進める）"], RED),
             (8.75, "比較：ランダム", ["ランダムな解", "を作る"], None, ["SA / GA", "（同じ設定）"], BLACK)]
    for x, title, b1, b2, b3, col in lanes:
        text(s, x, 1.9, 4.0, 0.4, [(title, "r" if col == RED else "")], 18, "c")
        box(s, x + 0.5, 2.4, 3.0, 0.9, b1, 18, line=col, lw=2.0)
        line(s, x + 2.0, 3.3, x + 2.0, 3.7, lw=2.0, arrow=True)
        if b2:
            box(s, x + 0.5, 3.7, 3.0, 0.9, b2, 18)
        else:
            line(s, x + 2.0, 3.7, x + 2.0, 4.6, lw=2.0)
        line(s, x + 2.0, 4.6, x + 2.0, 5.0, lw=2.0, arrow=True)
        box(s, x + 0.5, 5.0, 3.0, 0.9, b3, 18)
        line(s, x + 2.0, 5.9, x + 2.0, 6.25, lw=2.0, arrow=True)
        line(s, 6.65, 1.55, x + 2.0, 1.95, lw=1.5, arrow=True)
    box(s, 1.5, 6.25, 10.3, 0.6, ["カット値の比較（既知の最良値 BKS との差 gap）"], 18)
    text(s, 0.5, 6.9, 12.3, 0.5, [["① SA 1本あたりの到達点　② ", ("CIM の時間も含めた同じ総時間", "r"), "での最良値"]], 18, "c")
    return s


def evaluation(prs):
    s = new_slide(prs, "手法:評価方法と条件")
    text(s, 0.4, 0.95, 6.0, 0.4, ["評価指標：既知の最良値（BKS）との差"], 18)
    box(s, 0.9, 1.45, 4.6, 0.85, None, kind="rect", lw=1.5)
    picture(s, "s16_17.png", 1.25, 1.6, w=3.9)
    text(s, 0.4, 2.4, 6.0, 0.45, ["小さいほど良い（0% で既知の最良値）"], 16, "c")
    text(s, 0.4, 3.0, 6.0, 0.4, ["ベンチマーク問題（G-set）"], 18)
    rows = [["問題", "頂点数", "辺数", "BKS"], ["G22", "2000", "19990", "13359"], ["G55", "5000", "12498", "10299"],
            ["G70", "10000", "9999", "9591"]]
    gf = s.shapes.add_table(4, 4, Inches(0.8), Inches(3.5), Inches(5.0), Inches(1.6))
    tbl = gf.table
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 16, "c")
    text(s, 0.4, 5.2, 6.0, 0.8, ["G22：密度高め、G55・G70：疎で大きい", "（BKS：文献 [5] の既知の最良値）"], 16, "c")
    # 右：同じ総時間
    text(s, 6.6, 0.95, 6.5, 0.4, ["比較②：CIM の時間も含めた同じ総時間"], 18)
    text(s, 6.6, 1.5, 1.6, 0.5, ["ランダム開始"], 16)
    for k in range(9):
        box(s, 8.25 + k * 0.5, 1.5, 0.45, 0.5, ["SA"], 12, fill=WHITE, line=BLUE, lw=1.5)
    text(s, 6.6, 2.3, 1.6, 0.5, ["CIM 開始"], 16)
    box(s, 8.25, 2.3, 3.0, 0.5, ["CIM（256 回）"], 14, fill=PINK, line=RED, lw=1.5)
    box(s, 11.3, 2.3, 1.45, 0.5, ["SA×16"], 12, fill=WHITE, line=BLUE, lw=1.5)
    line(s, 8.25, 3.0, 12.75, 3.0, lw=1.5, arrow=True)
    line(s, 12.75, 3.0, 8.25, 3.0, lw=1.5, arrow=True)
    text(s, 8.25, 3.05, 4.5, 0.4, ["同じ総時間"], 16, "c")
    text(s, 6.6, 3.6, 6.6, 2.4, [
        "・ランダム開始は、同じ時間で回せるだけ SA を回す",
        "・それぞれ全 SA の最良値で比べる",
        ["・SA：温度 ", ("T", "v"), ("0", "vs"), " = 2.0 → 0.001、500 / 5000 スイープ"],
        "・CIM・SA・GA は CPU 上で実行し、時間は CPU 時間"], 16, space=4)
    text(s, 0.5, 6.4, 12.3, 0.8, [["比較①（1本あたり）は ", ("CIM の時間を含まない", "r"),
                                   "：初期解の「質」だけを比べる"]], 18, "c")
    return s


# ---------------- 結果・考察 ----------------

def result(prs, title, img, w, comments, concl, x=None, y=1.0, size=16):
    s = new_slide(prs, title)
    pic = picture(s, img, x if x is not None else (13.333 - w) / 2, y, w=w)
    top = pic.top / 914400 + pic.height / 914400 + 0.2
    n_lines = sum(-(-len("".join(c if isinstance(c, str) else c[0] for c in ([cm] if isinstance(cm, str) else cm))) // 60)
                  for cm in comments)
    est = n_lines * 0.34
    text(s, 0.6, top, 12.2, est, comments, size, space=3)
    ya = min(top + est + 0.25, 6.4)
    block_arrow(s, 6.07, ya, 1.2, 0.32)
    text(s, 0.5, ya + 0.38, 12.3, 0.6, concl, 20, "c")
    return s


def discussion(prs):
    s = new_slide(prs, "考察:なぜCIMの解から始めると良いのか")
    pos = landscape(s, 1.0, 1.4, 5.8, 3.4)
    u2 = local_min(0.68)
    for u in (0.63, 0.66, 0.71, 0.73):
        ball(s, pos, u, d=0.2)
    for u in (0.05, 0.2, 0.42, 0.86, 0.95):
        ball(s, pos, u, d=0.2, color=GRAY)
    cx, cy = pos(u2)
    box(s, cx - 0.75, cy - 1.05, 1.5, 1.2, None, kind="oval", line=RED, lw=2.0, dash=True)
    text(s, cx - 1.4, 1.05, 2.8, 0.4, [("谷（最良解の近く）", "r")], 16, "c")
    line(s, cx, 1.45, cx, cy - 1.05, color=RED, lw=1.5)
    text(s, 0.9, 5.25, 6.0, 0.4, [[("●", "r"), "：CIM の解　", ("●", "g"), "：ランダムな解"]], 16, "c")
    text(s, 7.2, 1.0, 5.9, 0.4, [("観察", "B")], 18)
    text(s, 7.2, 1.4, 5.9, 0.8, ["・SA 1本あたり、CIM 開始が良い（結果①）", "・同じカット値の解より良い（結果②）"], 16)
    block_arrow(s, 9.6, 2.25, 0.9, 0.28)
    text(s, 7.2, 2.55, 5.9, 0.4, [("仮説", "B")], 18)
    text(s, 7.2, 2.95, 5.9, 0.8, [["・CIM は", ("最良解に近い領域（谷）", "r"), "に解を出す"], "・SA はその近くで仕上げるだけでよい"], 16)
    block_arrow(s, 9.6, 3.8, 0.9, 0.28)
    text(s, 7.2, 4.1, 5.9, 0.4, [("裏付け（G22）", "B")], 18)
    text(s, 7.2, 4.5, 5.9, 0.8, [["・最良解との距離 ", ("d", "v"), "：CIM 0.16、ランダム 0.49"],
                                 "・SA 後：CIM 開始 0.14、ランダム開始 0.28"], 16)
    box(s, 1.0, 5.85, 11.3, 1.3, None, kind="round", lw=1.5)
    text(s, 1.1, 5.9, 11.1, 1.2, [
        [("長い SA で同じ総時間にすると逆転する理由", "B")],
        ["CIM 開始は 1 本のばらつきが小さく（標準偏差 10 vs 19 カット）最良値が伸びにくい"],
        ["＋ ランダム開始は同じ時間で 8 倍の本数（134 本 vs 16 本）を回せる"]], 16, "c")
    return s


def conclusion(prs):
    s = new_slide(prs, "結論と今後の予定")
    text(s, 1.1, 1.05, 6.7, 0.5, ["結論"], 24)
    text(s, 1.1, 1.6, 11.3, 2.9, [
        ["CIM の解を SA の初期解にすると、", ("SA 1本あたりの解の品質が向上する", "r"), "（G22、5000 スイープでも）。"],
        "",
        ["CIM の時間を含めた同じ総時間では、", ("短い SA（500 スイープ）では有利", "r"),
         "、長い SA（5000 スイープ）では逆転する。"],
        "",
        ["CIM の解は最良解の近く（谷）に集まり、これが「カット値以外の良さ」の候補である。"],
        "",
        ["GA の初期集団に使うと、G70 では同じ総時間でも有意に改善した。"]], 18)
    text(s, 1.1, 4.65, 6.7, 0.5, ["今後の予定"], 24)
    text(s, 1.1, 5.2, 11.3, 2.0, [
        "CIM の周回数と、CIM の解の分布（ばらつき）の関係を明らかにする。",
        "最良解の近くに集まる性質が、他の問題（G55・G70）でも成り立つかを明らかにする。",
        "比較の公平性（SA の開始温度の調整など）を確認し、グラフの規模・密度への依存性を明らかにする。"], 18, space=6)
    return s


def schedule(prs):
    s = new_slide(prs, "研究計画")
    rows = [["月", "予定"], ["11月", "中間報告会・CIM の周回数と解の分布の関係"], ["12月", "比較の公平性の確認・他の問題（G55・G70）での検証"],
            ["1月", "問題の拡張（グラフの規模・密度）・卒論執筆開始"], ["2月", "卒論提出及び卒論発表（2/16・17）"]]
    gf = s.shapes.add_table(5, 2, Inches(1.6), Inches(1.8), Inches(10.1), Inches(3.4))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    tbl = gf.table
    tbl.columns[0].width = Inches(1.9)
    tbl.columns[1].width = Inches(8.2)
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 18, "c" if j == 0 else "l")
    return s


# ---------------- 付録 ----------------

def appendix_params(prs):
    s = new_slide(prs, "付録:実験パラメータ")
    data = [["", "G22", "G55", "G70"], ["a（ループ損失）", "0.303", "0.312", "0.084"], ["b（結合）", "−0.049", "−0.130", "−0.153"],
            ["c（雑音）", "7.0×10⁻⁶", "5.6×10⁻⁶", "5.7×10⁻⁶"], ["α（利得）", "7.06", "3.90", "12.5"],
            ["β（利得飽和）", "7.21", "15.1", "0.79"], ["ΔP（1周回の増分）", "1.6×10⁻⁵", "2.5×10⁻⁵", "8.8×10⁻⁶"]]
    text(s, 0.5, 1.0, 6.2, 0.4, ["CIM（Optuna で CIM 単体のカット値の平均を最大化）"], 16)
    gf = s.shapes.add_table(len(data), 4, Inches(0.5), Inches(1.45), Inches(6.2), Inches(2.9))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    tbl = gf.table
    tbl.columns[0].width = Inches(2.6)
    for j in (1, 2, 3):
        tbl.columns[j].width = Inches(1.2)
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 14, "c")
    text(s, 0.5, 4.5, 6.2, 2.6, [
        ["・", ("p", "v"), "(", ("n", "v"), ") = (", ("n", "v"), " Δ", ("P", "v"), ")", ("1/2", "p"), "、", ("n", "v"), ("0", "s"), " は標準正規乱数"],
        "・周回数：SA の実験 2400、GA の実験 4800（G55 は 9600）",
        "・SA：1 スピンを選んで反転を試み、メトロポリス法で受理。温度は 1 反転ごとに指数的に下げる（1 スイープ ＝ n 回の反転）",
        "・SA の開始温度は両者共通で調整なし"], 14, space=3)
    text(s, 7.0, 1.0, 6.0, 0.4, ["GA（文献 [7] のメメティック GA）"], 16)
    text(s, 7.0, 1.45, 6.0, 2.0, [
        "・親 2 個をランダムに選び、一致する頂点を残す交叉",
        "・子と初期個体にタブー探索（1 頂点反転）。改善がないと 150 頂点を摂動",
        "・置き換え：カット値と距離の重み付けで最も悪い個体を除く",
        "・パラメータはランダム初期化で調整（ランダム開始側に有利な条件）",
        "・集団サイズ N = 10、各 16 試行"], 14, space=3)
    data2 = [["", "G22", "G55", "G70"], ["タブー探索の反復", "35238", "58453", "41152"], ["摂動までの反復", "2159", "930", "521"],
             ["タブー期間の係数", "40", "13", "11"], ["カット値の重み", "0.84", "0.43", "0.66"]]
    gf = s.shapes.add_table(len(data2), 4, Inches(7.0), Inches(4.0), Inches(6.0), Inches(2.1))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = "{5940675A-B579-460E-94D1-54222C63F5DA}"
    tbl = gf.table
    tbl.columns[0].width = Inches(2.4)
    for j in (1, 2, 3):
        tbl.columns[j].width = Inches(1.2)
    for i, r in enumerate(data2):
        for j, v in enumerate(r):
            c = tbl.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 14, "c")
    return s


def references(prs):
    s = new_slide(prs, "参考文献")
    refs = ["[1] T. Inagaki et al., Science, vol. 354, pp. 603–606, 2016.",
            "[2] T. Honjo et al., Sci. Adv., vol. 7, eabh0952, 2021.",
            "[3] R. Hamerly et al., Sci. Adv., vol. 5, eaau0823, 2019.",
            "[4] Y. Haribara et al., arXiv:1501.07030, 2015.",
            "[5] F. Ma and J.-K. Hao, Ann. Oper. Res., vol. 248, pp. 365–403, 2017.",
            "[6] C. Chen et al., arXiv:2307.00783, 2023.",
            "[7] Q. Wu and J.-K. Hao, Lect. Notes Comput. Sci., vol. 7492 (PPSN XII), pp. 297–306, 2012."]
    text(s, 0.8, 1.2, 11.8, 5.0, refs, 18, space=8)
    return s


def main():
    if DST.exists():
        raise FileExistsError(DST)
    prs = Presentation(str(SRC))
    orig = list(prs.slides)
    keep = {0: "cover", 1: None, 3: None, 4: None, 5: None}
    cover(orig[0])
    set_title(orig[1], "背景:最適化問題の分類と例")

    S = []
    S.append(orig[0])
    S.append(orig[1])                     # 最適化問題の分類と例（佐々木さんの図）
    S.append(maxcut(prs))
    S.append(orig[3])                     # イジングモデルとマッピング（佐々木さんの図）
    S.append(orig[4])                     # CIM によるイジングモデルの実装（佐々木さんの図）
    S.append(orig[5])                     # CIM によるエネルギー最小状態への到達方法（佐々木さんの図）
    S.append(sim_model(prs, orig[6]))
    S.append(prior(prs))
    S.append(sa(prs))
    S.append(ga(prs))
    S.append(init_sol(prs))
    S.append(purpose(prs))
    S.append(method(prs))
    S.append(evaluation(prs))
    S.append(result(prs, "結果:SA 1本あたりの到達点（G22）", "s17_13.png", 11.0, [
        "・SA 5000 スイープ後も、CIM 開始がランダム開始より良い（gap 0.10% vs 0.22%、p < 0.001）",
        "・点線は SA 前の CIM の解（上位 16 個の平均 0.22%）：ランダム開始で SA を 5000 スイープ回した後とほぼ同じ品質",
        "・CIM の独立な 100 組（各 16 本）で比較（Mann–Whitney 検定）"],
        [["良い地点から始めると、長い SA でも", ("1本あたりの到達点は良くなる", "r")]]))
    S.append(result(prs, "結果:開始のカット値をそろえたSA（G22）", "s18_13.png", 9.6, [
        "・SA の解（ランダム開始の SA で得た解）から、CIM の解と同じカット値のものを選んで開始",
        "・カット値の良い解から始めると良くなる（ランダム 0.22% → SA の解 0.13%）",
        "・同じカット値でも、CIM の解から始めた方が良い（0.10%、p < 0.001）"],
        [["CIM の解には、", ("カット値以外の良さ", "r"), "がある"]]))
    S.append(result(prs, "結果:最良解（BKS解）との距離（G22）", "s19_13.png", 9.0, [
        ["・距離 ", ("d", "v"), "：値が違うスピンの割合（0 で同じ解、0.5 で無関係な解）"],
        "・ランダムな解は最良解と無関係（0.49）。CIM の解は 0.16 で、ランダムの約 1/3",
        "・SA 後、CIM 開始は 0.14、ランダム開始は 0.28（gap 0.22% でも 0.28 離れる → 別の谷にも良い解がある）",
        "・最良解を 88 個集めても 1 つの谷に集まり、距離はほぼ同じ"],
        [["CIM は", ("最良解に近い領域", "r"), "に解を出し、SA はその近くを探して仕上げる"]]))
    S.append(result(prs, "結果:同じ総時間での比較（G22）", "s20_13.png", 9.4, [
        "・500 スイープ：CIM 開始が有利（ランダム開始 1156 本 vs CIM 開始 16 本、p < 0.001）",
        "・5000 スイープ：ランダム開始が有利（134 本 vs 16 本、p < 0.001）。ただし差は平均 0.7 カット",
        "・総時間の 99%（500）・88%（5000）は CIM の時間"],
        [["同じ総時間では、", ("短い SA では CIM 開始が有利", "r"), "、長い SA では逆転する"]]))
    S.append(result(prs, "結果:GAの初期集団にCIMの解", "s22_13.png", 10.0, [
        "・集団サイズ N = 10。CIM：CIM 10 回の解をそのまま使う。CIM + cut / + Hamming：40 回から 10 個を選別",
        "・CIM の時間も含めた同じ総時間で比較",
        "・G70：CIM の初期集団が有意に良い（Holm 補正後 p < 0.001）。G55：p = 0.07、G22：差なし"],
        [["GA では", ("G70 で CIM 初期化の効果", "r"), "が見られた（1 問題での結果）"]]))
    S.append(discussion(prs))
    S.append(conclusion(prs))
    S.append(schedule(prs))
    S.append(result(prs, "付録:CIMの実行回数を減らす", "s23_13.png", 10.0, [
        "・CIM の実行回数 K を 256 → 16 に減らし、上位 16 個を SA に渡す（SA 5000 スイープ）",
        "・K = 16 では上位を選べず CIM 開始 1 本の質も下がる → G22 はなおランダム開始が有利",
        "・500 スイープでは K = 16〜256 のすべてで、3 問題とも CIM 開始が有利"],
        [["長い SA での逆転は、CIM の回数だけでは説明できない"]]))
    S.append(result(prs, "付録:1本の平均と最良値（G22）", "s21_13.png", 11.0, [
        "・1 本の平均では CIM 開始が 2.3 倍良いが、ばらつきが半分（標準偏差 10 vs 19 カット）で最良値が伸びにくい",
        "・同じ総時間ではランダム開始が 8 倍の本数（134 本）を回せるので逆転"],
        [["逆転＝「最良値の伸びにくさ」＋「本数の差」。1本あたりの利点は残る"]]))
    S.append(result(prs, "付録:CIMの解256個からどれを選ぶか", "s24_13.png", 8.8, [
        "・Random pick：ランダムに 16 個　Top by cut：カット値の良い順に 16 個　Cut + Hamming：上位 32 個から互いに離れるよう 16 個",
        "・カット値上位が最も良い（G55 は同点）。距離で選んでも改善しない（CIM の解どうしが近い：0.13）"],
        [["短い探索（500 スイープ）では多様性は効かなかった"]]))
    S.append(appendix_params(prs))
    S.append(references(prs))

    # 元の資料の残りのスライドを消し、並べ替える
    lst = prs.slides._sldIdLst
    by_part = {prs.part.related_part(sid.rId): sid for sid in lst}
    keep_parts = {sl.part for sl in S}
    for part, sid in list(by_part.items()):
        if part not in keep_parts:
            prs.part.drop_rel(sid.rId)
            lst.remove(sid)
    for sl in S:
        sid = by_part[sl.part]
        lst.remove(sid)
        lst.append(sid)
    for sl in S[2:]:
        renumber(sl) if sl not in orig else None
    prs.save(str(DST))
    print("saved", DST, len(S))


if __name__ == "__main__":
    main()
