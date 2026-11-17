"""v45 の背景ページ：先輩の図を使わず、すべて自前の図で描き直す（より詳しく）。"""
import math

from pptx.dml.color import RGBColor

import build_v38 as b
from lib38 import (BLACK, BLUE, GRAY, RED, WHITE, ball, block_arrow, box, line, picture, polyline, spin,
                   text)

PINK = RGBColor(0xFF, 0xE5, 0xE5)
SKY = RGBColor(0xE5, 0xEE, 0xFF)


def axes(s, x0, y0, w, h, xlabel, ylabel, size=14):
    line(s, x0, y0 + h, x0 + w, y0 + h, lw=1.5, arrow=True)
    line(s, x0, y0 + h, x0, y0, lw=1.5, arrow=True)
    text(s, x0, y0 + h + 0.02, w, 0.35, [xlabel], size, "c")
    tb = text(s, x0 - 0.45, y0, 0.4, h, [ylabel], size, "c", "m")
    tb.text_frame._bodyPr.set("vert", "eaVert")


# ---------------- 組合せ最適化問題とその難しさ ----------------

def optim_problem(prs):
    s = b.new_slide(prs, "背景:組合せ最適化問題とその難しさ")
    # 左：連続最適化
    text(s, 0.4, 0.95, 6.0, 0.4, [("連続最適化", "B")], 18, "c")
    x0, y0, w, h = 1.0, 1.45, 5.0, 2.2
    axes(s, x0, y0, w, h, "解（連続な値）", "目的関数")
    pts = [(x0 + 0.3 + k / 40 * (w - 0.6), y0 + 0.2 + (h - 0.4) * (1 - ((k / 40 - 0.55) / 0.55) ** 2)) for k in range(41)]
    polyline(s, [(px, min(py, y0 + h - 0.1)) for px, py in pts], lw=2.5)
    for k in (6, 12, 18):
        (ax_, ay_), (bx_, by_) = pts[k], pts[k + 4]
        line(s, ax_, ay_ - 0.15, bx_, by_ - 0.15, color=RED, lw=2.0, arrow=True)
    cx, cy = pts[22]
    box(s, cx - 0.12, cy - 0.24, 0.24, 0.24, kind="oval", fill=RED, line=None)
    text(s, 0.6, 3.95, 5.8, 0.4, [["傾き（勾配）をたどれば最小へ進める"]], 16, "c")
    # 右：組合せ最適化
    text(s, 6.9, 0.95, 6.0, 0.4, [("組合せ最適化（マックスカットなど）", "B")], 18, "c")
    x0, y0 = 7.5, 1.45
    axes(s, x0, y0, w, h, "解（スピンの配置：飛び飛び）", "目的関数")
    heights = [0.55, 0.2, 0.75, 0.4, 0.9, 0.3, 0.65, 0.1, 0.5, 0.8, 0.35, 0.6, 0.15, 0.7, 0.45, 0.85, 0.25, 0.95]
    for k, v in enumerate(heights):
        px = x0 + 0.3 + k * (w - 0.6) / (len(heights) - 1)
        py = y0 + 0.2 + (1 - v) * (h - 0.45)
        best = v == min(heights)
        box(s, px - 0.08, py - 0.08, 0.16, 0.16, kind="oval", fill=RED if best else BLACK, line=None)
    text(s, 10.3, 1.0 + 2.05, 2.6, 0.35, [("最良解はどれか？", "r")], 14, "r")
    text(s, 7.1, 3.95, 5.8, 0.4, [["傾きが使えない → 候補を調べて比べるしかない"]], 16, "c")
    # 下：候補の数と全部調べる時間
    text(s, 0.4, 4.45, 12.5, 0.4, [["頂点数 ", ("N", "v"), " のマックスカットの候補は 2", ("N", "vp"),
                                     " 通り（1 秒に 10 億個調べた場合）"]], 18, "c")
    rows = [("N = 10", "1024", "約 1 マイクロ秒"), ("N = 50", "約 1.1 × 10¹⁵", "約 13 日"),
            ("N = 100", "約 1.3 × 10³⁰", "約 4 × 10¹³ 年"), ("N = 2000（本研究の G22）", "約 10⁶⁰²", "不可能")]
    for k, (n, num, t) in enumerate(rows):
        y = 4.95 + k * 0.38
        hot = k == 3
        box(s, 1.6, y, 3.6, 0.36, [n], 14, fill=PINK if hot else WHITE, lw=1.0)
        box(s, 5.2, y, 3.2, 0.36, [num], 14, fill=PINK if hot else WHITE, lw=1.0)
        box(s, 8.4, y, 3.3, 0.36, [(t, "r") if hot else t], 14, fill=PINK if hot else WHITE, lw=1.0)
    text(s, 0.5, 6.6, 12.3, 0.6, [["厳密に解くのは現実的でない → ", ("限られた時間で良い近似解", "r"), "を得る手法が必要"]], 20, "c")
    return s


# ---------------- マックスカットとイジングモデル ----------------

def ising_map(prs):
    s = b.new_slide(prs, "背景:マックスカットとイジングモデル")
    text(s, 0.4, 0.95, 12.5, 0.45, ["頂点をスピン、辺を結合とみなすと、マックスカットは「エネルギー最小の状態を探す問題」になる"], 18, "c")
    pts = {0: (1.2, 2.0), 1: (3.2, 1.75), 2: (5.0, 2.4), 3: (1.6, 3.9), 4: (3.6, 3.6), 5: (5.2, 4.3)}
    up = {0: True, 1: False, 2: True, 3: False, 4: True, 5: False}
    edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5), (0, 4), (2, 4)]
    for i, j in edges:
        cut = up[i] != up[j]
        line(s, *pts[i], *pts[j], color=RED if cut else GRAY, lw=3.0 if cut else 1.5, dash=not cut)
    for k, (cx, cy) in pts.items():
        box(s, cx - 0.33, cy - 0.33, 0.66, 0.66, kind="oval", fill=SKY if up[k] else PINK, lw=1.5)
        spin(s, cx - 0.1, cy - 0.2, up[k], w=0.2, h=0.4)
    text(s, 0.5, 4.75, 5.6, 0.75, [[("赤の実線", "r"), "：向きが逆のスピンを結ぶ辺 ＝ 切断される辺"],
                                   "灰の破線：向きが同じ（切断されない）"], 14, "c")
    # 右：式と対応
    picture(s, "s04_14.png", 7.3, 1.55, w=4.9)
    pairs = [("頂点 i", [("スピン ", ""), ("σ", "v"), ("i", "vs"), " = ±1"]), ("グループ（2 つ）", "向き（↑ / ↓）"),
             ([("辺の重み ", ""), ("w", "v"), ("ij", "vs")], [("結合 ", ""), ("J", "v"), ("ij", "vs"), " = −", ("w", "v"), ("ij", "vs")]),
             ("カット値が最大", [("エネルギー ", ""), ("H", "v"), (" が最小", "")])]
    text(s, 6.6, 2.35, 2.9, 0.35, [("マックスカット", "B")], 16, "c")
    text(s, 10.1, 2.35, 2.9, 0.35, [("イジングモデル", "B")], 16, "c")
    for k, (l_, r_) in enumerate(pairs):
        y = 2.75 + k * 0.6
        hot = k == 3
        box(s, 6.6, y, 2.9, 0.5, [l_ if isinstance(l_, list) else l_], 15, line=RED if hot else BLACK,
            lw=2.0 if hot else 1.2)
        line(s, 9.6, y + 0.25, 10.0, y + 0.25, lw=1.5, arrow=True)
        line(s, 10.0, y + 0.25, 9.6, y + 0.25, lw=1.5, arrow=True)
        box(s, 10.1, y, 2.9, 0.5, [r_ if isinstance(r_, list) else r_], 15, line=RED if hot else BLACK,
            lw=2.0 if hot else 1.2)
    # 下：1 本の辺のエネルギー
    text(s, 0.4, 5.55, 12.5, 0.4, ["1 本の辺（重み 1）で見ると"], 16, "c")
    for k, (u1, u2, lab, e) in enumerate([(True, False, "向きが逆：切断される", "エネルギー低い"),
                                          (True, True, "向きが同じ：切断されない", "エネルギー高い")]):
        x = 0.7 + k * 6.2
        spin(s, x, 5.95, u1, w=0.22, h=0.42)
        line(s, x + 0.3, 6.16, x + 0.9, 6.16, color=RED if k == 0 else GRAY, lw=2.5, dash=k == 1)
        spin(s, x + 0.98, 5.95, u2, w=0.22, h=0.42)
        text(s, x + 1.35, 5.98, 4.7, 0.4, [[lab, " → ", (e, "r" if k == 0 else "")]], 14)
    text(s, 0.5, 6.65, 12.3, 0.55, [["問題を結合 ", ("J", "v"), " に書き込み、物理系に ", ("H", "v"),
                                      " の小さい状態を探させる装置 → ", ("イジングマシン", "r")]], 18, "c")
    return s


# ---------------- CIM の構成 ----------------

def cim_setup(prs):
    s = b.new_slide(prs, "背景:コヒーレントイジングマシン（CIM）の構成")
    text(s, 0.4, 0.95, 12.5, 0.45, ["測定フィードバック型 CIM：光ファイバの輪を光パルスが周回し、パルス 1 個がスピン 1 個を表す"], 18, "c")
    x0, y0, x1, y1 = 0.7, 2.2, 7.9, 4.55
    box(s, x0, y0, x1 - x0, y1 - y0, None, kind="round", line=GRAY, lw=5.0)
    # 位相感応増幅器とポンプ
    box(s, 2.0, y0 - 0.35, 2.4, 0.7, ["位相感応増幅器", "（PSA）"], 14, fill=WHITE, lw=1.5)
    line(s, 3.2, 1.45, 3.2, y0 - 0.35, color=RED, lw=2.5, arrow=True)
    text(s, 3.3, 1.38, 1.6, 0.35, [("ポンプ光", "r")], 14)
    # パルス列（下辺）
    signs = [1, -1, 1, 1, -1, 1, -1, -1, 1, -1]
    for k, sg in enumerate(signs):
        cx = 1.3 + k * 0.6
        box(s, cx - 0.2, y1 - 0.2, 0.4, 0.4, [("+" if sg > 0 else "−", "w")], 14, kind="oval",
            fill=RED if sg > 0 else BLUE, line=None)
        text(s, cx - 0.3, y1 + 0.22, 0.6, 0.3, [[("σ", "v"), (str(k + 1), "vs")]], 12, "c")
    line(s, 3.5, y1 + 0.62, 5.2, y1 + 0.62, lw=1.5, arrow=True)
    text(s, 5.25, y1 + 0.45, 2.6, 0.35, ["周回の向き"], 14)
    text(s, 1.0, 2.9, 6.6, 1.0, ["N 個のパルスが時間をずらして", "同じ輪を回る（時分割多重）"], 15, "c")
    # 右側：測定 → 計算 → 注入
    box(s, x1 - 0.12, 2.5, 0.24, 0.5, None, fill=BLACK, line=None)
    box(s, x1 - 0.12, 3.6, 0.24, 0.5, None, fill=BLACK, line=None)
    box(s, 8.75, 1.75, 4.2, 0.85, ["ホモダイン測定", [("各パルスの振幅 ", ""), ("x", "v"), ("i", "vs"), " を測る"]], 14)
    box(s, 8.75, 2.95, 4.2, 0.9, ["計算機（FPGA）", [("結合の和 Σ", ""), ("J", "v"), ("ij", "vs"), ("x", "v"), ("j", "vs"), " を計算"]], 14,
        line=RED, lw=2.0)
    box(s, 8.75, 4.2, 4.2, 0.75, ["変調器", "計算結果の光を作る"], 14)
    line(s, x1 + 0.12, 2.75, 8.75, 2.3, lw=2.0, arrow=True)
    text(s, 7.85, 2.05, 0.95, 0.35, ["一部を取出"], 11, "c")
    line(s, 10.85, 2.6, 10.85, 2.95, lw=2.0, arrow=True)
    line(s, 10.85, 3.85, 10.85, 4.2, lw=2.0, arrow=True)
    polyline(s, [(8.75, 4.58), (8.35, 4.58), (8.35, 3.85), (x1 + 0.12, 3.85)], lw=2.0, arrow=True)
    text(s, 7.95, 4.1, 0.9, 0.35, ["注入"], 11, "c")
    # 下：対応表
    items = [["光パルス 1 個 ＝ スピン 1 個"], [("位相 0 / π ＝ ", ""), ("σ", "v"), " = +1 / −1"],
             [("測定と注入で結合 ", ""), ("J", "v"), " を再現"]]
    for k, it in enumerate(items):
        box(s, 0.6 + k * 4.15, 5.45, 3.95, 0.6, [it], 15, kind="round", lw=1.5)
    block_arrow(s, 6.07, 6.15, 1.2, 0.28)
    text(s, 0.5, 6.5, 12.3, 0.6, [["全スピンを光で", ("並列に", "r"), "更新し、短時間で", ("良い解", "r"), "を出力できる"]], 20, "c")
    return s


# ---------------- 分岐で解を選ぶ仕組み ----------------

def cim_bifurcation(prs):
    s = b.new_slide(prs, "背景:CIMが解を選ぶ仕組み（分岐）")
    text(s, 0.4, 0.95, 12.5, 0.45, ["ポンプ光を少しずつ強くすると、しきい値で振幅が + か − の 2 つに分かれる（分岐）"], 18, "c")
    # 左：分岐図
    x0, y0, w, h = 1.1, 2.45, 5.2, 2.9
    axes(s, x0, y0, w, h, "ポンプ光の強さ", "パルスの振幅")
    ymid = y0 + h / 2
    th = x0 + 2.0
    line(s, x0, ymid, th, ymid, lw=2.5)
    line(s, th, ymid, x0 + w - 0.2, ymid, color=GRAY, lw=1.5, dash=True)
    up = [(th + t * (w - 2.3), ymid - 1.15 * math.sqrt(t)) for t in (k / 30 for k in range(31))]
    dn = [(px, 2 * ymid - py) for px, py in up]
    polyline(s, up, color=RED, lw=2.5)
    polyline(s, dn, color=BLUE, lw=2.5)
    line(s, th, ymid - 0.1, th, y0 + h, color=GRAY, lw=1.0, dash=True)
    text(s, th - 1.7, y0 + h - 0.38, 1.6, 0.35, ["しきい値"], 13, "r")
    text(s, x0 + w - 1.6, y0 - 0.05, 1.7, 0.35, [[("+（", "r"), ("σ", "vr"), (" = +1）", "r")]], 13)
    text(s, th + 0.2, y0 + h - 0.38, 1.7, 0.35, [[("−（", "b"), ("σ", "vb"), (" = −1）", "b")]], 13)
    # 上：ポテンシャルの形（しきい値の前後）
    for k, (cxp, kind) in enumerate(((x0 + 0.95, "one"), (x0 + 3.7, "two"))):
        pts = []
        for i in range(31):
            u = -1 + i / 15
            v = u * u if kind == "one" else (u ** 4 - u ** 2) * 2.2 + 0.55
            pts.append((cxp + u * 0.6, 1.5 + (1 - v) * 0.5))
        polyline(s, pts, lw=1.8)
        if kind == "one":
            box(s, cxp - 0.07, 2.0 - 0.14, 0.14, 0.14, kind="oval", fill=GRAY, line=None)
        else:
            for sx in (-1, 1):
                px = cxp + sx * 0.6 * 0.707
                box(s, px - 0.07, 2.0 - 0.14, 0.14, 0.14, kind="oval", fill=RED if sx > 0 else BLUE, line=None)
    line(s, x0 + 1.75, 1.75, x0 + 2.85, 1.75, lw=1.5, arrow=True)
    text(s, x0 + 0.1, 2.12, 1.9, 0.3, ["安定点 1 つ"], 12, "c")
    text(s, x0 + 2.85, 2.12, 1.9, 0.3, ["安定点 2 つ（±）"], 12, "c")
    # 右：本研究のシミュレーションの実データ
    picture(s, "s06_15.png", 6.9, 1.45, w=6.1)
    text(s, 6.9, 4.75, 6.1, 0.6, ["本研究のシミュレーション（G22、一部のパルス）", "しきい値を越えると、各パルスが + か − に分かれる"], 14, "c")
    block_arrow(s, 6.07, 5.6, 1.2, 0.3)
    text(s, 0.5, 5.95, 12.3, 1.1, [["どちらに分かれるかは、", ("雑音", "r"), "と", ("結合 ", "r"), ("J", "vr"),
                                     "（他のパルスからの注入）で決まる"],
                                    ["→ 全パルスが結合に合う向き（", ("エネルギーの低い状態＝良いカット", "r"), "）にそろいやすい"]], 18, "c")
    return s


# ---------------- 本研究のシミュレーションモデル ----------------

def sim_model(prs):
    s = b.new_slide(prs, "背景:本研究で用いるCIMのシミュレーションモデル")
    text(s, 0.4, 0.95, 12.5, 0.4, ["1 周回ごとに、全パルスの振幅をまとめて更新する（測定フィードバック型 CIM の模擬）"], 18, "c")
    picture(s, "s06_17.png", 0.9, 1.5, w=4.9)
    picture(s, "s06_18.png", 6.9, 1.45, w=5.3)

    def vbox(x, y, w, h, top, sub, col=BLACK, fill=None):
        return box(s, x, y, w, h, [top, sub], 14, line=col, lw=1.8 if col == RED else 1.3, fill=fill)

    E1 = [("E", "vB"), ("(", ""), ("n", "v"), (" − 1)", "")]
    vbox(0.4, 3.35, 1.9, 0.9, E1, "前の周回の振幅")
    vbox(2.9, 2.35, 3.0, 0.75, [("a", "v")] + E1, "ループの損失")
    vbox(2.9, 3.42, 3.0, 0.75, [("b", "v"), ("J", "vB")] + E1, "測定フィードバック（結合）", col=RED)
    vbox(2.9, 4.5, 3.0, 0.75, [("c", "v"), ("n", "vB"), ("0", "s")], "雑音（毎周回の乱数）")
    line(s, 2.3, 3.8, 2.9, 2.72, lw=1.5, arrow=True)
    line(s, 2.3, 3.8, 2.9, 3.8, lw=1.5, arrow=True)
    box(s, 6.35, 3.55, 0.5, 0.5, [("+", "B")], 18, kind="oval", lw=1.5)
    for yy in (2.72, 3.8, 4.87):
        line(s, 5.9, yy, 6.4, 3.8 + (yy - 3.8) * 0.25, lw=1.5, arrow=True)
    line(s, 6.85, 3.8, 7.25, 3.8, lw=1.5, arrow=True)
    vbox(7.25, 3.35, 1.1, 0.9, [("F", "vB"), ("(", ""), ("n", "v"), (")", "")], "")
    line(s, 8.35, 3.8, 8.7, 3.8, lw=1.5, arrow=True)
    box(s, 8.7, 3.2, 3.15, 1.2, [[("× exp[", ""), ("α p", "v"), ("(", ""), ("n", "v"), (")(1 − ", ""), ("β", "v"), ("|", ""),
                                   ("F", "vB"), ("|", ""), ("2", "p"), (")]", "")], "増幅（ポンプ）と飽和"], 14, lw=1.3)
    line(s, 11.85, 3.8, 12.2, 3.8, lw=1.5, arrow=True)
    vbox(12.2, 3.35, 0.95, 0.9, [("E", "vB"), ("(", ""), ("n", "v"), (")", "")], "")
    polyline(s, [(12.67, 4.25), (12.67, 5.6), (1.35, 5.6), (1.35, 4.25)], lw=1.8, arrow=True)
    text(s, 4.5, 5.62, 4.5, 0.35, [["次の周回へ（", ("n", "v"), " → ", ("n", "v"), " + 1）"]], 14, "c")
    text(s, 8.7, 2.6, 3.4, 0.55, [[("p", "v"), "(", ("n", "v"), ") を周回ごとに少しずつ増やす"]], 13, "c")
    text(s, 0.5, 6.05, 12.3, 1.1, [
        ["最後に振幅の符号（+ / −）をスピンとして読み出し、カット値を計算する"],
        [("本研究では CIM を CPU 上で模擬し、", ""), ("出力解をそのまま SA・GA の初期解にする", "r")]], 18, "c")
    return s
