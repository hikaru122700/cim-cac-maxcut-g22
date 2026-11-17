"""卒論中間報告 v39：v38 から、結果を「開始のカット値をそろえた SA（G22）」の 1 つだけにする。
手法・目的・考察・結論もその結果に合わせ、ページ番号を右下に移す。"""
import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).parent))
import build_v38 as b  # noqa: E402
from lib38 import (BLACK, GRAY, RED, WHITE, ball, block_arrow, box, fill_text, landscape, line,  # noqa: E402
                   local_min, renumber, text)

SRC = b.SRC
DST = Path(os.environ.get("V39_OUT", "C:/Users/owner/Downloads/卒論中間報告_三輪光_v39.pptx"))
GRID = "{5940675A-B579-460E-94D1-54222C63F5DA}"


def page_number_bottom_right(prs):
    """先輩のレイアウトは右上にページ番号の欄がある。右下へ移し、右寄せにする。"""
    for layout in prs.slide_layouts:
        for sh in layout.shapes:
            if sh.has_text_frame and 'type="slidenum"' in sh._element.xml and sh.top < Inches(1.5):
                sh.left, sh.top = Inches(12.25), Inches(7.0)
                sh.width, sh.height = Inches(0.95), Inches(0.4)
                sh.text_frame.word_wrap = True
                sh.text_frame._bodyPr.set("wrap", "square")
                for p in sh.text_frame.paragraphs:
                    p.alignment = PP_ALIGN.RIGHT


def purpose(prs):
    s = b.new_slide(prs, "本研究の目的")
    box(s, 1.09, 1.9, 11.15, 1.9, [["コヒーレントイジングマシンの出力解をSA・GAの初期解とすることで、"
                                    "限られた計算時間内に得られる解の品質が改善するかを明らかにする。"]], 24,
        kind="round", lw=3.0)
    text(s, 1.6, 4.4, 10.5, 1.8, [
        "・CIM の解を初期解にしたときの、SA の到達点を明らかにする。",
        [("・CIM の解の初期解としての良さが、カット値だけで決まるのかを明らかにする。", "r")],
        "・CIM の計算時間を含めた同じ総時間で、ランダムな初期解と比べる。",
        "・効果がグラフ（規模・密度）にどう依存するかを明らかにする。"], 18, space=6)
    return s


def method(prs):
    s = b.new_slide(prs, "手法:開始のカット値をそろえた比較")
    text(s, 0.4, 0.95, 12.5, 0.45, [["開始解の", ("カット値をそろえて", "r"), "SA を回し、カット値以外の違いが到達点に表れるかを調べる"]], 18, "c")
    box(s, 4.15, 1.5, 5.0, 0.6, ["マックスカット問題（G22）"], 18)
    lanes = [(0.55, "CIM の解", ["CIM を 256 回実行", "（2400 周回）"], ["カット値の上位", "16 個を選ぶ"], RED),
             (4.65, "SA の解", ["ランダム開始の SA", "で解を作る"], ["CIM の解と同じ", "カット値の解を選ぶ"], BLACK),
             (8.75, "ランダムな解", ["ランダムな解", "を作る"], None, BLACK)]
    for x, title, b1, b2, col in lanes:
        text(s, x, 2.3, 4.0, 0.4, [(title, "r" if col == RED else "")], 18, "c")
        line(s, 6.65, 2.1, x + 2.0, 2.35, lw=1.5, arrow=True)
        box(s, x + 0.5, 2.75, 3.0, 0.85, b1, 18, line=col, lw=2.0)
        line(s, x + 2.0, 3.6, x + 2.0, 3.95, lw=2.0, arrow=True)
        if b2:
            box(s, x + 0.5, 3.95, 3.0, 0.85, b2, 18)
        else:
            line(s, x + 2.0, 3.95, x + 2.0, 4.8, lw=2.0)
        line(s, x + 2.0, 4.8, x + 2.0, 5.15, lw=2.0, arrow=True)
        box(s, x + 0.5, 5.15, 3.0, 0.85, ["SA", "（5000 スイープ）"], 18)
        line(s, x + 2.0, 6.0, x + 2.0, 6.3, lw=2.0, arrow=True)
    text(s, 4.55, 4.85, 4.2, 0.35, [("開始のカット値が同じ", "r")], 14, "c")
    box(s, 1.5, 6.3, 10.3, 0.6, ["SA 後のカット値の比較（既知の最良値 BKS との差 gap）"], 18)
    return s


def evaluation(prs):
    s = b.new_slide(prs, "手法:評価方法と条件")
    text(s, 0.4, 0.95, 6.0, 0.4, ["評価指標：既知の最良値（BKS）との差"], 18)
    box(s, 0.9, 1.45, 4.6, 0.85, None, kind="rect", lw=1.5)
    b.picture(s, "s16_17.png", 1.25, 1.6, w=3.9)
    text(s, 0.4, 2.4, 6.0, 0.45, ["小さいほど良い（0% で既知の最良値）"], 16, "c")
    text(s, 0.4, 3.1, 6.0, 0.4, ["ベンチマーク問題（G-set）"], 18)
    rows = [["問題", "頂点数", "辺数", "BKS"], ["G22", "2000", "19990", "13359"]]
    gf = s.shapes.add_table(2, 4, Inches(0.8), Inches(3.6), Inches(5.0), Inches(0.8))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = GRID
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            c = gf.table.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = b.LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 16, "c")
    text(s, 0.4, 4.5, 6.0, 0.45, ["BKS：文献 [5] の既知の最良値"], 16, "c")
    text(s, 6.6, 0.95, 6.5, 0.4, ["条件"], 18)
    text(s, 6.6, 1.45, 6.5, 4.5, [
        ["・SA：温度 ", ("T", "v"), ("0", "vs"), " = 2.0 → 0.001、5000 スイープ（3 者で同じ）"],
        "・CIM：2400 周回を 256 回、カット値の上位 16 個を使う",
        "・これを独立に 100 組（計 1600 本の SA）",
        "・SA の解：ランダム開始の SA で得た解から、CIM の解 1 つずつに同じカット値の解を選ぶ",
        "・そろえたカット値：13317〜13353（平均 13330）",
        "・検定：16 本の平均どうしを 100 組で比べる（Wilcoxon 検定）"], 16, space=6)
    text(s, 0.5, 6.4, 12.3, 0.8, [["開始のカット値は同じなので、差が出れば", ("カット値以外の違い", "r"), "による"]], 18, "c")
    return s


def discussion(prs):
    s = b.new_slide(prs, "考察:CIMの解の「カット値以外の良さ」")
    pos = landscape(s, 1.0, 1.4, 5.8, 3.4)
    u2 = local_min(0.68)
    for u in (0.63, 0.66, 0.71, 0.73):
        ball(s, pos, u, d=0.2)
    for u in (0.32, 0.88):
        ball(s, pos, u, d=0.2, color=GRAY)
    cx, cy = pos(u2)
    box(s, cx - 0.75, cy - 1.05, 1.5, 1.2, None, kind="oval", line=RED, lw=2.0, dash=True)
    text(s, cx - 1.4, 1.05, 2.8, 0.4, [("谷（最良解の近く）", "r")], 16, "c")
    line(s, cx, 1.45, cx, cy - 1.05, color=RED, lw=1.5)
    text(s, 0.9, 5.25, 6.0, 0.4, [[("●", "r"), "：CIM の解　", ("●", "g"), "：同じカット値の SA の解"]], 16, "c")
    text(s, 0.9, 5.6, 6.0, 0.4, ["（仮説のイメージ）"], 16, "c")
    text(s, 7.2, 1.0, 5.9, 0.4, [("観察", "B")], 18)
    text(s, 7.2, 1.4, 5.9, 0.8, ["・同じカット値から始めても、CIM の解から始めた SA の方が良い（0.10% vs 0.13%）"], 16)
    block_arrow(s, 9.6, 2.25, 0.9, 0.28)
    text(s, 7.2, 2.55, 5.9, 0.4, [("解釈", "B")], 18)
    text(s, 7.2, 2.95, 5.9, 0.8, [["・初期解の良さは", ("カット値だけでは決まらない", "r")]], 16)
    block_arrow(s, 9.6, 3.6, 0.9, 0.28)
    text(s, 7.2, 3.9, 5.9, 0.4, [("仮説（今後確かめる）", "B")], 18)
    text(s, 7.2, 4.3, 5.9, 1.2, [["・CIM は", ("最良解に近い領域（谷）", "r"), "に解を出す"],
                                 "・SA はその近くを探すだけで仕上げられる"], 16)
    box(s, 1.0, 6.2, 11.3, 0.75, [["今後：CIM の解と最良解との", ("距離", "r"), "を測り、仮説を確かめる"]], 18, kind="round", lw=1.5)
    return s


def conclusion(prs):
    s = b.new_slide(prs, "結論と今後の予定")
    text(s, 1.1, 1.05, 6.7, 0.5, ["結論"], 24)
    text(s, 1.1, 1.6, 11.3, 2.6, [
        ["CIM の解から始めた SA は、同じカット値の解から始めた SA よりも", ("良い解に到達した", "r"),
         "（G22、gap 0.10% vs 0.13%、p < 0.001）。"],
        "",
        ["CIM の解には、", ("カット値以外の良さ", "r"), "がある。"]], 18)
    text(s, 1.1, 4.2, 6.7, 0.5, ["今後の予定"], 24)
    text(s, 1.1, 4.8, 11.3, 2.3, [
        "CIM の解と最良解との距離を測り、カット値以外の良さの正体を明らかにする。",
        "CIM の計算時間を含めた同じ総時間で、ランダムな初期解と比べる。",
        "GA の初期集団や、他の問題（G55・G70）でも成り立つかを明らかにする。"], 18, space=6)
    return s


def appendix_params(prs):
    s = b.new_slide(prs, "付録:実験パラメータ")
    data = [["", "G22"], ["a（ループ損失）", "0.303"], ["b（結合）", "−0.049"], ["c（雑音）", "7.0×10⁻⁶"],
            ["α（利得）", "7.06"], ["β（利得飽和）", "7.21"], ["ΔP（1周回の増分）", "1.6×10⁻⁵"]]
    text(s, 0.5, 1.0, 6.2, 0.4, ["CIM（Optuna で CIM 単体のカット値の平均を最大化）"], 16)
    gf = s.shapes.add_table(len(data), 2, Inches(0.8), Inches(1.5), Inches(5.0), Inches(2.9))
    gf._element.graphic.graphicData.tbl.tblPr.find(
        "{http://schemas.openxmlformats.org/drawingml/2006/main}tableStyleId").text = GRID
    gf.table.columns[0].width = Inches(3.0)
    gf.table.columns[1].width = Inches(2.0)
    for i, r in enumerate(data):
        for j, v in enumerate(r):
            c = gf.table.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = b.LIGHT if i == 0 else WHITE
            fill_text(c.text_frame, [v], 16, "c")
    text(s, 7.0, 1.0, 6.0, 0.4, ["CIM・SA の設定"], 16)
    text(s, 7.0, 1.5, 6.0, 4.0, [
        ["・", ("p", "v"), "(", ("n", "v"), ") = (", ("n", "v"), " Δ", ("P", "v"), ")", ("1/2", "p"), "、", ("n", "v"), ("0", "s"),
         " は周回ごと・スピンごとに引き直す標準正規乱数"],
        "・CIM の周回数：2400",
        "・SA：1 スピンを選んで反転を試み、メトロポリス法で受理",
        "・温度は 1 反転ごとに指数的に下げる（1 スイープ ＝ n 回の反転）",
        "・SA の設定は 3 者で共通（調整なし）"], 16, space=4)
    return s


def main():
    if DST.exists():
        raise FileExistsError(DST)
    prs = Presentation(str(SRC))
    page_number_bottom_right(prs)
    orig = list(prs.slides)
    b.cover(orig[0])
    b.set_title(orig[1], "背景:最適化問題の分類と例")
    box4 = next(sh for sh in orig[3].shapes if sh.name == "四角形: 角を丸くする 2")  # 右下の番号にかからないよう細くする
    box4.width = Inches(5.7)
    S = [orig[0], orig[1], b.maxcut(prs), orig[3], orig[4], orig[5], b.sim_model(prs, orig[6]), b.prior(prs),
         b.sa(prs), b.ga(prs), b.init_sol(prs), purpose(prs), method(prs), evaluation(prs)]
    S.append(b.result(prs, "結果:開始のカット値をそろえたSA（G22）", "s18_13.png", 9.6, [
        "・SA の解（ランダム開始の SA で得た解）から、CIM の解と同じカット値のものを選んで開始",
        "・カット値の良い解から始めると良くなる（ランダム 0.22% → SA の解 0.13%）",
        "・同じカット値でも、CIM の解から始めた方が良い（0.10%、p < 0.001）"],
        [["CIM の解には、", ("カット値以外の良さ", "r"), "がある"]]))
    S += [discussion(prs), conclusion(prs), b.schedule(prs), appendix_params(prs), b.references(prs)]

    lst = prs.slides._sldIdLst
    by_part = {prs.part.related_part(sid.rId): sid for sid in lst}
    keep = {sl.part for sl in S}
    for part, sid in list(by_part.items()):
        if part not in keep:
            prs.part.drop_rel(sid.rId)
            lst.remove(sid)
    for sl in S:
        sid = by_part[sl.part]
        lst.remove(sid)
        lst.append(sid)
    for sl in S:
        if sl not in orig:
            renumber(sl)
    prs.save(str(DST))
    print("saved", DST, len(S))


if __name__ == "__main__":
    main()
