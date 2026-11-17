"""卒論中間報告 v40：v39 に 11/16 進捗報告の検証1・2（最良解の谷、CIM の線形段と u1）を足す。
考察・結論・目的もそれに合わせて書き直す。"""
import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).parent))
import build_v38 as b  # noqa: E402
import build_v39 as b9  # noqa: E402
from lib38 import (BLACK, GRAY, RED, ball, block_arrow, box, landscape, line, local_min, picture,  # noqa: E402
                   renumber, spin_row, text)

DST = Path(os.environ.get("V40_OUT", "C:/Users/owner/Downloads/卒論中間報告_三輪光_v40.pptx"))


def U(k, opt=""):
    return [("u", "v" + opt), (str(k), "s" + opt)]


def purpose(prs):
    s = b.new_slide(prs, "本研究の目的")
    box(s, 1.09, 1.9, 11.15, 1.9, [["コヒーレントイジングマシンの出力解をSA・GAの初期解とすることで、"
                                    "限られた計算時間内に得られる解の品質が改善するかを明らかにする。"]], 24,
        kind="round", lw=3.0)
    text(s, 1.3, 4.3, 11.0, 2.2, [
        "・CIM の解を初期解にしたときの、SA の到達点を明らかにする。",
        [("・CIM の解の良さが、カット値で決まるのか、解の位置で決まるのかを明らかにする。", "r")],
        [("・CIM がなぜ良い位置に解を出すのかを、CIM の式から明らかにする。", "r")],
        "・CIM の計算時間を含めた同じ総時間で、ランダムな初期解と比べる。"], 18, space=6)
    return s


def method_distance(prs):
    s = b.new_slide(prs, "手法:最良解からの距離と「谷」（検証1）")
    text(s, 0.4, 0.95, 12.5, 0.45, [["解の位置を、最良解（BKS 解）からの", ("距離 ", "r"), ("d", "vr"), "で表す"]], 18, "c")
    best = [1, 0, 0, 1, 1, 0, 1, 0, 1, 1]
    text(s, 0.4, 1.65, 1.4, 0.4, ["最良解"], 16)
    spin_row(s, 1.8, 1.65, best)
    moved = best[:]
    moved[2], moved[6] = 1, 0
    text(s, 0.4, 2.3, 1.4, 0.4, ["別の解"], 16)
    spin_row(s, 1.8, 2.3, moved)
    for k in (2, 6):
        box(s, 1.8 + k * 0.29 - 0.06, 2.23, 0.34, 0.55, kind="round", lw=2.0, line=RED)
    text(s, 0.4, 2.95, 5.6, 1.6, [
        [("d", "v"), " = （値が違うスピンの数）/ ", ("n", "v"), "　（例：2 / 10 = 0.2）"],
        [("d", "v"), " = 0：同じ解、", ("d", "v"), " = 0.5：無関係な解"],
        "（全部反転した解は同じ分割なので同じとみなす）"], 16, space=3)
    box(s, 0.4, 4.55, 5.7, 0.9, [[("谷", "r"), "：", ("d", "v"), " < 0.3 の範囲"],
                                 "（ここから SA を始めると最良解と同じ質になる）"], 16, kind="round", lw=2.0)
    # 右：やったこと（ブロック図）
    text(s, 6.4, 1.5, 6.6, 0.4, ["やったこと（G22）"], 18, "c")
    box(s, 6.6, 2.0, 3.0, 0.85, ["最良解（89 個）", "を用意"], 16)
    box(s, 9.9, 2.0, 3.0, 0.85, ["CIM の解・同じ", "カット値の SA の解"], 16, line=RED, lw=2.0)
    line(s, 8.1, 2.85, 8.1, 3.15, lw=2.0, arrow=True)
    line(s, 11.4, 2.85, 11.4, 3.15, lw=2.0, arrow=True)
    box(s, 6.6, 3.15, 3.0, 0.85, [[("k", "v"), " 個を反転して"], [("d", "v"), " = ", ("k", "v"), " / ", ("n", "v"), " の開始解"]], 16)
    box(s, 9.9, 3.15, 3.0, 0.85, [["最良解との"], ["距離 ", ("d", "v"), " を測る"]], 16)
    line(s, 8.1, 4.0, 8.1, 4.3, lw=2.0, arrow=True)
    line(s, 11.4, 4.0, 11.4, 4.3, lw=2.0, arrow=True)
    box(s, 6.6, 4.3, 6.3, 0.75, [["SA（", ("T", "v"), ("0", "vs"), " = 2.0、5000 スイープ）を各 1600 本"]], 16)
    line(s, 9.75, 5.05, 9.75, 5.35, lw=2.0, arrow=True)
    box(s, 6.6, 5.35, 6.3, 0.6, ["SA 後の BKS との差"], 16)
    block_arrow(s, 6.07, 6.15, 1.2, 0.3)
    text(s, 0.5, 6.5, 12.3, 0.6, [["SA 後の質は、開始解の", ("カット値", "r"), "と", ("位置（", "r"), ("d", "vr"), ("）", "r"),
                                   "のどちらで決まるか？"]], 20, "c")
    return s


def two_fig_result(prs, title, figs, comments, concl):
    s = b.new_slide(prs, title)
    for k, (cap, img) in enumerate(figs):
        x = 0.3 + k * 4.6
        text(s, x, 0.95, 4.5, 0.45, [cap], 18, "c")
        picture(s, img, x, 1.45, w=4.5)
    text(s, 9.45, 1.1, 3.8, 4.4, comments, 16, space=8)
    block_arrow(s, 6.07, 5.3, 1.2, 0.32)
    text(s, 0.5, 5.7, 12.3, 0.8, concl, 20, "c")
    return s


def method_linear(prs):
    s = b.new_slide(prs, "手法:CIMの線形段と最大固有ベクトル（検証2）")
    text(s, 0.4, 0.95, 12.5, 0.45, ["CIM はなぜ最良解の谷に解を出すのか：CIM の式から予測を立てて確かめる"], 18, "c")
    picture(s, "s06_17.png", 0.5, 1.55, w=4.4)
    picture(s, "s06_18.png", 0.5, 2.1, w=4.55)
    text(s, 0.4, 2.75, 6.2, 3.0, [
        [("・線形段", "B"), "：振幅が小さい間は利得が全スピンで同じ → ", ("E", "v"), " に行列 ", ("a", "v"), " + ", ("bJ", "v"),
         " をかけ続けるだけ"],
        ["　→ ", ("J", "v"), " の固有値が最大の固有ベクトル "] + U(1) + [" の向きが一番速く育つ"],
        [("・非線形段", "B"), "：振幅が大きくなると利得が飽和し、各スピンが ±1 に決まる"],
        [("・予測", "r"), "：線形段が "] + U(1) + [" の側の谷を選び、非線形段が谷の奥へ運ぶ"]], 16, space=6)
    picture(s, "d1_amplitude_slide.png", 6.75, 1.45, w=6.3)
    text(s, 6.75, 4.3, 6.3, 0.45, ["CIM の振幅の育ち方（1024 本の中央値）"], 16, "c")
    text(s, 6.75, 4.7, 6.3, 0.45, [["線形段の終わり：ラウンド 236（振幅が急に立ち上がる所）"]], 16, "c")
    box(s, 0.6, 5.85, 12.1, 1.2, [
        [("やったこと：", "B"), "① 解と sign("] + U(1) + [")・sign("] + U(2) + [") との一致度を、SA 後の「谷の中」と「別の谷」で比べる"],
        ["② CIM を線形段の終わりで止めた解から SA を回し、最後まで回した CIM の解と比べる"]], 16, kind="round",
        align="l", lw=1.5)
    return s


def discussion(prs):
    s = b.new_slide(prs, "考察:なぜCIMの解から始めると良いのか")
    pos = landscape(s, 1.0, 1.4, 5.8, 3.4)
    u2 = local_min(0.68)
    for u in (0.63, 0.66, 0.71, 0.73):
        ball(s, pos, u, d=0.2)
    for u in (0.32, 0.65, 0.88):
        ball(s, pos, u, d=0.2, color=GRAY)
    cx, cy = pos(u2)
    box(s, cx - 0.75, cy - 1.05, 1.5, 1.2, None, kind="oval", line=RED, lw=2.0, dash=True)
    text(s, cx - 1.4, 1.05, 2.8, 0.4, [[("谷（", "r"), ("d", "vr"), (" < 0.3）", "r")]], 16, "c")
    line(s, cx, 1.45, cx, cy - 1.05, color=RED, lw=1.5)
    text(s, 0.6, 5.25, 6.6, 0.4, [[("●", "r"), "：CIM の解（99.7% が谷の中）"]], 16, "c")
    text(s, 0.6, 5.6, 6.6, 0.4, [[("●", "g"), "：同じカット値の SA の解（28% が別の谷）"]], 16, "c")
    text(s, 7.2, 1.0, 5.9, 0.4, [("結果", "B")], 18)
    text(s, 7.2, 1.4, 5.9, 0.8, ["・同じカット値から始めても、CIM の解から始めた SA の方が良い"], 16)
    block_arrow(s, 9.6, 2.15, 0.9, 0.28)
    text(s, 7.2, 2.45, 5.9, 0.4, [("検証1", "B")], 18)
    text(s, 7.2, 2.85, 5.9, 0.8, [["・CIM の解は", ("最良解の谷の中", "r"), "にあり、SA 後の質は解の位置で決まる"]], 16)
    block_arrow(s, 9.6, 3.65, 0.9, 0.28)
    text(s, 7.2, 3.95, 5.9, 0.4, [("検証2", "B")], 18)
    text(s, 7.2, 4.35, 5.9, 1.2, [["・CIM の", ("線形段", "r"), "が解を "] + U(1) + ["（最良解の谷の側）へ向け、カット値上位を選ぶと谷に入る"]], 16)
    box(s, 0.8, 6.15, 11.7, 0.85, [[("CIM は問題の結合 ", ""), ("J", "v"), " の構造（最大固有ベクトル）を使って、",
                                    ("最良解の谷に初期解を置ける", "r")]], 18, kind="round", lw=1.5)
    return s


def conclusion(prs):
    s = b.new_slide(prs, "結論と今後の予定")
    text(s, 1.1, 1.0, 6.7, 0.5, ["結論"], 24)
    text(s, 1.1, 1.55, 11.3, 3.0, [
        ["CIM の解から始めた SA は、同じカット値の解から始めた SA よりも", ("良い解に到達した", "r"),
         "（G22、gap 0.10% vs 0.13%）。"],
        ["CIM の解の良さはカット値ではなく", ("解の位置", "r"), "である。カット値上位の CIM の解は 99.7% が最良解の谷の中にあり、"
         "谷の中から始めれば SA 後の質は最良解と同程度になる。"],
        ["谷に入るのは、CIM の", ("線形段", "r"), "（", ("J", "v"), " の最大固有ベクトル "] + U(1) +
        [" の側へ向ける）と、カット値上位を選ぶ段階による。"]], 18, space=10)
    text(s, 1.1, 4.6, 6.7, 0.5, ["今後の予定"], 24)
    text(s, 1.1, 5.15, 11.3, 2.0, [
        "大きい問題（G55・G70）でも、CIM の解が谷に入るかを明らかにする。",
        "CIM の計算時間を含めた同じ総時間で、ランダムな初期解と比べる。",
        "GA の初期集団でも同じ効果があるかを明らかにする。"], 18, space=6)
    return s


def definitions(prs):
    s = b.new_slide(prs, "付録:言葉の定義（G22）")
    text(s, 0.8, 1.1, 11.8, 5.8, [
        "・最良解（BKS 解）：カット値 13359（G22 で知られている最良）に届いた解。長い SA 7200 本で集めた 89 個。"
        "互いに 15 スピン以下しか違わず、ほぼ 1 点に集まっている",
        ["・距離 ", ("d", "v"), "：解と、最も近い BKS 解とで向きが違うスピンの割合（全部反転した解は同じとみなす）"],
        ["・最良解の谷：", ("d", "v"), " < 0.3 の範囲。検証1 で、", ("d", "v"),
         " ≤ 0.3 から SA で仕上げると約 0.10% で一定、0.4 から悪くなったので 0.3 を境にした"],
        ["・SA 後の「谷の中」と「別の谷」：ランダムな解から SA（", ("T", "v"), ("0", "vs"),
         " = 2.0、5000 スイープ）を 1600 本回し、最後の解の ", ("d", "v"), " が 0.3 未満（880 本）か 0.3 以上（720 本）かで分けた"],
        ["・一致度 ov", ("k", "s"), "：解と sign("] + U("k") + [") で符号が一致するスピンの割合を ", ("p", "v"),
                                                               " として |2", ("p", "v"), " − 1|。0 = 無関係、1 = 同じ"],
        "・線形段の終わり：利得の指数が下がる割合（飽和度）が初めて 0.01 以上になったラウンド（中央値 236）"],
        16, space=10)
    return s


def main():
    if DST.exists():
        raise FileExistsError(DST)
    prs = Presentation(str(b9.SRC))
    b9.page_number_bottom_right(prs)
    orig = list(prs.slides)
    b.cover(orig[0])
    b.set_title(orig[1], "背景:最適化問題の分類と例")
    box4 = next(sh for sh in orig[3].shapes if sh.name == "四角形: 角を丸くする 2")
    box4.width = Inches(5.7)
    S = [orig[0], orig[1], b.maxcut(prs), orig[3], orig[4], orig[5], b.sim_model(prs, orig[6]), b.prior(prs),
         b.sa(prs), b.ga(prs), b.init_sol(prs), purpose(prs), b9.method(prs), b9.evaluation(prs)]
    S.append(b.result(prs, "結果:開始のカット値をそろえたSA（G22）", "s18_13.png", 9.6, [
        "・SA の解（ランダム開始の SA で得た解）から、CIM の解と同じカット値のものを選んで開始",
        "・カット値の良い解から始めると良くなる（ランダム 0.22% → SA の解 0.13%）",
        "・同じカット値でも、CIM の解から始めた方が良い（0.10%、p < 0.001）"],
        [["CIM の解には、", ("カット値以外の良さ", "r"), "がある → それは何か？"]]))
    S.append(method_distance(prs))
    S.append(two_fig_result(prs, "結果:SA後の質は開始解の位置で決まる（検証1）",
                            [("開始解の距離と仕上げ後", "b2_gap_vs_distance_slide.png"),
                             ("開始解の距離の分布", "b1_start_distance_slide.png")], [
        [("d", "v"), " ≤ 0.3 なら、開始時にカット値が 21% 悪くても、仕上げ後は約 0.10%（CIM の解と同じ）"],
        ["カット値上位 16/256 の CIM の解は 99.7% が ", ("d", "v"), " < 0.3（谷の中）"],
        "カット値が同じ SA の解は 28% が別の谷"],
        [["CIM の解の良さは、カット値ではなく", ("解の位置（最良解の谷の中）", "r")]]))
    S.append(method_linear(prs))
    S.append(two_fig_result(prs, "結果:CIMの線形段が解を谷の側へ向ける（検証2）",
                            [([("u", "v"), ("1", "s"), "・", ("u", "v"), ("2", "s"), " との一致度"], "c1_overlap_counts_slide.png"),
                             ("線形段で止めると", "d2_linear_stop_slide.png")], [
        ["SA 後、谷の中は "] + U(1) + ["（0.38 > 0.18）、別の谷は "] + U(2) + ["（0.32 > 0.21）に寄る → ",
                                                               ("最良解の谷は ", "r")] + U(1, "r") + [("の側", "r")],
        ["線形段で止めて SA：0.111%（最後まで回すと 0.097%、p = 0.001）→ 非線形段の効果は小さい"],
        "谷に入るかは、カット値上位 16/256 を選ぶ段階が大きい（1 本ずつでは 68% が谷の外）"],
        [["谷に入るのは CIM の", ("線形段", "r"), "と、", ("カット値上位を選ぶこと", "r")]]))
    S += [discussion(prs), conclusion(prs), b.schedule(prs), definitions(prs), b9.appendix_params(prs),
          b.references(prs)]

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
