"""卒論中間報告 v46：中身と図は v45 のまま、体裁を海津さんの中間報告（v37 までの土台）に合わせる。
上の紺の帯にタイトル、右上に大阪大学のロゴ、下の帯に進行ドットと右下のページ番号、Outline のページ。"""
import copy
import os
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Inches, Pt

sys.path.insert(0, str(Path(__file__).parent))
import build_v38 as b  # noqa: E402
import build_v39 as b9  # noqa: E402
import build_v45 as b45  # noqa: E402  （背景ページの言い換えもここで有効になる）
import bg_v45  # noqa: E402
import eval_v43  # noqa: E402
from lib38 import WHITE, box, fill_text, renumber  # noqa: E402

SRC = Path("C:/Users/owner/Downloads/卒論中間報告_三輪光_v37.pptx")  # 海津さんの体裁の資料
DST = Path(os.environ.get("V46_OUT", "C:/Users/owner/Downloads/卒論中間報告_三輪光_v46.pptx"))
TOP, BOTTOM = 1.02, 7.0          # 海津さんの体裁で中身を置ける範囲（上の帯の下〜下の帯の上）
SRC_TOP, SRC_BOTTOM = 0.95, 7.35  # v45 で中身を置いていた範囲
TITLES = {"結論と今後の予定": "まとめ・今後の課題", "研究計画": "今後の予定",
          "背景:発見的解法 シミュレーテッドアニーリング(SA)": "発見的解法：SA（焼きなまし法）",
          "背景:発見的解法 遺伝的アルゴリズム(GA)": "発見的解法：GA（遺伝的アルゴリズム）",
          "手法:CIMの線形段と最大固有ベクトル（検証2）": "手法：CIM の線形段と固有ベクトル（検証2）"}
SECTIONS = [1, 10, 1, 4, 1, 2, 1]  # 進行ドット：Outline、背景、目的、手法、結果、考察・まとめ、今後の予定

_sldnum = None


def kaizu_title(title):
    title = TITLES.get(title, title)
    if title.startswith("背景:"):
        title = title[len("背景:"):]
    return title.replace(":", "：")


def kaizu_new_slide(prs, title):
    s = prs.slides.add_slide(prs.slide_layouts[1])  # タイトルとコンテンツ
    for ph in list(s.placeholders):
        if ph.placeholder_format.idx != 0:
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Emu(118872), Emu(137160), Emu(9144000), Emu(548640)  # v37 と同じ位置
    tf = t.text_frame
    tf.text = kaizu_title(title)
    tf.paragraphs[0].runs[0].font.size = Pt(30)
    s.shapes._spTree.append(copy.deepcopy(_sldnum))
    return s


def squeeze(slide):
    """v45 の配置（縦 0.95〜7.35）を、海津さんの体裁の範囲（1.02〜7.0）に縦方向で収める。"""
    k = (BOTTOM - TOP) / (SRC_BOTTOM - SRC_TOP)
    for sh in slide.shapes:
        if sh.is_placeholder:
            continue
        top, h, w = sh.top / 914400, sh.height / 914400, sh.width / 914400
        new_top = TOP + (top - SRC_TOP) * k
        keep_aspect = sh.shape_type == 13 or (sh.name.startswith("Oval") and abs(w - h) < 0.02)
        if keep_aspect:
            nw = w * k
            sh.left = Emu(int(sh.left + (w - nw) / 2 * 914400))
            sh.width = Inches(nw)
        sh.top, sh.height = Inches(new_top), Inches(h * k)


def dots(slide, current):
    """下の帯の進行ドット：current 番目まで白く塗る。"""
    k, x = 0, 0.15
    for g, n in enumerate(SECTIONS):
        for _ in range(n):
            box(slide, x, 7.2, 0.2, 0.2, None, kind="oval", fill=WHITE if k <= current else None, line=WHITE, lw=1.0)
            k += 1
            x += 0.25
        x += 0.22


def outline(slide):
    tb = next(sh for sh in slide.shapes if sh.has_text_frame and "研究背景" in sh.text_frame.text)
    fill_text(tb.text_frame, [
        "1.  研究背景",
        "　　　組合せ最適化・イジングモデル・CIM・先行研究・SA / GA・初期解",
        "2.  研究目的",
        "3.  手法",
        "　　　開始のカット値をそろえた比較・最良解からの距離・CIM の線形段",
        "4.  結果",
        "5.  考察・まとめ",
        "6.  今後の予定"], 20, space=6)


def main():
    global _sldnum
    if DST.exists():
        raise FileExistsError(DST)
    prs = Presentation(str(SRC))
    orig = list(prs.slides)
    _sldnum = next(sh for sh in orig[2].shapes if sh.is_placeholder and sh.placeholder_format.idx == 12)._element
    b.new_slide = kaizu_new_slide
    outline(orig[1])

    main_slides = [orig[1], bg_v45.optim_problem(prs), b.maxcut(prs), bg_v45.ising_map(prs), bg_v45.cim_setup(prs),
                   bg_v45.cim_bifurcation(prs), bg_v45.sim_model(prs), b.prior(prs), b.sa(prs), b.ga(prs),
                   b.init_sol(prs), b45.purpose(prs), b9.method(prs), eval_v43.evaluation(prs),
                   b45.method_distance(prs), b45.method_linear(prs), b45.one_result(prs),
                   b45.discussion(prs), b45.conclusion(prs), b.schedule(prs)]
    appendix = [b45.definitions(prs), b9.appendix_params(prs), b.references(prs)]
    assert len(main_slides) == sum(SECTIONS)
    for sl in main_slides[1:] + appendix:
        squeeze(sl)
    for sh in [sh for sh in orig[1].shapes if sh.name.startswith("Oval")]:
        sh._element.getparent().remove(sh._element)  # v37 のドット（13 個）を消して付け直す
    for i, sl in enumerate(main_slides):
        dots(sl, i)
    S = [orig[0]] + main_slides + appendix

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
