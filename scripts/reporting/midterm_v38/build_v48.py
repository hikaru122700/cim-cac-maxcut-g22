"""卒論中間報告 v48：Drive 上で編集された v47 から、非表示（スキップ）にしたスライドを消す。
残ったページに合わせて、進行ドットと Outline を直す。"""
import sys
from pathlib import Path

from pptx import Presentation

sys.path.insert(0, str(Path(__file__).parent))
from lib38 import WHITE, box, fill_text  # noqa: E402

SECTIONS = [1, 9, 1, 1, 1, 1]  # Outline、背景、目的、手法、結果、今後の予定
N_APPENDIX = 2                 # 末尾の付録：実験パラメータ、参考文献（ドットなし）


def is_dot(sh):
    return abs(sh.top / 914400 - 7.2) < 0.02 and abs(sh.width / 914400 - 0.2) < 0.02


def dots(slide, current):
    for sh in [sh for sh in slide.shapes if is_dot(sh)]:
        sh._element.getparent().remove(sh._element)
    k, x = 0, 0.15
    for n in SECTIONS:
        for _ in range(n):
            box(slide, x, 7.2, 0.2, 0.2, None, kind="oval", fill=WHITE if k <= current else None, line=WHITE, lw=1.0)
            k += 1
            x += 0.25
        x += 0.22


def outline(slide):
    tb = next(sh for sh in slide.shapes if sh.has_text_frame and "研究背景" in sh.text_frame.text)
    fill_text(tb.text_frame, [
        "1.  研究背景",
        "　　　組合せ最適化・イジングモデル・CIM・SA / GA・初期解",
        "2.  研究目的",
        "3.  手法",
        "　　　開始のカット値をそろえた比較・最良解からの距離・CIM の線形段",
        "4.  結果",
        "5.  今後の予定"], 20, space=6)


def main(src, dst):
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    prs = Presentation(src)
    lst = prs.slides._sldIdLst
    for sl, sid in list(zip(prs.slides, list(lst))):
        if sl._element.get("show") == "0":
            print("remove:", sl.shapes.title.text if sl.shapes.title is not None else "")
            prs.part.drop_rel(sid.rId)
            lst.remove(sid)
    slides = list(prs.slides)
    main_slides = slides[1:len(slides) - N_APPENDIX]
    assert len(main_slides) == sum(SECTIONS), (len(main_slides), sum(SECTIONS))
    outline(main_slides[0])
    for i, sl in enumerate(main_slides):
        dots(sl, i)
    prs.save(str(dst))
    print("saved", dst, len(slides))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
