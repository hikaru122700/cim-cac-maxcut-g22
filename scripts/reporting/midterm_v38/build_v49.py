"""卒論中間報告 v49：v48 の赤い太矢印（横に潰れていた）だけを直す。
高さと上端はそのまま、幅を高さの 1.9 倍に縮めて中央に置き、矢じりを全高の 6 割にする。ほかは変えない。"""
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.util import Emu

RED = RGBColor(0xFF, 0x00, 0x00)
ASPECT, SHAFT, HEAD = 1.9, 0.5, 0.6


def is_red_block_arrow(sh):
    if sh.shape_type != MSO_SHAPE_TYPE.AUTO_SHAPE or sh.auto_shape_type != MSO_SHAPE.DOWN_ARROW:
        return False
    return sh.fill.type == 1 and sh.fill.fore_color.rgb == RED and sh.width > sh.height


def main(src, dst):
    dst = Path(dst)
    if dst.exists():
        raise FileExistsError(dst)
    prs = Presentation(src)
    for i, sl in enumerate(prs.slides, 1):
        for sh in sl.shapes:
            if is_red_block_arrow(sh):
                cx, nw = sh.left + sh.width // 2, int(sh.height * ASPECT)
                sh.left, sh.width = Emu(cx - nw // 2), Emu(nw)
                sh.adjustments[0], sh.adjustments[1] = SHAFT, HEAD
                print(i, sh.name)
    prs.save(str(dst))
    print("saved", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
