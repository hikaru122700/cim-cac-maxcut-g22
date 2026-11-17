"""v38 用の部品：佐々木さんの中間報告（2025/11/27）の書式で図形・文字を置く。"""
import copy
import math
from pathlib import Path

from lxml import etree
from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

IMG = Path(__file__).parent / "img"
RED = RGBColor(0xFF, 0, 0)
BLUE = RGBColor(0, 0, 0xFF)
BLACK = RGBColor(0, 0, 0)
GRAY = RGBColor(0x80, 0x80, 0x80)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
COLORS = {"r": RED, "b": BLUE, "g": GRAY, "w": WHITE}
JP = "ＭＳ ゴシック"
EN = "Arial"
VAR = "Times New Roman"


def _no_style(sh):
    st = sh._element.find(qn("p:style"))
    if st is not None:
        sh._element.remove(st)


def _set_font(run, opts, size):
    f = run.font
    if size:
        f.size = Pt(size)
    f.color.rgb = next((COLORS[c] for c in opts if c in COLORS), BLACK)
    if "B" in opts:
        f.bold = True
    rpr = run._r.get_or_add_rPr()
    if "v" in opts:
        f.name = VAR
        f.italic = True
    else:
        f.name = EN
    latin = rpr.find(qn("a:latin"))
    ea = etree.SubElement(rpr, qn("a:ea"))
    ea.set("typeface", JP)
    latin.addnext(ea)
    if "s" in opts:
        rpr.set("baseline", "-25000")
    if "p" in opts:
        rpr.set("baseline", "30000")


def fill_text(tf, paras, size=18, align=None, space=None):
    """paras：[段落, ...]。段落は文字列か [(文字, opts), ...]。
    opts：r 赤 / b 青 / g 灰 / B 太字 / v 変数（Times 斜体）/ s 下付き / p 上付き。"""
    tf.clear()
    for k, para in enumerate(paras):
        p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
        if align:
            p.alignment = {"l": PP_ALIGN.LEFT, "c": PP_ALIGN.CENTER, "r": PP_ALIGN.RIGHT}[align]
        if space is not None:
            p.space_after = Pt(space)
        runs = [(para, "")] if isinstance(para, str) else ([para] if isinstance(para, tuple) else para)
        for item in runs:
            text, opts = (item, "") if isinstance(item, str) else item
            r = p.add_run()
            r.text = text
            _set_font(r, opts, size)


def text(slide, x, y, w, h, paras, size=18, align="l", anchor="t", space=None):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE, "b": MSO_ANCHOR.BOTTOM}[anchor]
    for m in ("margin_left", "margin_right"):
        setattr(tf, m, Inches(0.05))
    fill_text(tf, paras, size, align, space)
    return tb


def box(slide, x, y, w, h, paras=None, size=18, kind="rect", line=BLACK, lw=1.5, fill=None,
        align="c", anchor="m", dash=False):
    shp = {"rect": MSO_SHAPE.RECTANGLE, "round": MSO_SHAPE.ROUNDED_RECTANGLE,
           "oval": MSO_SHAPE.OVAL, "tri": MSO_SHAPE.ISOSCELES_TRIANGLE}[kind]
    sh = slide.shapes.add_shape(shp, Inches(x), Inches(y), Inches(w), Inches(h))
    _no_style(sh)
    if kind == "round":
        sh.adjustments[0] = min(0.16, 0.25 * min(w, h) / max(w, h) * 2)
    if fill is None:
        sh.fill.background()
    else:
        sh.fill.solid()
        sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(lw)
        if dash:
            sh.line.dash_style = 4  # dash
    tf = sh.text_frame
    tf.word_wrap = True
    for m in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, m, Inches(0.04))
    tf.vertical_anchor = {"t": MSO_ANCHOR.TOP, "m": MSO_ANCHOR.MIDDLE}[anchor]
    if paras:
        fill_text(tf, paras, size, align)
    return sh


def block_arrow(slide, x, y, w, h, direction="down", color=RED):
    kind = {"down": MSO_SHAPE.DOWN_ARROW, "right": MSO_SHAPE.RIGHT_ARROW,
            "up": MSO_SHAPE.UP_ARROW, "left": MSO_SHAPE.LEFT_ARROW}[direction]
    sh = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    _no_style(sh)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def _arrowhead(sh, end="tail"):
    ln = sh._element.spPr.find(qn("a:ln"))
    el = etree.SubElement(ln, qn(f"a:{end}End"))
    el.set("type", "triangle")
    el.set("w", "med")
    el.set("len", "med")


def line(slide, x1, y1, x2, y2, color=BLACK, lw=1.5, arrow=False, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    _no_style(c)
    c.line.color.rgb = color
    c.line.width = Pt(lw)
    if dash:
        c.line.dash_style = 4
    if arrow:
        _arrowhead(c)
    return c


def polyline(slide, pts, color=BLACK, lw=2.25, arrow=False, dash=False):
    """pts：インチの点列。開いた折れ線（曲線は細かい点列で）。"""
    fb = slide.shapes.build_freeform(Inches(pts[0][0]), Inches(pts[0][1]), scale=1.0)
    fb.add_line_segments([(Inches(px), Inches(py)) for px, py in pts[1:]], close=False)
    sh = fb.convert_to_shape()
    _no_style(sh)
    sh.fill.background()
    sh.line.color.rgb = color
    sh.line.width = Pt(lw)
    if dash:
        sh.line.dash_style = 4
    if arrow:
        _arrowhead(sh)
    return sh


def spin(slide, x, y, up, w=0.26, h=0.44, gray=False):
    """佐々木さんの図と同じスピン矢印：白抜き、↓ は赤枠、↑ は青枠。"""
    sh = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    _no_style(sh)
    sh.fill.solid()
    sh.fill.fore_color.rgb = RGBColor(0xD0, 0xD0, 0xD0) if gray else WHITE
    sh.line.color.rgb = GRAY if gray else (BLUE if up else RED)
    sh.line.width = Pt(1.5)
    if up:
        sh.rotation = 180
    return sh


def spin_row(slide, x, y, bits, w=0.22, h=0.4, gap=0.07, gray=()):
    for k, b in enumerate(bits):
        spin(slide, x + k * (w + gap), y, b == 1, w, h, gray=k in gray)
    return x + len(bits) * (w + gap) - gap


def node(slide, cx, cy, d, black):
    sh = box(slide, cx - d / 2, cy - d / 2, d, d, kind="oval", fill=BLACK if black else WHITE, lw=1.5)
    return sh


def picture(slide, name, x, y, w=None, h=None):
    path = IMG / name
    iw, ih = Image.open(path).size
    if w is not None and h is None:
        h = w * ih / iw
    elif h is not None and w is None:
        w = h * iw / ih
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))


def copy_shape(src_slide, sid, dst_slide, x, y, w=None):
    """同じ資料の別スライドから図形（グループ可）を写す。w を指定すると縦横比を保って拡大縮小。"""
    src = next(s for s in src_slide.shapes if s.shape_id == sid)
    xml = etree.tostring(src._element).decode()
    assert "r:embed" not in xml and "r:id=" not in xml, f"shape {sid} refers to external parts"
    el = copy.deepcopy(src._element)
    dst_slide.shapes._spTree.append(el)
    sh = dst_slide.shapes[-1]
    ratio = (w / (src.width / 914400)) if w else 1.0
    sh.left, sh.top = Inches(x), Inches(y)
    sh.width, sh.height = Emu(int(src.width * ratio)), Emu(int(src.height * ratio))
    return sh


def renumber(slide):
    for k, el in enumerate(slide.shapes._spTree.iter(qn("p:cNvPr")), start=2):
        el.set("id", str(k))


# ---------- エネルギーの山谷（SA・初期解・考察で使う） ----------

def energy(u):
    """u∈[0,1] のエネルギー。深い谷（最良解）は u≈0.68。"""
    return (0.62 - 0.40 * math.exp(-((u - 0.68) / 0.07) ** 2) - 0.20 * math.exp(-((u - 0.33) / 0.06) ** 2)
            - 0.16 * math.exp(-((u - 0.90) / 0.045) ** 2) - 0.14 * math.exp(-((u - 0.12) / 0.05) ** 2)
            + 0.04 * math.sin(20 * u))


def landscape(slide, x0, y0, w, h, xlabel="解（スピンの配置）", ylabel="エネルギー（−カット値）", size=16):
    us = [k / 200 for k in range(201)]
    es = [energy(u) for u in us]
    lo, hi = min(es), max(es)

    def pos(u, lift=0.0):
        e = energy(u)
        return x0 + 0.15 + u * (w - 0.3), y0 + 0.15 + (hi - e) / (hi - lo) * (h - 0.3) - lift

    polyline(slide, [pos(u) for u in us], lw=2.5)
    line(slide, x0, y0 + h, x0 + w, y0 + h, lw=1.5, arrow=True)
    line(slide, x0, y0 + h, x0, y0 - 0.1, lw=1.5, arrow=True)
    text(slide, x0, y0 + h + 0.02, w, 0.4, [xlabel], size, "c")
    tb = text(slide, x0 - 0.62, y0, 0.45, h, [ylabel], size, "c", "m")
    tb.text_frame._bodyPr.set("vert", "eaVert")
    return pos


def ball(slide, pos, u, d=0.26, color=RED):
    cx, cy = pos(u)
    return box(slide, cx - d / 2, cy - d, d, d, kind="oval", fill=color, line=None)


def local_min(u0, step=0.002):
    u = u0
    while True:
        nxt = min((u - step, u, u + step), key=energy)
        if nxt == u:
            return u
        u = nxt


def bezier(p0, p1, p2, n=30):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1]) for t in (k / n for k in range(n + 1))]
