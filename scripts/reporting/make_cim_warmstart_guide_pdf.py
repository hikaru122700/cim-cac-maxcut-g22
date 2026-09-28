"""Create a Japanese PDF guide for the CIM warm-start result graphs."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "cim_warmstart_graph_guide_ja.pdf"
G22_GRAPH = (
    ROOT
    / "results"
    / "2026-07-24"
    / "cim_warmstart"
    / "v3_G22_nt16_main"
    / "gap_vs_total_time.png"
)
G55_GRAPH = (
    ROOT
    / "results"
    / "2026-07-24"
    / "cim_warmstart"
    / "v4_G55_nt16_replication"
    / "gap_vs_total_time.png"
)
SA_MATCHED_GRAPH = (
    ROOT
    / "results"
    / "2026-07-24"
    / "cim_warmstart"
    / "sa_matched_v1_G22_G55_nt16"
    / "sa_matched_cold_vs_cim.png"
)

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2C6E9B")
PALE_BLUE = colors.HexColor("#EAF3F8")
PALE_GRAY = colors.HexColor("#F3F5F7")
RED = colors.HexColor("#D62728")
PURPLE = colors.HexColor("#9467BD")
ORANGE = colors.HexColor("#FF7F0E")
GREEN = colors.HexColor("#147D64")
DARK = colors.HexColor("#27333D")
MID = colors.HexColor("#596773")


def register_fonts() -> tuple[str, str]:
    regular_path = Path(r"C:\Windows\Fonts\YuGothM.ttc")
    bold_path = Path(r"C:\Windows\Fonts\YuGothB.ttc")
    pdfmetrics.registerFont(TTFont("YuGothic", str(regular_path), subfontIndex=0))
    if bold_path.exists():
        pdfmetrics.registerFont(TTFont("YuGothic-Bold", str(bold_path), subfontIndex=0))
    else:
        pdfmetrics.registerFont(TTFont("YuGothic-Bold", str(regular_path), subfontIndex=0))
    return "YuGothic", "YuGothic-Bold"


FONT, FONT_BOLD = register_fonts()
BASE = getSampleStyleSheet()

TITLE = ParagraphStyle(
    "TitleJa",
    parent=BASE["Title"],
    fontName=FONT_BOLD,
    fontSize=24,
    leading=31,
    textColor=NAVY,
    alignment=TA_CENTER,
    spaceAfter=8 * mm,
)
SUBTITLE = ParagraphStyle(
    "SubtitleJa",
    parent=BASE["Normal"],
    fontName=FONT,
    fontSize=12,
    leading=18,
    textColor=MID,
    alignment=TA_CENTER,
    spaceAfter=5 * mm,
)
H1 = ParagraphStyle(
    "H1Ja",
    parent=BASE["Heading1"],
    fontName=FONT_BOLD,
    fontSize=18,
    leading=23,
    textColor=NAVY,
    spaceAfter=3 * mm,
)
H2 = ParagraphStyle(
    "H2Ja",
    parent=BASE["Heading2"],
    fontName=FONT_BOLD,
    fontSize=12,
    leading=16,
    textColor=NAVY,
    spaceAfter=1.2 * mm,
)
BODY = ParagraphStyle(
    "BodyJa",
    parent=BASE["BodyText"],
    fontName=FONT,
    fontSize=9.2,
    leading=14,
    textColor=DARK,
    alignment=TA_LEFT,
)
BODY_SMALL = ParagraphStyle(
    "BodySmallJa",
    parent=BODY,
    fontSize=8,
    leading=11.5,
)
BOX_TITLE = ParagraphStyle(
    "BoxTitleJa",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=11,
    leading=15,
    textColor=NAVY,
    spaceAfter=1.2 * mm,
)
CALLOUT = ParagraphStyle(
    "CalloutJa",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=11,
    leading=16,
    textColor=GREEN,
    alignment=TA_CENTER,
)
WHITE_BOLD = ParagraphStyle(
    "WhiteBoldJa",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=10,
    leading=14,
    textColor=colors.white,
    alignment=TA_CENTER,
)
TABLE_HEAD = ParagraphStyle(
    "TableHeadJa",
    parent=BODY_SMALL,
    fontName=FONT_BOLD,
    fontSize=8,
    leading=10,
    textColor=colors.white,
    alignment=TA_CENTER,
)
TABLE_CELL = ParagraphStyle(
    "TableCellJa",
    parent=BODY_SMALL,
    fontSize=7.7,
    leading=10,
    alignment=TA_CENTER,
)


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def colored_label(color, text: str) -> Table:
    dot = Table([[""]], colWidths=[5 * mm], rowHeights=[5 * mm])
    dot.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), color),
                ("BOX", (0, 0), (-1, -1), 0.5, color),
            ]
        )
    )
    table = Table([[dot, p(text, BODY)]], colWidths=[8 * mm, 70 * mm])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1 * mm),
            ]
        )
    )
    return table


def info_box(title: str, text: str, background=PALE_BLUE) -> Table:
    box = Table(
        [[p(title, BOX_TITLE)], [p(text, BODY)]],
        colWidths=[82 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#C8D7E2")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return box


def key_takeaway(text: str) -> Table:
    table = Table([[p(text, CALLOUT)]], colWidths=[267 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#E8F6F1")),
                ("BOX", (0, 0), (-1, -1), 1.0, GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return table


def data_table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    content = [[p(x, TABLE_HEAD) for x in headers]]
    content.extend([[p(str(x), TABLE_CELL) for x in row] for row in rows])
    table = Table(content, colWidths=[x * mm for x in widths], repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#C8D1D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2 * mm),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2 * mm),
    ]
    for row in range(1, len(content)):
        style.append(
            (
                "BACKGROUND",
                (0, row),
                (-1, row),
                colors.white if row % 2 else PALE_GRAY,
            )
        )
    table.setStyle(TableStyle(style))
    return table


def chart_header(title: str, subtitle: str) -> list:
    return [
        p(title, H1),
        p(subtitle, BODY_SMALL),
        Spacer(1, 2 * mm),
    ]


def chart_image(path: Path) -> Image:
    probe = Image(str(path))
    width = 267 * mm
    height = width * probe.imageHeight / probe.imageWidth
    return Image(str(path), width=width, height=height)


def chart_image_width(path: Path, width_mm: float) -> Image:
    probe = Image(str(path))
    width = width_mm * mm
    height = width * probe.imageHeight / probe.imageWidth
    return Image(str(path), width=width, height=height, hAlign="CENTER")


def explanation_columns(cards: list[tuple[str, str]]) -> Table:
    cells = [info_box(title, text, PALE_GRAY) for title, text in cards]
    table = Table([cells], colWidths=[87 * mm] * 3, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.2 * mm),
            ]
        )
    )
    return table


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#D8DEE4"))
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MID)
    canvas.drawString(15 * mm, 6.5 * mm, "CIM warm-start 実験結果の読み方")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, f"{doc.page}")
    canvas.restoreState()


def build_pdf() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="CIM warm-start 実験結果の読み方",
        author="cim-cac-maxcut-g22 project",
    )
    story = []

    # Page 1: reading guide
    story.extend(
        [
            Spacer(1, 5 * mm),
            p("CIM warm-start 実験結果の読み方", TITLE),
            p(
                "GA・SA・PT-ICMをランダム初期化した場合と、CIM解から開始した場合を比較したグラフです。",
                SUBTITLE,
            ),
        ]
    )
    axis_table = Table(
        [
            [
                info_box(
                    "横軸：end-to-end wall time [秒]",
                    "CIMで初期解を作る時間と、後段ソルバを動かす時間の合計です。"
                    "<b>左ほど短時間</b>です。横軸は対数目盛なので、点の間隔は時間差に比例しません。",
                ),
                info_box(
                    "縦軸：gap to BKS",
                    "既知最良値（BKS）との差です。gap 1は既知最良より1小さい解、"
                    "gap 100は100小さい解を表します。<b>下ほど良い解</b>です。",
                ),
                info_box(
                    "理想の位置",
                    "<b>左下</b>ほど「短時間で良い解」です。warmの点がcoldより左下にあれば、"
                    "CIM初期解の利用に効果があったと読めます。",
                    colors.HexColor("#E8F6F1"),
                ),
            ]
        ],
        colWidths=[87 * mm] * 3,
    )
    axis_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 1.2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 1.2 * mm),
            ]
        )
    )
    story.extend([axis_table, Spacer(1, 6 * mm), p("線と点の意味", H1)])
    legend = Table(
        [
            [
                colored_label(colors.HexColor("#777777"), "<b>cold</b>：ランダムな初期解から開始"),
                colored_label(RED, "<b>CIM warm</b>：CIM解をGA・SAへ注入"),
                colored_label(PURPLE, "<b>CIM single</b>：最低温PTレプリカへCIM解を1つ注入"),
            ],
            [
                colored_label(ORANGE, "<b>CIM ladder</b>：温度ごとにCIM解を段階的に崩して注入"),
                p("<b>各点</b>：異なる計算予算で行ったbest-of-16の結果", BODY),
                p("<b>線</b>：各予算点を見やすく結んだもの。単一試行の連続軌跡ではありません。", BODY),
            ],
        ],
        colWidths=[87 * mm] * 3,
    )
    legend.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GRAY),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#D7DEE4")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D7DEE4")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    story.extend(
        [
            legend,
            Spacer(1, 7 * mm),
            key_takeaway(
                "まず同じパネル内で灰色（cold）と色付き（CIM warm）を比べます。"
                " 色付きの点が灰色より下なら、同程度の時間でより良い解を得ています。"
            ),
            PageBreak(),
        ]
    )

    # Page 2: G22
    story.extend(
        chart_header(
            "G22の結果：CIM初期解はPT-ICMの短時間性能を大きく改善",
            "BKS=13359、16 trialの最大値。横軸にはCIM生成時間も含まれています。",
        )
    )
    story.extend([chart_image(G22_GRAPH), Spacer(1, 2.5 * mm)])
    story.append(
        explanation_columns(
            [
                (
                    "左：GA",
                    "3世代ではcoldのgap 2に対し、CIM warmはgap 1です。"
                    "20世代では両方gap 1となり、<b>G22ではGAへの効果は小さい</b>と読めます。",
                ),
                (
                    "中央：SA",
                    "この図のSAはcoldとwarmで温度が異なる旧比較です。"
                    "同一温度・同一seedで再測定した公平比較では、10万・30万反復で有意に改善し、"
                    "100万反復以降は差が消えました。<b>詳細は8ページ</b>です。",
                ),
                (
                    "右：PT-ICM",
                    "30 sweepでcoldはgap 167、ladderはgap 31。200 sweepではcoldがgap 46、"
                    "ladderがgap 6です。<b>橙色のladderが一貫して最良</b>です。",
                ),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 3 * mm),
            key_takeaway(
                "G22の結論：CIM→PT-ICM ladderは、約1.3秒でgap 31、約6.7秒でgap 6まで改善。"
                " GAはほぼ同等。公平比較後も、SAは短い仕上げに限って利用価値が確認されました。"
            ),
            PageBreak(),
        ]
    )

    # Page 3: G55
    story.extend(
        chart_header(
            "G55：GAとPT-ICMでもCIM初期解が有効",
            "BKS=10299。G22と同じ設定・16 trialで再現実験を行いました。",
        )
    )
    story.extend([chart_image(G55_GRAPH), Spacer(1, 2.5 * mm)])
    story.append(
        explanation_columns(
            [
                (
                    "左：GA",
                    "3世代ではcold gap 212、CIM warm gap 85。20世代でもcold gap 122、"
                    "warm gap 85です。<b>GAの立ち上がりが大幅に改善</b>しました。",
                ),
                (
                    "中央：SA",
                    "この図は温度が異なる旧比較です。同一温度・同一seedの再測定では、"
                    "10万・30万反復で全16試行が改善し、100万反復以降の差は非有意でした。"
                    "<b>詳細は8ページ</b>です。",
                ),
                (
                    "右：PT-ICM",
                    "30 sweepではcold gap 267、ladder gap 81。200 sweepではcold gap 94、"
                    "ladder gap 56。<b>全予算でladderがcoldを上回りました</b>。",
                ),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 3 * mm),
            key_takeaway(
                "G55の結論：CIM warm-startはGAとPT-ICMの短時間性能を改善。"
                " 特にPT-ICM ladderはG22・G55の両方で有効でした。"
                "<br/><font size='8'>一般化にはG70やK2000など、未使用問題での追加検証が必要です。</font>"
            ),
            PageBreak(),
        ]
    )

    # Page 4: structural comparison
    story.extend(
        [
            p("なぜ効果が違うのか：まず問題構造と改善余地を比較", H1),
            p(
                "同じCIM初期化でも、後段手法がランダム初期化だけで十分強い場合は効果が見えません。"
                "構造の違いに加えて、cold startの到達点とCIM解の相対的位置が重要です。",
                BODY,
            ),
            Spacer(1, 4 * mm),
            data_table(
                [
                    "問題",
                    "頂点数",
                    "辺数",
                    "密度",
                    "平均次数",
                    "重み",
                    "CIM 600<br/>gap",
                    "CIM解間<br/>距離",
                    "cold GA<br/>3世代 gap",
                ],
                [
                    ["G22", "2,000", "19,990", "1.0%", "20.0", "+1", "89", "0.277", "2"],
                    ["G55", "5,000", "12,498", "0.1%", "5.0", "+1", "120", "0.383", "212"],
                    [
                        "K2000",
                        "2,000",
                        "1,999,000",
                        "100%",
                        "1,999",
                        "±1",
                        "823",
                        "0.447",
                        "90",
                    ],
                ],
                [25, 25, 32, 24, 27, 23, 28, 34, 39],
            ),
            Spacer(1, 5 * mm),
            p("効果を決める3つの要因", H1),
            explanation_columns(
                [
                    (
                        "1. 改善余地（天井効果）",
                        "G22のcold GAはわずか3世代でgap 2です。どれほど良い初期解を入れても、"
                        "改善できるのは最大2程度です。G55はcold gap 212なので、大きな差を出せます。",
                    ),
                    (
                        "2. 初期解の質と構造",
                        "CIMのcut値だけでなく、ランダム解にはない相関したスピン配置を渡せるかが重要です。"
                        "G55・K2000ではCIM解間の距離も大きく、異なる探索領域を供給しています。",
                    ),
                    (
                        "3. 後段手法の役割",
                        "GA・SA・PT-ICMは同じ動作ではありません。GAは交叉とTabu Search、SAは温度、"
                        "PT-ICMは温度レプリカとクラスタ移動を使うため、初期解の残り方が異なります。",
                    ),
                ]
            ),
            Spacer(1, 5 * mm),
            key_takeaway(
                "「CIMがG22に効かない」のではなく、G22ではGA単体がすでに強すぎて差が出にくい、"
                "というのが正確です。PT-ICMではG22でも大きな効果が出ています。"
            ),
            Spacer(1, 2 * mm),
            p(
                "研究上の注意：以下は構造統計・収束結果・局所gain診断と整合する機序仮説です。"
                "因果関係の確定には、次数・密度・重みを独立に変えるアブレーション実験が必要です。",
                BODY_SMALL,
            ),
            PageBreak(),
        ]
    )

    # Page 5: G22 versus G55 mechanism
    story.extend(
        [
            p("G22とG55：アルゴリズムごとに効果が違う理由", H1),
            Spacer(1, 2 * mm),
        ]
    )
    contrast = Table(
        [
            [p("観点", TABLE_HEAD), p("G22", TABLE_HEAD), p("G55", TABLE_HEAD)],
            [
                p("GA", TABLE_HEAD),
                p(
                    "<b>coldだけでほぼ解けています。</b><br/>"
                    "3世代でgap 2、20世代でgap 1。CIM warmもgap 1なので、"
                    "初期解の差よりGA内部のTabu Searchが支配的です。",
                    BODY_SMALL,
                ),
                p(
                    "<b>短いGAには探索予算が不足します。</b><br/>"
                    "3世代のcoldはgap 212ですが、CIM warmはgap 85。"
                    "CIMが先に良い盆地へ置くことで、少ない世代でも高品質になります。",
                    BODY_SMALL,
                ),
            ],
            [
                p("SA", TABLE_HEAD),
                p(
                    "同一のG22用Optuna温度（4.214→0.0517）・反復数・seedで再比較しました。"
                    "10万反復は平均+111.2、30万反復は+22.2で有意。"
                    "100万反復以降は95%信頼区間が0をまたぎ、明確な差はありません。",
                    BODY_SMALL,
                ),
                p(
                    "同一のG55用Optuna温度（1.822→0.0931）で再比較しました。"
                    "10万反復は平均+217.5、30万反復は+70.1で全16試行が改善。"
                    "100万反復以降は差が非有意となり、初期値の記憶が薄れます。",
                    BODY_SMALL,
                ),
            ],
            [
                p("PT-ICM", TABLE_HEAD),
                p(
                    "<b>G22でも効果は大きいです。</b><br/>"
                    "cold PTは短時間では温度間の混合に時間がかかります。"
                    "CIM seedはこのburn-inを省き、ladder摂動が多様性を維持します。",
                    BODY_SMALL,
                ),
                p(
                    "平均次数5の非常に疎な構造では、不一致クラスタが局所的に形成されやすく、"
                    "ICMが働きやすい条件です。CIM ladderが良い盆地と温度多様性を同時に与えます。",
                    BODY_SMALL,
                ),
            ],
        ],
        colWidths=[35 * mm, 116 * mm, 116 * mm],
    )
    contrast.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 1), (0, -1), BLUE),
                ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#C8D1D8")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BACKGROUND", (1, 1), (-1, -1), PALE_GRAY),
            ]
        )
    )
    story.extend(
        [
            contrast,
            Spacer(1, 5 * mm),
            key_takeaway(
                "G22とG55の差は、主にGAのcold start性能の差です。"
                " SAの公平比較では両問題とも短い反復だけ改善し、PT-ICMの改善は両問題で再現しています。"
            ),
            PageBreak(),
        ]
    )

    # Page 6: K2000
    story.extend(
        [
            p("K2000：効果はあるが「短時間化」とはまだ言えない", H1),
            p(
                "K2000は全頂点対がつながる±1重み付き問題です。1スピン反転が1,999本の辺へ影響し、"
                "疎なG22・G55とは異なる、強く競合した探索地形になります。",
                BODY,
            ),
            Spacer(1, 4 * mm),
            data_table(
                [
                    "GA予算",
                    "cold gap",
                    "cold時間",
                    "CIM warm gap",
                    "GA時間",
                    "CIM込み総時間",
                    "判定",
                ],
                [
                    ["3世代", "90", "10.43秒", "119", "10.05秒", "26.45秒", "warm悪化"],
                    ["8世代", "87", "14.80秒", "61", "14.55秒", "30.94秒", "解は改善"],
                    ["20世代", "70", "24.80秒", "41", "24.64秒", "41.03秒", "解は改善"],
                    ["50世代", "33", "50.86秒", "35", "48.50秒", "64.89秒", "ほぼ同等"],
                ],
                [31, 31, 37, 39, 37, 48, 44],
            ),
            Spacer(1, 5 * mm),
            explanation_columns(
                [
                    (
                        "なぜ途中で改善するか",
                        "CIM seed自体はgap 823と高品質ではありませんが、連続ダイナミクスが作った"
                        "全体的な相関を含みます。Tabu Searchが局所矛盾を修復すると、8-20世代で"
                        "coldより良い盆地へ到達できます。",
                    ),
                    (
                        "局所改善余地が大きい",
                        "CIM解の正の1-flip gain合計は平均254、最大1-flip gainは平均22です。"
                        "G22の40・3、G55の45・2より大きく、後段局所探索が修復できる余地があります。",
                    ),
                    (
                        "なぜ最終差が消えるか",
                        "GAを50世代まで回すとcold gap 33、warm gap 35で同等です。"
                        "十分な世代があればcold GA自身が良い盆地を発見し、CIMの先行効果が消えます。",
                    ),
                ]
            ),
            Spacer(1, 4 * mm),
            key_takeaway(
                "K2000では8-20世代の解品質は改善しましたが、CIM生成16.39秒を加えると総時間は長くなります。"
                " 現時点の結論は「収束補助」であり、「wall-clock高速化」ではありません。"
            ),
            Spacer(1, 2 * mm),
            p(
                "補足：K2000でのCIM warm PT-ICMは未評価です。密グラフではICMクラスタが巨大化しやすく、"
                "cold PT-ICM自体も弱いため、G22・G55の結果をそのまま外挿できません。",
                BODY_SMALL,
            ),
            PageBreak(),
        ]
    )

    # Page 7: tuning disclosure
    story.extend(
        [
            p("パラメータ調整の範囲：warm-start固有部分はほぼ未調整", H1),
            p(
                "今回の実験は「ソルバ本体の既存調整値」と「新しいwarm-start設定」を組み合わせています。"
                "warm-start固有設定についてOptunaや網羅探索は行っていません。",
                BODY,
            ),
            Spacer(1, 4 * mm),
            data_table(
                ["層", "今回使用した設定", "調整状況", "解釈"],
                [
                    [
                        "ソルバ本体",
                        "CIM・GA・SA・PT-ICMのデータセット別PARAMS",
                        "過去のOptuna値を読込",
                        "cold baselineは比較的強く調整済み",
                    ],
                    [
                        "CIM seed",
                        "600 rounds",
                        "未調整・1点のみ",
                        "品質と生成時間の最適点ではない",
                    ],
                    [
                        "GA warm",
                        "集団の1個体だけCIM解に置換",
                        "未調整",
                        "注入個体数・世代・多様性は未探索",
                    ],
                    [
                        "SA matched",
                        "coldと同じデータセット別Optuna温度",
                        "warm用の再調整なし",
                        "温度・反復・seedを固定し初期解だけ比較",
                    ],
                    [
                        "PT single",
                        "最低温Aレプリカへ1解",
                        "手設計・1方式",
                        "注入位置・A/B割当は未探索",
                    ],
                    [
                        "PT ladder",
                        "最高温で反転率0.5まで線形増加",
                        "0.5の1点のみ",
                        "反転率・曲線形状は未調整",
                    ],
                    [
                        "計算予算",
                        "GA 3/8/20、SA 1M/3M/10M、PT 30/80/200",
                        "性能曲線の測定点",
                        "パラメータ最適化ではない",
                    ],
                ],
                [39, 82, 61, 85],
            ),
            Spacer(1, 5 * mm),
            explanation_columns(
                [
                    (
                        "今回言えること",
                        "warm固有設定をほぼ調整していない状態でも、PT ladderはG22・G55で"
                        "一貫した改善を示しました。これは次の本格調整へ進む根拠になります。",
                    ),
                    (
                        "まだ言えないこと",
                        "今回の設定がwarm-startに最適とは言えません。SAはcold向け温度をそのまま"
                        "warmにも適用した公平比較であり、warm専用温度を調整すれば結果が変わる可能性があります。",
                    ),
                    (
                        "次に優先する調整",
                        "1) SA warm専用温度、2) CIM rounds、3) PT反転率と注入位置、"
                        "4) GA注入個体数を訓練問題で調整し、別問題で検証します。",
                    ),
                ]
            ),
            Spacer(1, 5 * mm),
            key_takeaway(
                "要約：ベースソルバは既存Optuna値を使用しましたが、"
                "CIM warm-startの新規パラメータはほぼチューニングしていません。"
            ),
            PageBreak(),
        ]
    )

    # Page 8: matched SA ablation graph
    story.extend(
        chart_header(
            "SAの公平比較：初期スピン配置だけを変更",
            "G22・G55各16 trial。同じ温度・反復数・seedを使い、warm側だけCIM解から開始しました。",
        )
    )
    story.extend(
        [
            chart_image_width(SA_MATCHED_GRAPH, 210),
            Spacer(1, 2.5 * mm),
            key_takeaway(
                "右列の点と95%信頼区間が0より上ならCIM初期解が有効です。"
                " G22・G55とも10万・30万反復で明確に改善し、100万反復以降は差が消えます。"
            ),
            PageBreak(),
        ]
    )

    # Page 9: matched SA explanation
    story.extend(
        [
            p("SA図の読み方と考察", H1),
            p(
                "8ページの図は、以前の温度が異なる比較を置き換えるために実施したmatched ablationです。"
                "初期スピン配置以外の主要条件を揃え、CIM初期解の因果効果を分離しました。",
                BODY,
            ),
            Spacer(1, 4 * mm),
        ]
    )
    story.append(
        explanation_columns(
            [
                (
                    "左列：最良gapと総時間",
                    "灰色がランダム初期化、赤がCIM初期化です。縦軸は下ほど良く、"
                    "横軸はCIM生成を含む総時間です。赤はCIM生成時間の分だけ右から始まります。"
                    "長時間では両者が同じ最良gapへ近づきます。",
                ),
                (
                    "右列：初期解の平均効果",
                    "縦軸は各seedのwarm cut−cold cutの平均で、0より上ならwarm有利です。"
                    "縦線は95% bootstrap信頼区間。区間が0をまたぐ点は、"
                    "<b>今回の16試行では差が明確でない</b>と読みます。",
                ),
                (
                    "得られた結論",
                    "G22は10万反復で+111.2（15/16勝）、30万で+22.2（13/16勝）。"
                    "G55は+217.5、+70.1で両方16/16勝です。"
                    "100万反復以降は両問題とも非有意で、初期解の優位が薄れました。",
                ),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 5 * mm),
            data_table(
                ["問題", "10万反復", "30万反復", "100万反復以降", "解釈"],
                [
                    ["G22", "+111.2、15/16勝", "+22.2、13/16勝", "全点で非有意", "短期のみ有効"],
                    ["G55", "+217.5、16/16勝", "+70.1、16/16勝", "全点で非有意", "短期のみ有効"],
                ],
                [28, 53, 53, 61, 72],
            ),
            Spacer(1, 5 * mm),
            key_takeaway(
                "SAへのCIM初期解は、短い探索予算で立ち上がりを確実に改善します。"
                " ただしCIM生成時間を含めると、単純なwall-clock優位とは限らず、"
                "100万反復以降はcold SAが追いつくため、長時間性能の改善効果は確認できません。"
            ),
            Spacer(1, 1.5 * mm),
            p(
                "注：cold初期化で消費する乱数個数をwarm側でも空消費し、提案乱数の開始位置を揃えました。"
                "受理判定の分岐後は状態が異なるため乱数列も分岐し得ます。p値は対応ありWilcoxon検定、"
                "信頼区間は対応差平均のbootstrapです。",
                BODY_SMALL,
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build_pdf()
    print(OUTPUT)
