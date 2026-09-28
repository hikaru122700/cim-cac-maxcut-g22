"""Create clear Japanese SA/GA matched-ablation figures and an explanation PDF."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SA_DIR = (
    ROOT
    / "results"
    / "2026-07-26"
    / "cim_warmstart"
    / "sa_matched_v2_G22_G55_nt16_to300m"
)
GA_DIR = (
    ROOT
    / "results"
    / "2026-07-26"
    / "cim_warmstart"
    / "ga_matched_v1_G22_G55_nt16"
)
FIG_DIR = (
    ROOT
    / "results"
    / "2026-07-26"
    / "cim_warmstart"
    / "sa_ga_matched_report"
)
OUTPUT = ROOT / "output" / "pdf" / "sa_ga_matched_comparison_ja.pdf"

FONT_REGULAR_PATH = Path(r"C:\Windows\Fonts\YuGothM.ttc")
FONT_BOLD_PATH = Path(r"C:\Windows\Fonts\YuGothB.ttc")

NAVY = "#17324d"
MID = "#596773"
COLD = "#64717a"
WARM = "#d4474f"
GREEN = "#147d64"
GRID = "#d8e0e5"
PALE_BLUE = colors.HexColor("#eaf3f8")
PALE_GREEN = colors.HexColor("#e8f6f1")
PALE_GRAY = colors.HexColor("#f3f5f7")
REPORT_NAVY = colors.HexColor(NAVY)
REPORT_GREEN = colors.HexColor(GREEN)
REPORT_DARK = colors.HexColor("#27333d")
REPORT_MID = colors.HexColor(MID)


def load_results() -> tuple[dict, dict]:
    sa = json.loads((SA_DIR / "results.json").read_text(encoding="utf-8"))
    ga = json.loads((GA_DIR / "results.json").read_text(encoding="utf-8"))
    return sa, ga


def configure_matplotlib() -> None:
    fm.fontManager.addfont(str(FONT_REGULAR_PATH))
    font_name = fm.FontProperties(fname=str(FONT_REGULAR_PATH)).get_name()
    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def budget_labels(points: list[dict], solver: str) -> list[str]:
    labels = []
    for point in points:
        value = point["budget"]
        if solver == "SA":
            labels.append(
                f"{value // 1_000_000}M"
                if value >= 1_000_000
                else f"{value // 1_000}k"
            )
        else:
            labels.append(str(value))
    return labels


def quality_figure(results: dict, solver: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), dpi=200)
    for ax, dataset in zip(axes, ("G22", "G55")):
        data = results["datasets"][dataset]
        points = data["points"]
        x = np.arange(len(points))
        cold_gap = np.array([point["cold"]["gap_best"] for point in points])
        warm_gap = np.array([point["warm"]["gap_best"] for point in points])

        for xi, yc, yw in zip(x, cold_gap, warm_gap):
            ax.plot([xi, xi], [yc, yw], color=GRID, lw=2, zorder=1)
        ax.scatter(
            x,
            cold_gap,
            s=90,
            facecolors="white",
            edgecolors=COLD,
            linewidths=2.4,
            marker="o",
            label="cold（ランダム初期化）",
            zorder=2,
        )
        ax.scatter(
            x,
            warm_gap,
            s=58,
            color=WARM,
            marker="D",
            label="CIM warm",
            zorder=3,
        )
        ax.set_yscale("symlog", linthresh=1)
        ymax = max(float(cold_gap.max()), float(warm_gap.max()))
        ticks = [0, 1, 3, 10, 30, 100, 300]
        ax.set_yticks([tick for tick in ticks if tick <= ymax * 1.35])
        ax.set_ylim(-0.2, ymax * 1.28)
        ax.set_xticks(x, budget_labels(points, solver))
        ax.set_xlabel("SA反復数" if solver == "SA" else "GA世代数")
        ax.set_ylabel(f"BKSとの差（0が最良、BKS={data['bks']}）")
        ax.set_title(f"{dataset}：各予算で独立に16試行")
        ax.grid(True, axis="y", which="major", color=GRID, ls=":", lw=1)
        ax.legend(loc="upper right")
        ax.text(
            0.01,
            0.02,
            "○と◆が同じ縦列 = 同じ計算予算",
            transform=ax.transAxes,
            color=MID,
            fontsize=8.5,
        )

    fig.suptitle(
        f"{solver}：同じ計算予算での最良gap比較",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def effect_figure(results: dict, solver: str, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), dpi=200)
    for ax, dataset in zip(axes, ("G22", "G55")):
        points = results["datasets"][dataset]["points"]
        x = np.arange(len(points))
        means = np.array([point["paired"]["delta_mean"] for point in points])
        cis = np.array([point["paired"]["delta_mean_ci95"] for point in points])
        yerr = np.vstack([means - cis[:, 0], cis[:, 1] - means])
        significant = np.array(
            [
                point["paired"]["wilcoxon_p"] < 0.05 and mean > 0
                for point, mean in zip(points, means)
            ]
        )
        point_colors = np.where(significant, GREEN, MID)

        ax.axhspan(0, max(cis[:, 1].max(), 1) * 1.12, color="#eff8f5", zorder=0)
        if cis[:, 0].min() < 0:
            ax.axhspan(cis[:, 0].min() * 1.25, 0, color="#fff2f2", zorder=0)
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="none",
            ecolor=MID,
            elinewidth=1.7,
            capsize=5,
            zorder=1,
        )
        ax.scatter(x, means, s=75, c=point_colors, zorder=2)
        ax.axhline(0, color="#333333", lw=1.2)
        for xi, yi, point in zip(x, means, points):
            paired = point["paired"]
            ax.annotate(
                f"{paired['wins']}/{paired['ties']}/{paired['losses']}",
                (xi, yi),
                xytext=(0, 10 if yi >= 0 else -16),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color="#27333d",
            )

        ax.set_xticks(x, budget_labels(points, solver))
        ax.set_xlabel("SA反復数" if solver == "SA" else "GA世代数")
        ax.set_ylabel("平均cut改善量（warm - cold）")
        ax.set_title(f"{dataset}：対応差平均と95%信頼区間")
        ax.grid(True, axis="y", color=GRID, ls=":", lw=1)
        ax.text(
            0.01,
            0.98,
            "緑：p < 0.05でwarm有利\n数字：勝 / 同点 / 負",
            transform=ax.transAxes,
            va="top",
            fontsize=8.5,
            color=MID,
        )

    fig.suptitle(
        f"{solver}：CIM初期解の平均効果",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_figures(sa: dict, ga: dict) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "sa_quality": FIG_DIR / "sa_quality_budget_aligned_ja.png",
        "sa_effect": FIG_DIR / "sa_effect_paired_ja.png",
        "ga_quality": FIG_DIR / "ga_quality_budget_aligned_ja.png",
        "ga_effect": FIG_DIR / "ga_effect_paired_ja.png",
    }
    quality_figure(sa, "SA", paths["sa_quality"])
    effect_figure(sa, "SA", paths["sa_effect"])
    quality_figure(ga, "GA", paths["ga_quality"])
    effect_figure(ga, "GA", paths["ga_effect"])
    return paths


def register_report_fonts() -> tuple[str, str]:
    pdfmetrics.registerFont(
        TTFont("YuGothic", str(FONT_REGULAR_PATH), subfontIndex=0)
    )
    pdfmetrics.registerFont(
        TTFont("YuGothic-Bold", str(FONT_BOLD_PATH), subfontIndex=0)
    )
    return "YuGothic", "YuGothic-Bold"


FONT, FONT_BOLD = register_report_fonts()
BASE = getSampleStyleSheet()
TITLE = ParagraphStyle(
    "TitleJa",
    parent=BASE["Title"],
    fontName=FONT_BOLD,
    fontSize=23,
    leading=30,
    textColor=REPORT_NAVY,
    alignment=TA_CENTER,
)
SUBTITLE = ParagraphStyle(
    "SubtitleJa",
    parent=BASE["Normal"],
    fontName=FONT,
    fontSize=10.5,
    leading=15,
    textColor=REPORT_MID,
    alignment=TA_CENTER,
)
H1 = ParagraphStyle(
    "H1Ja",
    parent=BASE["Heading1"],
    fontName=FONT_BOLD,
    fontSize=18,
    leading=23,
    textColor=REPORT_NAVY,
)
BODY = ParagraphStyle(
    "BodyJa",
    parent=BASE["BodyText"],
    fontName=FONT,
    fontSize=9.2,
    leading=13.5,
    textColor=REPORT_DARK,
    alignment=TA_LEFT,
)
BODY_SMALL = ParagraphStyle(
    "BodySmallJa",
    parent=BODY,
    fontSize=8,
    leading=11,
)
BOX_TITLE = ParagraphStyle(
    "BoxTitleJa",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=10.5,
    leading=14,
    textColor=REPORT_NAVY,
)
CALLOUT = ParagraphStyle(
    "CalloutJa",
    parent=BODY,
    fontName=FONT_BOLD,
    fontSize=10.5,
    leading=15,
    textColor=REPORT_GREEN,
    alignment=TA_CENTER,
)
TABLE_HEAD = ParagraphStyle(
    "TableHeadJa",
    parent=BODY_SMALL,
    fontName=FONT_BOLD,
    textColor=colors.white,
    alignment=TA_CENTER,
)
TABLE_CELL = ParagraphStyle(
    "TableCellJa",
    parent=BODY_SMALL,
    alignment=TA_CENTER,
)


def p(text: str, style=BODY) -> Paragraph:
    return Paragraph(text, style)


def info_box(title: str, text: str, background=PALE_GRAY) -> Table:
    box = Table(
        [[p(title, BOX_TITLE)], [p(text, BODY)]],
        colWidths=[82 * mm],
    )
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#cad6de")),
                ("LEFTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5 * mm),
            ]
        )
    )
    return box


def three_boxes(items: list[tuple[str, str, colors.Color | None]]) -> Table:
    cells = [
        info_box(title, text, background or PALE_GRAY)
        for title, text, background in items
    ]
    table = Table([cells], colWidths=[87 * mm] * 3)
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


def callout(text: str) -> Table:
    table = Table([[p(text, CALLOUT)]], colWidths=[267 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE_GREEN),
                ("BOX", (0, 0), (-1, -1), 1, REPORT_GREEN),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 3 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
            ]
        )
    )
    return table


def data_table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    content = [[p(value, TABLE_HEAD) for value in headers]]
    content.extend([[p(str(value), TABLE_CELL) for value in row] for row in rows])
    table = Table(content, colWidths=[width * mm for width in widths])
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), REPORT_NAVY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8d1d8")),
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


def chart(path: Path, width_mm: float = 245) -> Image:
    probe = Image(str(path))
    width = width_mm * mm
    height = width * probe.imageHeight / probe.imageWidth
    return Image(str(path), width=width, height=height, hAlign="CENTER")


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d8dee4"))
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(REPORT_MID)
    canvas.drawString(15 * mm, 6.5 * mm, "CIM warm-start：SA・GA公平比較")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, str(doc.page))
    canvas.restoreState()


def build_pdf(figures: dict[str, Path]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="CIM warm-start SA・GA公平比較",
        author="cim-cac-maxcut-g22 project",
    )
    story = []

    story.extend(
        [
            Spacer(1, 5 * mm),
            p("SA・GAのCIM warm-start公平比較", TITLE),
            Spacer(1, 2 * mm),
            p(
                "「同じ予算なのに点がずれる」「線が単一試行の推移に見える」という問題を解消した読み方資料",
                SUBTITLE,
            ),
            Spacer(1, 7 * mm),
            three_boxes(
                [
                    (
                        "1. 同じ縦列を比較",
                        "横軸はSA反復数またはGA世代数です。同じ列の○（cold）と◆（CIM warm）を比較します。"
                        "縦の薄い線は同一予算の組を示します。",
                        PALE_BLUE,
                    ),
                    (
                        "2. 下ほど良い",
                        "縦軸は既知最良値BKSとの差です。0がBKS到達です。"
                        "◆が○より下なら、同じ予算でCIM初期解が有利です。",
                        PALE_GREEN,
                    ),
                    (
                        "3. 予算間は結ばない",
                        "各予算は独立した16試行です。単一runを延長した軌跡ではないため、"
                        "予算間を線で結ばず、点の戻りを収束軌跡と誤解しない表示にしました。",
                        PALE_GRAY,
                    ),
                ]
            ),
            Spacer(1, 7 * mm),
            p("なぜ以前の図では短時間にcoldしかなかったのか", H1),
            Spacer(1, 2 * mm),
            p(
                "以前のend-to-end図では、warm側だけにCIM初期解生成時間（G22約0.2秒、G55約0.3秒）を"
                "加えていたため、warmの最初の点が右へ移動しました。データ欠損ではありません。"
                "今回の主図は初期解の効果を読むため計算予算で整列しています。実運用速度を論じる場合は、"
                "CIM生成時間込みのend-to-end評価を別途併記します。",
                BODY,
            ),
            Spacer(1, 7 * mm),
            data_table(
                ["手法", "G22で効果が明確な範囲", "G55で効果が明確な範囲", "長時間の結論"],
                [
                    ["SA", "10万・30万反復", "10万・30万反復", "100万反復以降は有意差なし"],
                    ["GA", "0・1・3世代", "0～50世代の全点", "G22は天井、G55は効果が持続"],
                ],
                [35, 72, 72, 88],
            ),
            Spacer(1, 7 * mm),
            callout(
                "主な結論：CIM初期解はSAでは短い立ち上がり、GAでは特にG55の収束を改善します。"
                " G22のGAはcold単体が早くBKS近傍へ達するため、差が短期間で消えます。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("SA：同じ反復数で最良gapを比較", H1),
            Spacer(1, 2 * mm),
            chart(figures["sa_quality"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "G22の読み方",
                        "10万反復ではgap 127 -> 76、30万では90 -> 74です。"
                        "100万以降は○と◆が交差・重複し、持続的な優位は見られません。",
                        None,
                    ),
                    (
                        "G55の読み方",
                        "10万反復ではgap 343 -> 120、30万では201 -> 120です。"
                        "100万以降は差が小さく、長時間ではほぼ同じ品質へ収束します。",
                        None,
                    ),
                    (
                        "点が戻る理由",
                        "例えばG22の1億反復でcoldがBKSへ到達しても、3億反復は別の独立実験です。"
                        "同じrunの続きではないため、best-of-16は単調になりません。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "SAではCIM初期解が短い探索の品質を改善しますが、長時間SAの最終品質は改善しません。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("SA：平均改善量と不確かさ", H1),
            Spacer(1, 2 * mm),
            chart(figures["sa_effect"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "縦軸",
                        "各seedについてwarm cut - cold cutを計算し、その平均を表示しています。"
                        "0より上がwarm有利、下がcold有利です。",
                        None,
                    ),
                    (
                        "縦線",
                        "対応差平均の95% bootstrap信頼区間です。区間が0をまたぐ点は、"
                        "今回の16試行では差が明確でないと読みます。",
                        None,
                    ),
                    (
                        "数字",
                        "勝 / 同点 / 負です。G55の10万・30万反復は16/0/0。"
                        "それ以降は勝敗が拮抗し、初期解効果が薄れています。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "有意なSA改善：G22は+111.2、+22.2。G55は+217.5、+70.1。"
                " いずれも10万・30万反復だけです。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("GA：同じ世代数で最良gapを比較", H1),
            Spacer(1, 2 * mm),
            chart(figures["ga_quality"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "公平条件",
                        "集団の1個体目だけをCIM解へ変更しました。残りの初期集団、GA・TS設定、"
                        "seed、初期化後の乱数位置を揃えています。",
                        PALE_BLUE,
                    ),
                    (
                        "G22",
                        "0世代でgap 10 -> 3、1世代で4 -> 3。3世代では最良gapは同じ2ですが、"
                        "16試行平均ではwarmが有意に良好です。",
                        None,
                    ),
                    (
                        "G55",
                        "0～8世代でwarmの最良gapは85を維持し、coldとの差が大きく残ります。"
                        "50世代でもcold 103に対しwarm 68です。",
                        None,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "GAではG22はごく短い立ち上がりだけ改善し、G55では50世代まで改善が持続します。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("GA：平均改善量とG22・G55の違い", H1),
            Spacer(1, 2 * mm),
            chart(figures["ga_effect"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "G22",
                        "0世代+13.3、1世代+11.5、3世代+7.9で有意です。"
                        "8世代以降は信頼区間が0をまたぎ、50世代では12/16が同点です。",
                        None,
                    ),
                    (
                        "G55",
                        "0世代+214.8から50世代+26.8まで、全予算で有意な改善です。"
                        "50世代でも16/16勝で、CIMの先行効果が残っています。",
                        None,
                    ),
                    (
                        "なぜ違うか",
                        "G22のcold GAは3世代でgap 2まで到達し、改善余地がほぼありません。"
                        "G55はcoldの収束が遅く、CIMが良い探索領域へ置く価値が長く残ります。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "研究上の使い分け：SA warmは短時間ブースト、GA warmはcold収束が遅い問題で有効。"
                " 総時間の高速化を主張する場合はCIM生成時間を必ず加えます。"
            ),
            Spacer(1, 2 * mm),
            p(
                "統計条件：G22・G55各16試行。同一seedの対応比較。エラーバーは対応差平均の95% bootstrap信頼区間、"
                "有意判定は対応ありWilcoxon検定（p < 0.05）。各予算点は独立実験です。",
                BODY_SMALL,
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    sa, ga = load_results()
    configure_matplotlib()
    figures = make_figures(sa, ga)
    build_pdf(figures)
    print(OUTPUT)


if __name__ == "__main__":
    main()
