"""Create Japanese figures/PDF for all-CIM GA initial populations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reporting.make_sa_ga_matched_report_pdf import (
    BODY,
    BODY_SMALL,
    FONT,
    H1,
    PALE_BLUE,
    PALE_GRAY,
    PALE_GREEN,
    REPORT_MID,
    TITLE,
    SUBTITLE,
    callout,
    chart,
    configure_matplotlib,
    data_table,
    p,
    three_boxes,
)


RESULT_DIR = (
    ROOT
    / "results"
    / "2026-07-26"
    / "cim_warmstart"
    / "ga_full_cim_population_v1_G22_G55_nt16"
)
FIG_DIR = RESULT_DIR / "report_figures"
OUTPUT = ROOT / "output" / "pdf" / "ga_full_cim_population_comparison_ja.pdf"

NAVY = "#17324d"
MID = "#596773"
COLD = "#64717a"
SINGLE = "#d4474f"
FULL = "#2c6e9b"
GREEN = "#147d64"
GRID = "#d8e0e5"


def load_results() -> dict:
    return json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))


def labels(points):
    return [str(point["budget"]) for point in points]


def quality_figure(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), dpi=200)
    for ax, dataset in zip(axes, ("G22", "G55")):
        data = results["datasets"][dataset]
        points = data["points"]
        x = np.arange(len(points))
        series = {
            "cold": np.array([point["cold"]["gap_best"] for point in points]),
            "single": np.array([point["single"]["gap_best"] for point in points]),
            "full": np.array([point["full"]["gap_best"] for point in points]),
        }
        for i in range(len(points)):
            values = [series[name][i] for name in ("cold", "single", "full")]
            ax.plot(
                [i, i],
                [min(values), max(values)],
                color=GRID,
                lw=2,
                zorder=1,
            )
        ax.scatter(
            x,
            series["cold"],
            s=100,
            facecolors="white",
            edgecolors=COLD,
            linewidths=2.4,
            marker="o",
            label="cold：全個体random",
            zorder=2,
        )
        ax.scatter(
            x,
            series["single"],
            s=62,
            color=SINGLE,
            marker="D",
            label="CIM×1個体",
            zorder=3,
        )
        ax.scatter(
            x,
            series["full"],
            s=78,
            color=FULL,
            marker="P",
            label=f"CIM×全{data['pop_size']}個体",
            zorder=4,
        )
        ymax = max(values.max() for values in series.values())
        ax.set_yscale("symlog", linthresh=1)
        ticks = [0, 1, 3, 10, 30, 100, 300]
        ax.set_yticks([tick for tick in ticks if tick <= ymax * 1.35])
        ax.set_ylim(-0.2, ymax * 1.28)
        ax.set_xticks(x, labels(points))
        ax.set_xlabel("GA世代数")
        ax.set_ylabel(f"BKSとの差（0が最良、BKS={data['bks']}）")
        ax.set_title(f"{dataset}：各予算で独立に16試行")
        ax.grid(True, axis="y", color=GRID, ls=":", lw=1)
        ax.legend(loc="upper right")
    fig.suptitle(
        "GA初期集団：random・CIM×1・CIM×全個体",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def effect_figure(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), dpi=200)
    for ax, dataset in zip(axes, ("G22", "G55")):
        points = results["datasets"][dataset]["points"]
        x = np.arange(len(points), dtype=float)
        comparisons = [
            ("full_vs_cold", "全CIM - cold", FULL, -0.10, "o"),
            ("full_vs_single", "全CIM - CIM×1", "#8a5ca8", 0.10, "D"),
        ]
        ax.axhspan(0, 1, transform=ax.get_yaxis_transform(), color="#eff8f5")
        ax.axhspan(-1, 0, transform=ax.get_yaxis_transform(), color="#fff2f2")
        for key, label, color, offset, marker in comparisons:
            stats_list = [point["paired"][key] for point in points]
            means = np.array([item["delta_mean"] for item in stats_list])
            cis = np.array([item["delta_mean_ci95"] for item in stats_list])
            yerr = np.vstack([means - cis[:, 0], cis[:, 1] - means])
            ax.errorbar(
                x + offset,
                means,
                yerr=yerr,
                fmt=marker,
                ms=7,
                color=color,
                ecolor=color,
                elinewidth=1.5,
                capsize=4,
                label=label,
                zorder=2,
            )
        ax.axhline(0, color="#333333", lw=1.2)
        ax.set_xticks(x, labels(points))
        ax.set_xlabel("GA世代数")
        ax.set_ylabel("対応差平均（cut改善量）")
        ax.set_title(f"{dataset}：全CIM集団の追加効果")
        ax.grid(True, axis="y", color=GRID, ls=":", lw=1)
        ax.legend(loc="upper right")
    fig.suptitle(
        "全CIM初期集団の平均効果と95%信頼区間",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def time_quality_figure(results: dict, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2), dpi=200)
    styles = {
        "cold": ("cold", COLD, "o"),
        "single": ("CIM×1", SINGLE, "D"),
        "full": ("CIM×全個体", FULL, "P"),
    }
    for col, dataset in enumerate(("G22", "G55")):
        data = results["datasets"][dataset]
        points = data["points"]
        ymax = max(
            point[condition]["gap_best"]
            for point in points
            for condition in ("cold", "single", "full")
        )
        for row, (time_key, row_title) in enumerate(
            (
                ("refiner_time", "CIM生成を除外"),
                ("total_time", "CIM生成を含む総時間"),
            )
        ):
            ax = axes[row, col]
            for condition, (label, color, marker) in styles.items():
                times = np.array([point[condition][time_key] for point in points])
                gaps = np.array([point[condition]["gap_best"] for point in points])
                ax.plot(
                    times,
                    gaps,
                    color=color,
                    marker=marker,
                    ms=6,
                    lw=1.5,
                    label=label,
                )
            ax.set_yscale("symlog", linthresh=1)
            ticks = [0, 1, 3, 10, 30, 100, 300]
            ax.set_yticks([tick for tick in ticks if tick <= ymax * 1.35])
            ax.set_ylim(-0.2, ymax * 1.28)
            ax.set_xlabel("16試行バッチの実測時間（秒）")
            ax.set_ylabel(f"BKSとの差（0が最良、BKS={data['bks']}）")
            ax.set_title(f"{dataset}：{row_title}")
            ax.grid(True, color=GRID, ls=":", lw=1)
            if row == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "時間で見る解品質（各線の点は左から0 → 1 → 3 → 8 → 20 → 50世代）",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_figures(results: dict) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "quality": FIG_DIR / "ga_full_cim_quality_ja.png",
        "effect": FIG_DIR / "ga_full_cim_effect_ja.png",
        "time_quality": FIG_DIR / "ga_full_cim_time_quality_ja.png",
    }
    quality_figure(results, paths["quality"])
    effect_figure(results, paths["effect"])
    time_quality_figure(results, paths["time_quality"])
    return paths


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d8dee4"))
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(REPORT_MID)
    canvas.drawString(15 * mm, 6.5 * mm, "GA全CIM初期集団比較")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, str(doc.page))
    canvas.restoreState()


def timing_rows(results: dict) -> list[list[str]]:
    rows = []
    for dataset in ("G22", "G55"):
        data = results["datasets"][dataset]
        rows.append(
            [
                dataset,
                str(data["pop_size"]),
                f"{data['cim_single']['num_solutions']}解 / "
                f"{data['cim_single']['time']:.3f}秒",
                f"{data['cim_full']['num_solutions']}解 / "
                f"{data['cim_full']['time']:.3f}秒",
                f"{data['cim_full']['diversity']['mean']:.3f}",
            ]
        )
    return rows


def result_rows(results: dict, dataset: str) -> list[list[str]]:
    rows = []
    for point in results["datasets"][dataset]["points"]:
        fs = point["paired"]["full_vs_single"]
        rows.append(
            [
                str(point["budget"]),
                f"{point['cold']['gap_best']:.0f}",
                f"{point['single']['gap_best']:.0f}",
                f"{point['full']['gap_best']:.0f}",
                f"{fs['delta_mean']:+.1f}",
                f"{fs['wilcoxon_p']:.3g}",
            ]
        )
    return rows


def time_rows(results: dict, dataset: str) -> list[list[str]]:
    rows = []
    for point in results["datasets"][dataset]["points"]:
        rows.append(
            [
                str(point["budget"]),
                f"{point['cold']['refiner_time']:.2f}",
                f"{point['single']['refiner_time']:.2f}",
                f"{point['full']['refiner_time']:.2f}",
                f"{point['cold']['total_time']:.2f}",
                f"{point['single']['total_time']:.2f}",
                f"{point['full']['total_time']:.2f}",
            ]
        )
    return rows


def build_pdf(results: dict, figures: dict[str, Path]) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="GA全CIM初期集団比較",
        author="cim-cac-maxcut-g22 project",
    )
    story = []

    story.extend(
        [
            Spacer(1, 5 * mm),
            p("GA初期集団を全てCIM解にした比較", TITLE),
            Spacer(1, 2 * mm),
            p(
                "cold、CIM×1個体、異なるCIM seedで作った全CIM集団を同一GA条件で比較",
                SUBTITLE,
            ),
            Spacer(1, 7 * mm),
            three_boxes(
                [
                    (
                        "cold",
                        "初期集団の全個体をrandom生成します。従来のGA baselineです。",
                        PALE_GRAY,
                    ),
                    (
                        "CIM×1個体",
                        "集団の1個体目だけをCIM解へ置換し、残りはcoldと同じrandom個体です。",
                        PALE_GREEN,
                    ),
                    (
                        "CIM×全個体",
                        "各trialの全個体を異なるCIM seedで生成します。G22は15個体、G55は6個体です。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 6 * mm),
            p("CIM初期集団の生成コストと多様性", H1),
            Spacer(1, 2 * mm),
            data_table(
                ["問題", "GA集団", "CIM×1用", "全CIM用", "全CIM平均距離"],
                timing_rows(results),
                [34, 37, 65, 72, 59],
            ),
            Spacer(1, 5 * mm),
            three_boxes(
                [
                    (
                        "乱数の公平性",
                        "GA世代数・GA/TS設定・GA seedを固定し、cold初期化で消費する乱数もwarm側で空消費しました。",
                        None,
                    ),
                    (
                        "CIM seed",
                        "集団内の全CIM個体は異なるseedです。個体0はCIM×1条件と同じseedを使用しています。",
                        None,
                    ),
                    (
                        "多様性",
                        "全体反転を考慮した平均Hamming距離はG22 0.330、G55 0.356。単なる同一解の複製ではありません。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 5 * mm),
            callout(
                "全CIM集団は解品質をさらに改善しますが、CIM生成時間はG22で約2.95秒、"
                "G55で約1.95秒かかります。品質と総時間を分けて判断します。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("最良gap：全CIM集団はどこまで改善するか", H1),
            Spacer(1, 2 * mm),
            chart(figures["quality"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "G22",
                        "全CIMは0～20世代で最良gap 1。CIM×1のgap 3～1より早く高品質化します。"
                        "50世代ではcoldがBKSへ到達し差が消えます。",
                        None,
                    ),
                    (
                        "G55",
                        "0世代でcold 314、CIM×1 85、全CIM 82。"
                        "50世代では103、68、66で、全CIMが最良値をわずかに改善します。",
                        None,
                    ),
                    (
                        "読み方",
                        "同じ縦列の3記号を比較します。各世代は独立した16試行で、"
                        "世代間を結んだ単一runの軌跡ではありません。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "全CIM集団は初期～中期の品質を改善します。長くGAを回すほど追加効果は小さくなります。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("時間軸で見る解品質", H1),
            Spacer(1, 1 * mm),
            chart(figures["time_quality"], width_mm=220),
            Spacer(1, 2 * mm),
            three_boxes(
                [
                    (
                        "上段：CIM生成を除外",
                        "GA側の実行時間だけで比較します。初期解を既に持っている場合の性能に対応します。",
                        PALE_BLUE,
                    ),
                    (
                        "下段：CIM生成を含む",
                        "CIM初期解の作成からGA終了までの総時間です。実運用の費用対効果はこちらで判断します。",
                        PALE_GREEN,
                    ),
                    (
                        "点と世代数",
                        "各線の点は左から0、1、3、8、20、50世代です。時間は各条件16試行をまとめて実行した実測値です。",
                        PALE_GRAY,
                    ),
                ]
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("世代数から実測秒への対応表", H1),
            Spacer(1, 3 * mm),
            Table(
                [
                    [
                        data_table(
                            [
                                "G22世代",
                                "GA秒 cold",
                                "GA秒 ×1",
                                "GA秒 全",
                                "総秒 cold",
                                "総秒 ×1",
                                "総秒 全",
                            ],
                            time_rows(results, "G22"),
                            [15, 18, 18, 18, 18, 18, 18],
                        ),
                        data_table(
                            [
                                "G55世代",
                                "GA秒 cold",
                                "GA秒 ×1",
                                "GA秒 全",
                                "総秒 cold",
                                "総秒 ×1",
                                "総秒 全",
                            ],
                            time_rows(results, "G55"),
                            [15, 18, 18, 18, 18, 18, 18],
                        ),
                    ]
                ],
                colWidths=[132 * mm, 132 * mm],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 1 * mm),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
                    ]
                ),
            ),
            Spacer(1, 6 * mm),
            three_boxes(
                [
                    (
                        "GA秒",
                        "CIM生成時間を含まないGA側の実測時間です。初期集団が準備済みという前提の比較です。",
                        PALE_BLUE,
                    ),
                    (
                        "総秒",
                        "coldはGA秒と同じです。CIM×1と全CIMは、それぞれのCIM生成時間をGA秒へ加えています。",
                        PALE_GREEN,
                    ),
                    (
                        "測定単位",
                        "表の秒数は16試行バッチ全体のwall-clockです。計算機・並列化・負荷によって絶対値は変わります。",
                        PALE_GRAY,
                    ),
                ]
            ),
            Spacer(1, 5 * mm),
            callout(
                "時間制約がある場合は、世代数ではなく「総秒」列でcold・CIM×1・全CIMを比較してください。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("平均効果：全CIMはCIM×1より良いか", H1),
            Spacer(1, 2 * mm),
            chart(figures["effect"]),
            Spacer(1, 3 * mm),
            three_boxes(
                [
                    (
                        "青：全CIM - cold",
                        "両問題とも0～20世代で大きく正です。G55では50世代でも平均+25.9でcoldより有意に良好です。",
                        None,
                    ),
                    (
                        "紫：全CIM - CIM×1",
                        "G22は0～20世代で+16.5から+4.6。"
                        "G55は0～8世代で約+15～+13ですが、20世代以降は差が不明確です。",
                        None,
                    ),
                    (
                        "95%信頼区間",
                        "縦線が0をまたぐ場合、16試行では追加効果が明確ではありません。"
                        "G55の50世代は平均−0.8で、CIM×1と実質同等です。",
                        PALE_BLUE,
                    ),
                ]
            ),
            Spacer(1, 3 * mm),
            callout(
                "全CIM化の上乗せ効果は短いGAで最大です。50世代ではCIM×1との差がほぼ消えます。"
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            p("数値結果と実運用上の判断", H1),
            Spacer(1, 3 * mm),
            Table(
                [
                    [
                        data_table(
                            ["G22世代", "cold gap", "×1 gap", "全CIM gap", "全-×1平均", "p"],
                            result_rows(results, "G22"),
                            [17, 19, 19, 22, 27, 20],
                        ),
                        data_table(
                            ["G55世代", "cold gap", "×1 gap", "全CIM gap", "全-×1平均", "p"],
                            result_rows(results, "G55"),
                            [17, 19, 19, 22, 27, 20],
                        ),
                    ]
                ],
                colWidths=[132 * mm, 132 * mm],
                style=TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 1 * mm),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
                    ]
                ),
            ),
            Spacer(1, 6 * mm),
            three_boxes(
                [
                    (
                        "解品質を最優先",
                        "短いGAでできるだけ良い解が必要なら全CIM集団が有効です。"
                        "特にG22の0～20世代、G55の0～8世代でCIM×1を上回ります。",
                        PALE_GREEN,
                    ),
                    (
                        "総時間を最優先",
                        "CIM×1の方が費用対効果は高いです。全CIMはG22で約2.75秒、"
                        "G55で約1.61秒の追加CIM生成時間が必要です。",
                        PALE_BLUE,
                    ),
                    (
                        "推奨する次の実験",
                        "全個体か1個体かの二択ではなく、CIM個体数を1、25%、50%、100%と変え、"
                        "品質と総時間のPareto最適点を探索します。",
                        PALE_GRAY,
                    ),
                ]
            ),
            Spacer(1, 5 * mm),
            callout(
                "結論：全CIM集団は短期品質を確実に押し上げますが、長期ではCIM×1との差が消えます。"
                " 実用上はCIM個体の投入比率を調整する価値があります。"
            ),
            Spacer(1, 2 * mm),
            p(
                "統計条件：G22・G55各16試行、同一GA seedの対応比較。"
                "エラーバーは対応差平均の95% bootstrap信頼区間、p値は対応ありWilcoxon検定。"
                "CIMは600 roundsで、warm固有パラメータの追加調整は行っていません。",
                BODY_SMALL,
            ),
        ]
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    results = load_results()
    configure_matplotlib()
    figures = make_figures(results)
    build_pdf(results, figures)
    print(OUTPUT)


if __name__ == "__main__":
    main()
