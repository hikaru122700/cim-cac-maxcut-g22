"""Create the four-dataset all-CIM-population GA plot and Japanese PDF guide."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reporting.make_sa_ga_matched_report_pdf import (
    BODY,
    BODY_SMALL,
    H1,
    PALE_BLUE,
    PALE_GRAY,
    PALE_GREEN,
    callout,
    chart,
    configure_matplotlib,
    data_table,
    footer,
    p,
    three_boxes,
)

RESULT_DIR = (
    ROOT
    / "results"
    / "2026-07-26"
    / "cim_warmstart"
    / "ga_full_population_until_nonsig_G22_G55_G70_K2000_nt16"
)
FIGURE = RESULT_DIR / "ga_bks_gap_vs_generations_ja.png"
OUTPUT = ROOT / "output" / "pdf" / "ga_full_population_gap_four_datasets_ja.pdf"

NAVY = "#17324d"
COLD = "#687780"
WARM = "#d24a43"
GRID = "#d7e0e6"
DATASETS = ("G22", "G55", "G70", "K2000")


def bootstrap_ci(values: np.ndarray, seed: int, reps: int = 4000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def load():
    results = json.loads((RESULT_DIR / "results.json").read_text(encoding="utf-8"))
    raw = dict(np.load(RESULT_DIR / "cuts.npz"))
    return results, raw


def make_figure(results: dict, raw: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), dpi=200)
    for ax, dataset in zip(axes.flat, DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        generations = np.asarray([point["generations"] for point in points], dtype=float)
        for condition, label, color, marker in (
            ("cold", "random初期集団", COLD, "o"),
            ("warm", "全CIM初期集団", WARM, "s"),
        ):
            means = []
            intervals = []
            for point in points:
                key = f"{dataset}_{point['generations']}_{condition}"
                gaps = data["bks"] - np.asarray(raw[key], dtype=float)
                means.append(float(gaps.mean()))
                intervals.append(
                    bootstrap_ci(
                        gaps,
                        seed=3000 + point["generations"] + (0 if condition == "cold" else 101),
                    )
                )
            means = np.asarray(means)
            intervals = np.asarray(intervals)
            positive_means = np.maximum(means, 0.25)
            low = np.maximum(intervals[:, 0], 0.25)
            high = np.maximum(intervals[:, 1], positive_means)
            yerr = np.vstack([positive_means - low, high - positive_means])
            ax.errorbar(
                generations,
                positive_means,
                yerr=yerr,
                color=color,
                marker=marker,
                ms=7,
                lw=2.3,
                capsize=4,
                label=label,
            )
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("GA世代数", fontsize=20, labelpad=8)
        ax.set_ylabel("平均BKS gap（小さいほど良い）", fontsize=20, labelpad=8)
        ax.set_title(f"{dataset}  BKS={data['bks']}", fontsize=22, pad=12)
        ax.grid(True, which="both", color=GRID, ls=":")
        ax.set_xticks(generations, [str(int(value)) for value in generations])
        ax.tick_params(axis="both", which="major", labelsize=16)
        if len(generations) == 1:
            ax.set_xlim(-0.6, 1.6)
        ax.legend(loc="upper right", fontsize=14)
    fig.suptitle(
        "GA：random初期集団と全CIM初期集団のBKS gap",
        fontsize=28,
        fontweight="bold",
        color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def endpoint_text(data: dict) -> str:
    if data.get("extended_through_generations") is not None:
        first = data.get("first_nonsignificant_generations")
        end = data["extended_through_generations"]
        return (
            f"{first}世代で差が不明確でしたが、追加測定として{end}世代まで継続しました。"
        )
    if data.get("stop_reached"):
        return f"{data['stop_generations']}世代で16試行では差を明確に検出できなくなりました。"
    cap = data.get("practical_cap_generations", data["points"][-1]["generations"])
    return f"{cap}世代まで差が残り、実用上の計算上限に達しました。"


def table_rows(data: dict) -> list[list[str]]:
    output = []
    for point in data["points"]:
        output.append(
            [
                str(point["generations"]),
                f"{point['cold']['gap_mean']:.1f}",
                f"{point['warm']['gap_mean']:.1f}",
                f"{point['paired']['delta_mean']:+.1f}",
                f"{point['paired']['wilcoxon_p']:.3g}",
                f"{point['cold']['total_time']:.1f}",
                f"{point['warm']['total_time']:.1f}",
            ]
        )
    return output


def build_pdf(results: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=13 * mm,
        bottomMargin=15 * mm,
        title="GA 全CIM初期集団 BKS gap比較",
    )
    story = [
        p("GA：世代数を増やすと全CIM初期集団の効果はいつ消えるか", H1),
        Spacer(1, 1 * mm),
        chart(FIGURE, width_mm=218),
        Spacer(1, 2 * mm),
        callout(
            "灰色がrandom初期集団、赤が全個体を異なるseedのCIMで作った初期集団です。"
            "縦軸は下にあるほど良い結果です。"
        ),
        PageBreak(),
        p("実験条件とグラフの読み方", H1),
        Spacer(1, 3 * mm),
        three_boxes(
            [
                (
                    "縦軸",
                    "同じseedで行った16試行の平均BKS gapです。0に近いほど既知最良解に近いことを示します。",
                    PALE_GREEN,
                ),
                (
                    "横軸",
                    "GAの世代数です。0世代でも、GA実装に含まれる初期集団へのTabu Searchは実行されています。",
                    PALE_BLUE,
                ),
                (
                    "公平な比較",
                    "GAパラメータ・世代数・GA seed・初期化後の乱数位置を揃え、初期集団だけを変えました。",
                    PALE_GRAY,
                ),
            ]
        ),
        Spacer(1, 6 * mm),
        data_table(
            ["問題", "初期集団数", "最終測定", "random gap", "全CIM gap", "p"],
            [
                [
                    dataset,
                    str(results["datasets"][dataset]["pop_size"]),
                    f"{results['datasets'][dataset]['points'][-1]['generations']}世代",
                    f"{results['datasets'][dataset]['points'][-1]['cold']['gap_mean']:.1f}",
                    f"{results['datasets'][dataset]['points'][-1]['warm']['gap_mean']:.1f}",
                    f"{results['datasets'][dataset]['points'][-1]['paired']['wilcoxon_p']:.3g}",
                ]
                for dataset in DATASETS
            ],
            [34, 38, 44, 48, 48, 40],
        ),
        Spacer(1, 6 * mm),
        p(
            "各世代数は独立した16試行です。線は1回のGAが時間とともに動いた軌跡ではなく、"
            "計算量ごとの平均値を見やすく結んだものです。",
            BODY_SMALL,
        ),
        PageBreak(),
    ]

    for left, right in (("G22", "G55"), ("G70", "K2000")):
        left_data = results["datasets"][left]
        right_data = results["datasets"][right]
        story += [
            p(f"数値と実行時間：{left}・{right}", H1),
            Spacer(1, 3 * mm),
            Table(
                [
                    [
                        data_table(
                            ["世代", "random gap", "全CIM gap", "改善", "p", "random秒", "CIM総秒"],
                            table_rows(left_data),
                            [15, 20, 20, 16, 16, 18, 18],
                        ),
                        data_table(
                            ["世代", "random gap", "全CIM gap", "改善", "p", "random秒", "CIM総秒"],
                            table_rows(right_data),
                            [15, 20, 20, 16, 16, 18, 18],
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
                    (left, endpoint_text(left_data), PALE_GREEN),
                    (right, endpoint_text(right_data), PALE_BLUE),
                    (
                        "時間の意味",
                        "random秒はGAのみ、CIM総秒は初期集団全個体のCIM生成時間を含む16試行バッチ時間です。",
                        PALE_GRAY,
                    ),
                ]
            ),
            PageBreak(),
        ]

    story += [
        p("結果の考察", H1),
        Spacer(1, 3 * mm),
        three_boxes(
            [
                (
                    "G22・G55",
                    "random側が世代を重ねて追いつき、G22は50世代、G55は120世代で差が不明確になりました。",
                    PALE_GREEN,
                ),
                (
                    "G70",
                    "全CIM集団は早い段階で良い探索領域へ入り、800世代でも効果が残りました。"
                    "300から800世代では両側とも改善が小さく、停滞傾向です。",
                    PALE_BLUE,
                ),
                (
                    "K2000",
                    "0～800世代の全点で一貫したCIM効果は確認できず、50・300・800世代ではrandom側の平均gapが小さくなりました。",
                    PALE_GRAY,
                ),
            ]
        ),
        Spacer(1, 7 * mm),
        p(
            "全CIM初期集団は、単一のCIM解を複製したものではありません。各個体を異なるCIM seedで生成しています。"
            "スピン全反転を同一視した平均集団多様性はG22 0.328、G55 0.344、G70 0.371、K2000 0.441でした。",
            BODY,
        ),
        Spacer(1, 5 * mm),
        p(
            "K2000では144個のCIM解生成に82.2秒かかりました。800世代ではrandom側が364.5秒・平均gap 19.9、"
            "全CIM側はCIM生成込み443.5秒・平均gap 26.2でした。今回の条件ではCIM生成費用に見合う改善は確認できません。",
            BODY,
        ),
        Spacer(1, 6 * mm),
        callout(
            "結論：全CIM初期集団はG22・G55の立ち上がりを改善し、G70では長時間効果が残りました。"
            "K2000では800世代まで延長しても効果がなく、最終点ではrandom側が良い結果です。"
        ),
        Spacer(1, 4 * mm),
        p(
            "今回の比較用にGAパラメータの追加チューニングはしていません。既存のデータセット別設定を固定しました。"
            "複数の世代点を順次検定した探索的解析で、多重比較補正は行っていません。",
            BODY_SMALL,
        ),
    ]
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    configure_matplotlib()
    results, raw = load()
    make_figure(results, raw)
    build_pdf(results)
    print(FIGURE)
    print(OUTPUT)


if __name__ == "__main__":
    main()
