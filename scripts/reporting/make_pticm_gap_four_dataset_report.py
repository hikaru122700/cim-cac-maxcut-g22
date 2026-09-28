"""Create four-dataset PT-ICM BKS-gap plots and a Japanese PDF guide."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    / "pticm_gap_until_nonsig_G22_G55_G70_K2000_nt16"
)
FIGURE = RESULT_DIR / "pticm_bks_gap_vs_sweeps_ja.png"
OUTPUT = ROOT / "output" / "pdf" / "pticm_bks_gap_four_datasets_ja.pdf"

NAVY = "#17324d"
COLD = "#687780"
WARM = "#d24a43"
GRID = "#d7e0e6"


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
    for ax, dataset in zip(axes.flat, ("G22", "G55", "G70", "K2000")):
        data = results["datasets"][dataset]
        points = data["points"]
        x = np.array([point["sweeps"] for point in points], dtype=float)
        for condition, label, color, marker in (
            ("cold", "random開始", COLD, "o"),
            ("warm", "CIM開始", WARM, "s"),
        ):
            means = []
            ci = []
            for point in points:
                key = f"{dataset}_{point['sweeps']}_{condition}"
                gaps = data["bks"] - np.asarray(raw[key], dtype=float)
                means.append(gaps.mean())
                ci.append(bootstrap_ci(gaps, seed=point["sweeps"] + (0 if condition == "cold" else 99)))
            means = np.asarray(means)
            ci = np.asarray(ci)
            yerr = np.vstack([means - ci[:, 0], ci[:, 1] - means])
            ax.errorbar(
                x, means, yerr=yerr, color=color, marker=marker, ms=6,
                lw=2, capsize=3, label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("PT-ICM sweeps数（対数目盛）", fontsize=20, labelpad=8)
        ax.set_ylabel("平均BKS gap（小さいほど良い）", fontsize=20, labelpad=8)
        ax.set_title(f"{dataset}  BKS={data['bks']}", fontsize=22, pad=12)
        ax.grid(True, which="both", color=GRID, ls=":")
        ax.set_xticks(x, [str(int(v)) for v in x])
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.legend(loc="upper right", fontsize=15)
    fig.suptitle(
        "PT-ICM：random開始とCIM開始のBKS gap",
        fontsize=28, fontweight="bold", color=NAVY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE, bbox_inches="tight")
    plt.close(fig)


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d8dee4"))
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(REPORT_MID)
    canvas.drawString(15 * mm, 6.5 * mm, "PT-ICM BKS gap対sweeps比較")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, str(doc.page))
    canvas.restoreState()


def rows(results: dict, dataset: str) -> list[list[str]]:
    output = []
    for point in results["datasets"][dataset]["points"]:
        output.append([
            str(point["sweeps"]),
            f"{point['cold']['gap_mean']:.1f}",
            f"{point['warm']['gap_mean']:.1f}",
            f"{point['paired']['delta_mean']:+.1f}",
            f"{point['paired']['wilcoxon_p']:.3g}",
            "あり" if point["paired"]["wilcoxon_p"] < 0.05 else "なし",
            f"{point['cold']['total_time']:.1f}",
            f"{point['warm']['total_time']:.1f}",
        ])
    return output


def stop_text(results: dict, dataset: str) -> str:
    data = results["datasets"][dataset]
    if data.get("stop_reached"):
        return f"有意差がなくなった最初の測定点は{data['stop_sweeps']} sweepsです。"
    cap = data.get("practical_cap_sweeps", data["points"][-1]["sweeps"])
    return f"{cap} sweepsまで有意差が残りました。差の消失点は未到達です。"


def build_pdf(results: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=landscape(A4),
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm, bottomMargin=15 * mm,
        title="PT-ICM BKS gap対sweeps比較",
    )
    story = [
        p("PT-ICM：sweeps数を増やすとCIM初期解の差はいつ消えるか", H1),
        Spacer(1, 1 * mm),
        chart(FIGURE, width_mm=218),
        Spacer(1, 2 * mm),
        callout("灰色がrandom開始、赤がCIM開始です。縦軸は下にあるほどBKSに近く、良い結果です。"),
        PageBreak(),
        p("グラフの読み方と停止点", H1),
        Spacer(1, 3 * mm),
        three_boxes([
            ("縦軸", "16試行の平均BKS gapです。下にある線ほど良く、0に近いほど既知最良解に近いことを示します。", PALE_GREEN),
            ("横軸", "PT-ICMを動かしたsweeps数です。幅広い範囲を見せるため対数目盛にしています。", PALE_BLUE),
            ("有意・差なし", "同じseedの16組を比較し、p<0.05を「有意」、p>=0.05を「差なし」と表示します。", PALE_GRAY),
        ]),
        Spacer(1, 6 * mm),
        data_table(
            ["問題", "最終測定", "random平均gap", "CIM平均gap", "p", "判定"],
            [
                [
                    dataset,
                    f"{results['datasets'][dataset]['points'][-1]['sweeps']} sweeps",
                    f"{results['datasets'][dataset]['points'][-1]['cold']['gap_mean']:.1f}",
                    f"{results['datasets'][dataset]['points'][-1]['warm']['gap_mean']:.1f}",
                    f"{results['datasets'][dataset]['points'][-1]['paired']['wilcoxon_p']:.3g}",
                    "差なし" if results["datasets"][dataset].get("stop_reached") else "有意差が残る",
                ]
                for dataset in ("G22", "G55", "G70", "K2000")
            ],
            [32, 48, 52, 48, 38, 48],
        ),
        Spacer(1, 6 * mm),
        callout("G22・G55・G70は1800 sweepsで差が不明確になりました。K2000は1800 sweepsでも差が残りました。"),
        Spacer(1, 4 * mm),
        p("各sweeps点は独立した16試行です。線は同じ1回の実行が時間とともに動いた軌跡ではありません。", BODY_SMALL),
        PageBreak(),
    ]

    for left, right in (("G22", "G55"), ("G70", "K2000")):
        story += [
            p(f"数値表：{left}・{right}", H1),
            Spacer(1, 3 * mm),
            Table(
                [[
                    data_table(
                        ["sweeps", "random gap", "CIM gap", "改善", "p", "有意差", "random秒", "CIM総秒"],
                        rows(results, left), [13, 18, 17, 14, 15, 15, 16, 16],
                    ),
                    data_table(
                        ["sweeps", "random gap", "CIM gap", "改善", "p", "有意差", "random秒", "CIM総秒"],
                        rows(results, right), [13, 18, 17, 14, 15, 15, 16, 16],
                    ),
                ]],
                colWidths=[132 * mm, 132 * mm],
                style=TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1 * mm),
                ]),
            ),
            Spacer(1, 6 * mm),
            three_boxes([
                (
                    left,
                    stop_text(results, left),
                    PALE_GREEN,
                ),
                (
                    right,
                    stop_text(results, right),
                    PALE_BLUE,
                ),
                (
                    "時間の意味",
                    "random秒はPT-ICMだけ、CIM総秒はCIM解の生成からPT-ICM終了までの16試行バッチ時間です。",
                    PALE_GRAY,
                ),
            ]),
            Spacer(1, 4 * mm),
            callout("各sweeps点は独立した実験です。線は同じ1回の実行が動いた軌跡ではなく、計算量ごとの平均結果を結ぶ補助線です。"),
            PageBreak(),
        ]

    story += [
        p("停止規則と解釈上の注意", H1),
        Spacer(1, 3 * mm),
        p("候補sweepsを10、30、80、200、600、1800の順に試し、"
          "対応あり両側Wilcoxon検定でp>=0.05となった最初の点を停止点としました。"
          "G22・G55・G70は1800で停止条件へ到達しました。K2000は1800でも有意差が残り、"
          "次の5000 sweepsは16試行の予測総計算時間が1時間を超えるため実用上の上限として打ち切りました。"
          "「差なし」は同じ性能を証明するものではなく、16試行では差を明確に検出できなかったという意味です。", BODY),
        Spacer(1, 4 * mm),
        p("CIMは600 rounds、warm側はCIM ladder初期化です。PT-ICMの温度数・温度範囲・交換間隔などは"
          "データセットごとの既存設定を使用し、今回の結果を見て追加調整はしていません。", BODY_SMALL),
        Spacer(1, 8 * mm),
        three_boxes([
            (
                "G22・G55・G70",
                "1800 sweepsでは両条件の平均gapが近づき、16試行では初期解による差を明確に検出できなくなりました。",
                PALE_GREEN,
            ),
            (
                "K2000",
                "1800 sweepsでも全16組でCIM開始が勝ちました。密な重み付き問題では初期解の影響が長く残っています。",
                PALE_BLUE,
            ),
            (
                "実用上の意味",
                "短時間では4問題すべてでCIM開始が有利です。長く回すなら、問題によってCIM生成費用の価値が変わります。",
                PALE_GRAY,
            ),
        ]),
        Spacer(1, 6 * mm),
        callout("結論：CIM初期解の効果はG22・G55・G70では1800 sweepsで不明確になりますが、K2000では1800 sweepsでも明確に残ります。"),
        Spacer(1, 4 * mm),
        p("統計上の注意：複数のsweeps点を順次検定した探索的解析で、多重比較補正は行っていません。"
          "確証実験では停止点を事前固定し、新しいseedで再検証する必要があります。", BODY_SMALL),
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
