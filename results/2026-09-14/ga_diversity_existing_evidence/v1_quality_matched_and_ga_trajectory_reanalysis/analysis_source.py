"""既存結果だけを用いて、GAにおける多様性の研究動機と因果限界を可視化する。

入力:
  - 同一カット値分布に抽出したCIM解集合の多様性
  - 全ランダム／全CIM初期集団から始めたGA 16 run × 300世代の軌跡

新しいGA実験は行わない。図、集計CSV/JSON、TeX原稿を版付き結果フォルダへ保存する。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KIND = "ga_diversity_existing_evidence"
MATCHED_DIR = (
    ROOT
    / "results/2026-09-14/cim_diversity_evidence/"
    "v2_G22_G55_G70_matched_r1000_discussion"
)
TRAJECTORY_DIR = (
    ROOT
    / "results/2026-09-14/ga_diversity_trajectory/"
    "v1_G22_G55_G70_nt16_gen300_measured"
)
GRAPHS = ("G22", "G55", "G70")
CONDITIONS = ("random", "cim")
GRAPH_COLORS = {"G22": "#2D6A9F", "G55": "#D17A22", "G70": "#4D8B57"}
COND_MARKERS = {"random": "o", "cim": "s"}
COND_LABELS = {"random": "全ランダム", "cim": "全CIM"}


def next_version(kind_root: Path) -> int:
    """既存結果を上書きしない次の版番号を返す。"""
    max_v = 0
    if kind_root.exists():
        for path in kind_root.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                head = path.name.split("_", 1)[0]
                if head[1:].isdigit():
                    max_v = max(max_v, int(head[1:]))
    return max_v + 1


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """runを独立単位とした平均と95% t区間の半幅。"""
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2 or np.all(values == values[0]):
        return mean, 0.0
    half = float(t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, half


def load_records() -> tuple[dict, list[dict], list[dict]]:
    """同品質データとGA軌跡を読み、群集計・run集計を返す。"""
    matched = json.loads((MATCHED_DIR / "metrics.json").read_text(encoding="utf-8"))
    meta = json.loads((TRAJECTORY_DIR / "metadata.json").read_text(encoding="utf-8"))
    grouped: list[dict] = []
    runs: list[dict] = []
    for graph in GRAPHS:
        bks = int(meta["datasets"][graph]["bks"])
        for condition in CONDITIONS:
            path = TRAJECTORY_DIR / f"{graph}_{condition}_trajectory.npz"
            with np.load(path) as data:
                generations = data["generation"]
                i0 = int(np.flatnonzero(generations == 0)[0])
                i_final = int(np.flatnonzero(generations == 300)[0])
                diversity = data["metrics"][i0, :, 0].astype(float)
                initial_best = data["best_ever"][i0].astype(float)
                final_best = data["best_ever"][i_final].astype(float)
                gain = final_best - initial_best
                initial_gap = bks - initial_best
                final_gap = bks - final_best
                seeds = data["seeds"].astype(int)
            d_mean, d_ci = mean_ci95(diversity)
            gain_mean, gain_ci = mean_ci95(gain)
            gap_mean, gap_ci = mean_ci95(initial_gap)
            grouped.append(
                {
                    "graph": graph,
                    "condition": condition,
                    "n_runs": len(diversity),
                    "generation0_diversity_mean": d_mean,
                    "generation0_diversity_ci95_halfwidth": d_ci,
                    "initial_gap_mean": gap_mean,
                    "initial_gap_ci95_halfwidth": gap_ci,
                    "gain_mean": gain_mean,
                    "gain_ci95_halfwidth": gain_ci,
                    "final_gap_mean": float(final_gap.mean()),
                }
            )
            for seed, d_value, gap_value, gain_value, final_gap_value in zip(
                seeds, diversity, initial_gap, gain, final_gap
            ):
                runs.append(
                    {
                        "graph": graph,
                        "condition": condition,
                        "seed": int(seed),
                        "generation0_diversity": float(d_value),
                        "initial_gap": float(gap_value),
                        "gain": float(gain_value),
                        "final_gap": float(final_gap_value),
                    }
                )
    return matched, grouped, runs


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Yu Gothic",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )


def inward_ticks(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.2, linewidth=0.7)


def make_main_figure(matched: dict, grouped: list[dict], runs: list[dict], out: Path) -> None:
    """研究動機、観測された関係、交絡を1枚にまとめる。"""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.25))

    # A: カット値の多重集合を完全一致させた保存済みCIM解の距離。
    ax = axes[0]
    x = np.arange(len(GRAPHS))
    width = 0.34
    baseline = [matched[g]["conditions"]["baseline"]["matched_mean_distance"] for g in GRAPHS]
    noise = [matched[g]["conditions"]["noise1000x"]["matched_mean_distance"] for g in GRAPHS]
    ax.bar(x - width / 2, baseline, width, label="基準CIM", color="#708090")
    ax.bar(x + width / 2, noise, width, label="ノイズ1000倍", color="#5A9BD4")
    for i, graph in enumerate(GRAPHS):
        delta = noise[i] - baseline[i]
        ax.text(i, max(baseline[i], noise[i]) + 0.004, f"差 {delta:+.3f}", ha="center", fontsize=9)
    ax.set_xticks(x, GRAPHS)
    ax.set_ylim(0.30, 0.46)
    ax.set_ylabel("平均反転同一視Hamming距離 $D$")
    ax.set_title("A  同じ品質でも、解の違いは残る", loc="left", fontweight="bold")
    ax.text(
        0.02,
        0.03,
        "各グラフでカット値の多重集合を完全一致\n（各条件 7 / 11 / 11 解）",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    ax.legend(frameon=False, loc="upper left")
    inward_ticks(ax)

    # B: GA群平均。見かけの関連だが初期品質は未調整。
    ax = axes[1]
    for row in grouped:
        graph = row["graph"]
        condition = row["condition"]
        ax.errorbar(
            row["generation0_diversity_mean"],
            row["gain_mean"],
            xerr=row["generation0_diversity_ci95_halfwidth"],
            yerr=row["gain_ci95_halfwidth"],
            fmt=COND_MARKERS[condition],
            markersize=8,
            color=GRAPH_COLORS[graph],
            markerfacecolor=(GRAPH_COLORS[graph] if condition == "random" else "white"),
            markeredgewidth=1.8,
            capsize=3,
            alpha=0.95,
        )
        ax.annotate(
            f"{graph} {COND_LABELS[condition]}",
            (row["generation0_diversity_mean"], row["gain_mean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("世代0の集団多様性 $D$")
    ax.set_ylabel("300世代での最良カット改善量")
    ax.set_title("B  高多様性群ほど改善が大きく見える", loc="left", fontweight="bold")
    ax.text(
        0.04,
        0.96,
        "注意：初期品質をそろえていない\n（各点 16 run の平均 ± 95% t区間）",
        transform=ax.transAxes,
        va="top",
        color="#9B2C2C",
        fontsize=9,
        bbox={"facecolor": "#FFF5F5", "edgecolor": "#D9A0A0", "alpha": 0.95},
    )
    inward_ticks(ax)

    # C: 改善量は初期の改善余地に支配されることをrun単位で示す。
    ax = axes[2]
    max_gap = max(row["initial_gap"] for row in runs)
    ax.plot([0, max_gap * 1.03], [0, max_gap * 1.03], "--", color="#777777", lw=1, label="改善余地の上限")
    for graph in GRAPHS:
        for condition in CONDITIONS:
            subset = [r for r in runs if r["graph"] == graph and r["condition"] == condition]
            ax.scatter(
                [r["initial_gap"] for r in subset],
                [r["gain"] for r in subset],
                s=34,
                marker=COND_MARKERS[condition],
                facecolors=(GRAPH_COLORS[graph] if condition == "random" else "none"),
                edgecolors=GRAPH_COLORS[graph],
                linewidths=1.3,
                alpha=0.82,
                label=f"{graph} {COND_LABELS[condition]}",
            )
    ax.set_xlabel("世代0のBKSとの差（改善余地）")
    ax.set_ylabel("300世代での最良カット改善量")
    ax.set_title("C  改善量は開始位置にも左右される", loc="left", fontweight="bold")
    ax.text(
        0.04,
        0.96,
        "右上ほど、もともと改善余地が大きい\n→ Bだけでは多様性の因果効果を分離できない",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.94},
    )
    inward_ticks(ax)

    fig.suptitle(
        "多様性はGAの『材料』になり得るが、性能への因果効果はまだ未検証",
        fontsize=17,
        fontweight="bold",
        x=0.5,
        y=1.01,
    )
    fig.text(
        0.5,
        -0.01,
        "既存結果から言えること：品質と多様性は別の評価軸。次に必要：同じ品質・同じ集団サイズで多様性だけを変えたGA比較。",
        ha="center",
        fontsize=11,
        fontweight="bold",
        color="#263D50",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96), w_pad=2.3)
    fig.savefig(out / "evidence_map.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "evidence_map.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def build_tex(matched: dict, grouped: list[dict], out: Path) -> None:
    matched_rows = []
    for graph in GRAPHS:
        base = matched[graph]["conditions"]["baseline"]
        noise = matched[graph]["conditions"]["noise1000x"]
        matched_rows.append(
            f"{graph} & {matched[graph]['match_count']} & {base['matched_cut_mean']:.3f} & "
            f"{base['matched_mean_distance']:.4f} & {noise['matched_mean_distance']:.4f} & "
            f"{noise['matched_mean_distance'] - base['matched_mean_distance']:+.4f} \\\\"
        )
    ga_rows = []
    for row in grouped:
        ga_rows.append(
            f"{row['graph']} & {COND_LABELS[row['condition']]} & "
            f"{row['generation0_diversity_mean']:.4f} & {row['initial_gap_mean']:.2f} & "
            f"{row['gain_mean']:.2f} & {row['final_gap_mean']:.2f} \\\\"
        )
    tex = r"""\documentclass[11pt,a4paper,landscape]{article}
\usepackage[margin=13mm]{geometry}
\usepackage{fontspec}
\usepackage{xeCJK}
\setmainfont{Arial}
\setCJKmainfont{Yu Gothic}
\usepackage{graphicx,booktabs,xcolor,amsmath,array}
\definecolor{ink}{HTML}{263D50}
\definecolor{warn}{HTML}{9B2C2C}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\newcommand{\heading}[2]{{\LARGE\bfseries\color{ink}#1}\par{\small #2}\par\vspace{2mm}}
\begin{document}
\heading{多様性はGAの「材料」になり得るが、効果はまだ未検証}
{保存済みデータだけを再解析：同品質CIM解集合と、16 run $\times$ 300世代のGA軌跡}

\begin{center}
\includegraphics[width=0.985\linewidth,height=118mm,keepaspectratio]{evidence_map.png}
\end{center}

\textbf{読み方。} Aは「品質が同じでも異なる分割を持てる」ことを直接示す。Bでは高多様性群ほどGA改善が大きく見えるが、Cが示すように全ランダム群は開始時のカット値が低く、改善余地も大きい。したがって、Bの差を多様性の効果とは解釈できない。

\colorbox{yellow!18}{\parbox{0.97\linewidth}{\textbf{結論：}既存結果は、品質だけでなく多様性を測る研究動機を支持する。一方、「同じ品質なら多様な集団ほどGAの最終性能が高い」という因果命題は、まだ検証していない。}}

{\footnotesize 距離 $D$ は全反転を同一分割とする平均ペアHamming距離 $\min(H,N-H)/N$。誤差棒はrunを単位とする平均の95\% $t$区間。BKSは保存済み既知最良値。}

\clearpage
\heading{数値と研究上の位置づけ}{図に使った値。新しいGA実行や結果の補間は行っていない}

\textbf{1. カット値分布を完全一致させたCIM解集合}
\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
問題 & 各条件の解数 & 同じ平均カット値 & 基準CIMの$D$ & ノイズ1000倍の$D$ & 距離差 \\\midrule
@@MATCHED_ROWS@@
\bottomrule\end{tabular}
\end{center}

カット値ごとに両条件の少ない方の個数だけ保存順で抽出したため、平均だけでなくカット値の多重集合が一致する。距離による選別はしていない。ただし各条件7--11解の事後抽出であり、独立したGA反復ではない。

\textbf{2. GA軌跡の群平均（初期品質は未調整）}
\begin{center}\small
\begin{tabular}{llrrrr}\toprule
問題 & 初期集団 & 世代0の$D$ & 世代0のBKS差 & 300世代の改善量 & 最終BKS差 \\\midrule
@@GA_ROWS@@
\bottomrule\end{tabular}
\end{center}

全ランダムと全CIMは同一GA設定、各16 run、300世代だが、世代0の品質をそろえていない。特にG55・G70では全ランダム群の改善余地が大きい。G70の全CIM群は16 runすべてで改善量0だったが、これは「多様性があれば改善する」とも「多様性が無意味」とも単独では言えない。

\textbf{3. 次に必要な検証}

同じカット値の多重集合、同じ集団サイズ、同じGA設定・計算予算を固定し、平均距離だけが低い／高い初期集団を複数構成する。対応する乱数で、(i) 最終カット、(ii) 最良値への到達世代、(iii) 交叉直後の新規性、(iv) 局所探索後の改善量を比較する。この比較で初めて、「改善につながる違い」を多様性指標が捉えるかを検証できる。

\textbf{出典と再現性}

同品質データ：\texttt{results/2026-09-14/cim\_diversity\_evidence/v2\_G22\_G55\_G70\_matched\_r1000\_discussion}。GA軌跡：\texttt{results/2026-09-14/ga\_diversity\_trajectory/v1\_G22\_G55\_G70\_nt16\_gen300\_measured}。元実験では864集団の距離と8352個体のカット値を独立照合済み。再解析コードと入力SHA-256は本PDFと同じフォルダに保存した。
\end{document}
"""
    tex = tex.replace("@@MATCHED_ROWS@@", "\n".join(matched_rows))
    tex = tex.replace("@@GA_ROWS@@", "\n".join(ga_rows))
    (out / "report.tex").write_text(tex, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(tag: str) -> Path:
    matched, grouped, runs = load_records()
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    version = next_version(kind_root)
    description = "quality_matched_and_ga_trajectory_reanalysis"
    if tag:
        description += "_" + tag.lower().replace(" ", "_")
    out = kind_root / f"v{version}_{description}"
    out.mkdir(parents=True, exist_ok=False)

    make_main_figure(matched, grouped, runs, out)
    write_csv(out / "group_summary.csv", grouped)
    write_csv(out / "run_data.csv", runs)
    build_tex(matched, grouped, out)
    shutil.copy2(__file__, out / "analysis_source.py")

    provenance = {
        "created": date.today().isoformat(),
        "new_ga_runs": 0,
        "claim": "品質と多様性は別軸だが、多様性のGA性能への因果効果は未検証",
        "inputs": {
            str((MATCHED_DIR / "metrics.json").relative_to(ROOT)): sha256(MATCHED_DIR / "metrics.json"),
            str((TRAJECTORY_DIR / "metadata.json").relative_to(ROOT)): sha256(TRAJECTORY_DIR / "metadata.json"),
            **{
                str((TRAJECTORY_DIR / f"{g}_{c}_trajectory.npz").relative_to(ROOT)): sha256(
                    TRAJECTORY_DIR / f"{g}_{c}_trajectory.npz"
                )
                for g in GRAPHS
                for c in CONDITIONS
            },
        },
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(out)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="出力フォルダへ付ける任意タグ")
    args = parser.parse_args()
    main(args.tag)
