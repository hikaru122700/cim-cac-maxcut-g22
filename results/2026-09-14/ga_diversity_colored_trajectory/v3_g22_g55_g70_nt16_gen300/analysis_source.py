"""GAの累積最大カット値を、同世代の集団多様性で着色して可視化する。

保存済みのG22・G55・G70、全ランダム／全CIM、各16 run × 300世代を使用する。
新しいGA実験は行わず、各点を「1 run・1世代」として描く。
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


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KIND = "ga_diversity_colored_trajectory"
SOURCE_DIR = (
    ROOT
    / "results/2026-09-14/ga_diversity_trajectory/"
    "v1_G22_G55_G70_nt16_gen300_measured"
)
GRAPHS = ("G22", "G55", "G70")
CONDITIONS = ("random", "cim")
CONDITION_LABELS = {"random": "全ランダム初期集団", "cim": "全CIM初期集団"}


def next_version(kind_root: Path) -> int:
    """過去結果を上書きしない次の版番号を返す。"""
    max_v = 0
    if kind_root.exists():
        for path in kind_root.iterdir():
            if path.is_dir() and path.name.startswith("v"):
                head = path.name.split("_", 1)[0]
                if head[1:].isdigit():
                    max_v = max(max_v, int(head[1:]))
    return max_v + 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_data() -> tuple[dict, dict[tuple[str, str], dict[str, np.ndarray]]]:
    """世代0から300までの軌跡と設定を読み込む。"""
    meta = json.loads((SOURCE_DIR / "metadata.json").read_text(encoding="utf-8"))
    datasets: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for graph in GRAPHS:
        for condition in CONDITIONS:
            path = SOURCE_DIR / f"{graph}_{condition}_trajectory.npz"
            with np.load(path) as data:
                keep = data["generation"] >= 0
                datasets[(graph, condition)] = {
                    "generation": data["generation"][keep].astype(int),
                    "diversity": data["metrics"][keep, :, 0].astype(float),
                    "best_ever": data["best_ever"][keep, :].astype(float),
                    "seeds": data["seeds"].astype(int),
                }
    return meta, datasets


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Yu Gothic",
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )


def make_figure(meta: dict, datasets: dict, out: Path) -> None:
    """3問題×2初期集団の色付き軌跡を作る。"""
    setup_style()
    fig, axes = plt.subplots(3, 2, figsize=(15.4, 9.2), sharex=True)
    last_scatter = None

    for row, graph in enumerate(GRAPHS):
        combined = np.concatenate(
            [datasets[(graph, condition)]["best_ever"].ravel() for condition in CONDITIONS]
        )
        ymin = float(combined.min())
        ymax = max(float(combined.max()), float(meta["datasets"][graph]["bks"]))
        padding = max(2.0, 0.045 * (ymax - ymin))

        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            item = datasets[(graph, condition)]
            generation = item["generation"]
            best = item["best_ever"]
            diversity = item["diversity"]

            # 各runを薄線でつなぎ、各run・各世代の点を同時点のDで着色する。
            for run in range(best.shape[1]):
                ax.plot(generation, best[:, run], color="#B8B8B8", lw=0.45, alpha=0.45, zorder=1)
                last_scatter = ax.scatter(
                    generation,
                    best[:, run],
                    c=diversity[:, run],
                    cmap="viridis",
                    vmin=0.0,
                    vmax=0.5,
                    s=6,
                    linewidths=0,
                    alpha=0.68,
                    rasterized=True,
                    zorder=2,
                )

            ax.plot(
                generation,
                best.mean(axis=1),
                color="black",
                lw=1.8,
                label="16 run平均",
                zorder=3,
            )
            bks = int(meta["datasets"][graph]["bks"])
            ax.axhline(bks, color="#C83E3E", lw=1.2, ls="--", label=f"BKS {bks}", zorder=0)
            ax.set_ylim(ymin - padding, ymax + padding)
            ax.set_xlim(0, 300)
            ax.set_title(f"{graph}：{CONDITION_LABELS[condition]}", loc="left", fontweight="bold")
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(alpha=0.18, linewidth=0.65)
            if col == 0:
                ax.set_ylabel("累積最大カット値")
            if row == len(GRAPHS) - 1:
                ax.set_xlabel("世代数")
            if row == 0:
                ax.legend(frameon=False, loc="lower right", fontsize=8)

    cbar_ax = fig.add_axes([0.91, 0.16, 0.014, 0.70])
    cbar = fig.colorbar(last_scatter, cax=cbar_ax)
    cbar.set_label("集団多様性スコア $D$")
    cbar.ax.tick_params(direction="in")
    fig.suptitle(
        "GAの累積最大カット値と集団多様性の同時推移",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "各色点＝1 run・1世代（各条件16 run）。黒線＝16 run平均。色が黄緑ほど多様性が高く、紫ほど低い。",
        ha="center",
        fontsize=10.5,
        color="#263D50",
    )
    fig.subplots_adjust(left=0.07, right=0.885, bottom=0.075, top=0.93, hspace=0.28, wspace=0.16)
    fig.savefig(out / "trajectory.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "trajectory.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_summary(meta: dict, datasets: dict) -> list[dict]:
    """PDF表に使う開始・終了時の群平均を集計する。"""
    rows: list[dict] = []
    for graph in GRAPHS:
        bks = int(meta["datasets"][graph]["bks"])
        for condition in CONDITIONS:
            item = datasets[(graph, condition)]
            best = item["best_ever"]
            diversity = item["diversity"]
            rows.append(
                {
                    "graph": graph,
                    "condition": condition,
                    "runs": best.shape[1],
                    "generation0_best_mean": float(best[0].mean()),
                    "generation300_best_mean": float(best[-1].mean()),
                    "cut_gain_mean": float((best[-1] - best[0]).mean()),
                    "generation0_diversity_mean": float(diversity[0].mean()),
                    "generation300_diversity_mean": float(diversity[-1].mean()),
                    "bks": bks,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_results_md(out: Path) -> None:
    text = """# GAの累積最大カット値と多様性の同時推移

実験日: 2026-09-14 / G22・G55・G70 / 各条件16 run・300世代  
新規GA実行: **なし（保存済み軌跡の再描画）**

> 各runの累積最大カット値を縦軸、世代を横軸、同じrun・世代の集団多様性を点の色で示す。

## 図

![色付きGA軌跡](./trajectory.png)

- 各色点は1 run・1世代。
- 黒線は16 runの平均累積最大カット値。
- 多様性 $D$ は、全反転を同一分割とした集団内平均ペアHamming距離。
- 黄緑ほど高多様性、紫ほど低多様性。
- 世代0は初期集団の全個体にTabu Searchを適用した後。

## 注意

全ランダム初期集団と全CIM初期集団では世代0の品質が一致していない。この図は品質と多様性の同時推移を示すが、多様性が性能差を生んだという因果検証ではない。

## 再現

解析コード: `modules/2026-09-14_GA_DIVERSITY_COLORED_TRAJECTORY.py`  
元データ: `results/2026-09-14/ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_measured`  
集計値: `summary.csv` / 入力ハッシュ: `provenance.json`  
TeX: `report.tex` / PDF: `report.pdf`
"""
    (out / "RESULTS.md").write_text(text, encoding="utf-8")


def build_tex(rows: list[dict], out: Path) -> None:
    table_rows = []
    for row in rows:
        table_rows.append(
            f"{row['graph']} & {CONDITION_LABELS[row['condition']]} & "
            f"{row['generation0_best_mean']:.2f} & {row['generation300_best_mean']:.2f} & "
            f"{row['cut_gain_mean']:.2f} & {row['generation0_diversity_mean']:.4f} & "
            f"{row['generation300_diversity_mean']:.4f} \\\\"
        )
    tex = r"""\documentclass[11pt,a4paper,landscape]{article}
\usepackage[margin=12mm]{geometry}
\usepackage{fontspec,xeCJK}
\setmainfont{Arial}
\setCJKmainfont{Yu Gothic}
\usepackage{graphicx,booktabs,xcolor,amsmath,xurl}
\urlstyle{tt}
\definecolor{ink}{HTML}{263D50}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\newcommand{\heading}[2]{{\LARGE\bfseries\color{ink}#1}\par{\small #2}\par\vspace{1mm}}
\begin{document}
\heading{GAの最大カット値を、多様性の色で追跡する}
{G22・G55・G70、全ランダム／全CIM初期集団、各16 run・300世代。保存済み軌跡の再描画}

\begin{center}
\includegraphics[width=0.97\linewidth,height=159mm,keepaspectratio]{trajectory.png}
\end{center}

\clearpage
\heading{開始時と300世代後の数値}{世代0は初期集団の全個体にTabu Searchを適用した後}

{\small \textbf{点の定義：}各点は「1 run・1世代」。横軸は世代数、縦軸はそのrunがその世代までに見つけた累積最大カット値、色は同じrun・同じ世代の集団多様性$D$。黒線は16 run平均、赤破線はBKS。$D=\operatorname{mean}_{i<j}\min(H_{ij},N-H_{ij})/N$で、全反転は同じ分割として扱う。}

\begin{center}\small
\begin{tabular}{llrrrrr}\toprule
問題 & 初期集団 & 世代0最良値 & 世代300最良値 & 改善量 & 世代0の$D$ & 世代300の$D$ \\\midrule
@@ROWS@@
\bottomrule\end{tabular}
\end{center}

\textbf{図から読み取れる例}

\begin{itemize}
\item G22では両初期集団とも世代とともに最大カット値が上がり、多様性は低下する。
\item G55の全CIMは比較的高い多様性を保つ点が多いが、最大カット値の改善は小さい。
\item G70の全CIMは多様性がほぼ一定のまま、16 runすべてで最大カット値が改善しない。
\end{itemize}

\colorbox{yellow!18}{\parbox{0.96\linewidth}{\textbf{解釈上の注意：}全ランダムと全CIMでは世代0の品質をそろえていない。この図は最大カット値と多様性の時間的な同時推移を示すが、多様性が改善を引き起こしたという因果関係は示さない。}}

\textbf{再現性}

元データ：\path{results/2026-09-14/ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_measured}。解析コード、群集計、入力SHA-256を本PDFと同じフォルダに保存した。元実験では864集団の距離と8352個体のカット値が独立照合済み。
\end{document}
"""
    (out / "report.tex").write_text(tex.replace("@@ROWS@@", "\n".join(table_rows)), encoding="utf-8")


def main(tag: str) -> Path:
    meta, datasets = load_data()
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    version = next_version(kind_root)
    description = "g22_g55_g70_nt16_gen300"
    if tag:
        description += "_" + tag.lower().replace(" ", "_")
    out = kind_root / f"v{version}_{description}"
    out.mkdir(parents=True, exist_ok=False)

    make_figure(meta, datasets, out)
    rows = build_summary(meta, datasets)
    write_csv(out / "summary.csv", rows)
    build_results_md(out)
    build_tex(rows, out)
    shutil.copy2(__file__, out / "analysis_source.py")

    inputs = {"metadata.json": sha256(SOURCE_DIR / "metadata.json")}
    for graph in GRAPHS:
        for condition in CONDITIONS:
            name = f"{graph}_{condition}_trajectory.npz"
            inputs[name] = sha256(SOURCE_DIR / name)
    provenance = {
        "created": date.today().isoformat(),
        "new_ga_runs": 0,
        "point_definition": "one run and one generation; y=best_ever; color=mean_pairwise diversity",
        "inputs_sha256": inputs,
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
