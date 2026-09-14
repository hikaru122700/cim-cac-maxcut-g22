"""Build explanatory Japanese plots and LuaLaTeX source from measured GA traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

LABELS = {"random": "ランダム初期集団", "cim": "全CIM初期集団"}
COLORS = {"random": "#506778", "cim": "#C34436"}


def build(out):
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    validation = json.loads((out / "independent_validation.json").read_text(encoding="utf-8"))
    assert validation["all_checks_passed"]
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False,
                         "font.size": 12, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    datasets = list(meta["datasets"])
    for kind in ["diversity", "quality"]:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.3 * len(datasets), 4.1), squeeze=False)
        for col, ds in enumerate(datasets):
            ax = axes[0, col]
            for condition in LABELS:
                with np.load(out / f"{ds}_{condition}_trajectory.npz") as z:
                    g = z["generation"][1:]
                    values = (z["metrics"][1:, :, 0] if kind == "diversity"
                              else meta["datasets"][ds]["bks"] - z["best_ever"][1:])
                    mean = values.mean(axis=1)
                    ci = student_t.ppf(.975, values.shape[1]-1) * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])
                    ax.plot(g, mean, color=COLORS[condition], lw=2.1, label=LABELS[condition])
                    ax.fill_between(g, mean-ci, mean+ci, color=COLORS[condition], alpha=.15)
            ax.set(title=ds, xlabel="世代（子を1個生成＝1世代）")
            ax.set_xscale("symlog", linthresh=1)
            ticks = [g for g in [0, 1, 3, 8, 20, 50, 120, 300, 800] if g <= meta["generations"]]
            ax.set_xticks(ticks, labels=[str(g) for g in ticks])
            if kind == "diversity":
                ax.set_ylabel("多様性スコア D（大きいほど多様）")
                ax.set_ylim(0, .5)
            else:
                ax.set_ylabel("最良カット値とBKSの差（小さいほど良い）")
            ax.grid(alpha=.2)
            ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"{kind}_only.png", dpi=180)
        fig.savefig(out / f"{kind}_only.svg")
        plt.close(fig)

    table_rows = []
    quality_rows = []
    param_rows = []
    md = ["# GA世代ごとの多様性の実測", "",
          f"各条件{meta['num_trials']} run、{meta['generations']}世代。各runの集団内で測定した。", "",
          "|問題|初期集団|初期TS前のD|世代0のD|最終世代のD|最終の異なる分割数|",
          "|---|---|---:|---:|---:|---:|"]
    for ds in datasets:
        for condition in LABELS:
            s = summary[ds][condition]
            label = "ランダム" if condition == "random" else "全CIM"
            table_rows.append(f"{ds} & {label} & {s['raw_D']:.4f} & {s['gen0_D']:.4f} & {s['final_D']:.4f} & {s['final_unique']:.2f}" + r"\\")
            quality_rows.append(f"{ds} & {label} & {s['gen0_gap']:.2f} & {s['final_gap']:.2f} & {s['gen0_gap']-s['final_gap']:.2f}" + r"\\")
            md.append(f"|{ds}|{label}|{s['raw_D']:.4f}|{s['gen0_D']:.4f}|{s['final_D']:.4f}|{s['final_unique']:.2f}|")
        p = meta["datasets"][ds]["ga_params"]
        param_rows.append(f"{ds} & {p['pop_size']} & {p['ts_iters']:,} & {p['cr']} & {p['alpha_tenure']} & {p['beta_quality']:.4f} & {meta['datasets'][ds]['cim_rounds']:,}" + r"\\")
    md += ["", "## 読み方", "",
           "Dは集団内の全ペアについて min(H,N-H)/N を平均した値。全反転は同一の分割。",
           "世代0は初期集団にTabu Searchをかけた後。初期TS前はgeneration=-1。",
           "1世代は子を1個生成して交叉、TS、DisQual更新を行う単位。",
           "最終的にDが低くても、全部同じ解とは限らない。異なる分割数も併記した。", "",
           "## 比較条件", "",
           "既存の調整済みGA/CIMパラメータを固定し、全ランダムと全CIM初期集団を比較。",
           "CIMは各集団の全個体を新規生成し、選別や別run間での流用はしていない。",
           "GAの乱数seedは条件間で対応させたが、交叉内部の乱数消費数は集団の状態で異なる。",
           "初期品質はそろえていない。多様性の違いが性能差の原因だという検証ではない。",
           "元スクリーンショットの実行条件は未確認であり、その厳密な再現実験ではない。",
           "世代数を合わせた比較であり、CIM生成時間を含む同一実時間の性能比較ではない。", "",
           "## 検証と再現", "",
           f"独立再計算で{validation['checked_populations']}集団の距離と{validation['checked_scores']}個体のカット値を照合。",
           "7件のGAテストが成功。観測の有無で結果不変、全初期集団の受け渡し、全反転同一視を検証。",
           "全世代のスコアは *_trajectory.npz、チェックポイントの集団は *_snapshots.npz に保存。",
           "metadata.jsonに全パラメータ、seed、入力・コードのハッシュを保存。",
           "実行スクリプト: scripts/benchmarks/ga_diversity_trajectory.py",
           "独立検証: scripts/benchmarks/validate_ga_diversity_trajectory.py",
           "資料生成: scripts/plotting/report_ga_diversity_trajectory.py",
           "LuaLaTeXでreport.texをコンパイルする。WindowsのArialと游ゴシックを使用。"]
    finding = "多様性の推移は問題と初期集団によって異なる。品質の推移と合わせて読む必要がある。"
    if "G55" in summary:
        r, c = summary["G55"]["random"], summary["G55"]["cim"]
        if c["final_D"] > r["final_D"] and c["final_gap"] > r["final_gap"]:
            finding = ("G55では、全CIM集団は多様性を比較的大きく残したが、最終の平均カット値はランダム側が高かった。"
                       "したがって、この距離スコアだけでCIM初期集団の改善が小さい理由を説明することはできない。")
    if "G70" in validation["datasets"]:
        result = validation["datasets"]["G70"]["cim"]
        if (result["runs_with_constant_D_all_generations"] == meta["num_trials"]
                and result["gen0_to_final_cut_gain"]["mean"] == 0):
            finding = (f"G70の全CIM初期集団は、{meta['num_trials']}回すべてで多様性スコアが全世代一定で、最良値も改善しなかった。"
                       "これは「GA中に多様性を失ったから停滞した」という説明と一致しない。"
                       "初期の多様性が十分だったかは、別の問いである。")
    md[4:4] = ["今回の要点：" + finding, ""]
    (out / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tex = r"""\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[11pt,a4paper,landscape]{ltjsarticle}
\usepackage[margin=14mm,footskip=7mm]{geometry}
\usepackage{luatexja-fontspec}
\setmainfont{arial.ttf}[Path=C:/Windows/Fonts/,BoldFont=arialbd.ttf]
\setmainjfont{YuGothM.ttc}[Path=C:/Windows/Fonts/,FontIndex=0,BoldFont=YuGothB.ttc,BoldFeatures={FontIndex=0}]
\setsansjfont{YuGothM.ttc}[Path=C:/Windows/Fonts/,FontIndex=0,BoldFont=YuGothB.ttc,BoldFeatures={FontIndex=0}]
\usepackage{graphicx,booktabs,xcolor,amsmath}
\definecolor{ink}{HTML}{263D50}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\newcommand{\heading}[2]{{\LARGE\bfseries\color{ink}#1}\par{\small #2}\par\vspace{2mm}}
\begin{document}
\heading{GAの世代が進むと、解同士は似てくるのか}{全ランダムと全CIMの初期集団を新たに比較。各条件@@TRIALS@@回、@@GENERATIONS@@世代を追跡}
\includegraphics[width=\linewidth,height=85mm,keepaspectratio]{diversity_only.png}

\textbf{線が下がる＝集団内の解同士が似てくる。}スコア$D$は、2解で分け方が違う頂点の割合を集団内の全ペアで平均した値。全反転は同一の解として扱う。$D=0.1$なら平均で10\%の頂点が異なる。

\begin{center}\small
\begin{tabular}{llrrrr}\toprule
問題 & 初期集団 & 初期TS前の$D$ & 世代0の$D$ & 最終世代の$D$ & 最終の異なる分割数\\\midrule
@@TABLE@@
\bottomrule\end{tabular}
\end{center}
{\footnotesize 線はrun平均、帯はrun単位の平均の95\% $t$区間（各世代の区間であり、曲線全体の同時区間ではない）。世代0は初期TS後。異なる分割数はrun平均。}

\clearpage
\heading{多様性の低下と、解の品質を並べて読む}{前ページと同じ実験・同じ世代。最良値は、各runでそれまでに見つけた最良カット値}
\includegraphics[width=\linewidth,height=85mm,keepaspectratio]{quality_only.png}

\textbf{今回の要点：}@@FINDING@@

\begin{center}\small
\begin{tabular}{llrrr}\toprule
問題 & 初期集団 & 世代0のBKSとの差 & 最終のBKSとの差 & カット値の改善量\\\midrule
@@QUALITY@@
\bottomrule\end{tabular}
\end{center}
{\footnotesize BKSはリポジトリに登録されている既知最良値。表は各runの最良値についての平均。初期品質が異なるため、この比較から多様性が性能差を生んだとは言えない。元のスクリーンショットの実行条件は未確認であり、今回の結果は別の追加実験である。}

\clearpage
\heading{何を測ったか：初期TSと世代更新を区別する}{CIMの生成条件は固定。GAの設定も両初期化条件で同じにした}
\includegraphics[width=\linewidth,height=77mm,keepaspectratio]{initial_refinement.png}

\textbf{処理の順番：}初期集団を作る $\rightarrow$ 全個体をTabu Searchで改善（ここが世代0）$\rightarrow$ 両親を選ぶ $\rightarrow$ 子を1個作る $\rightarrow$ 子をTSで改善 $\rightarrow$ 集団を更新。最後の4処理を1世代と数える。

集団更新は、カット値と最も近い他個体までの距離を併用するDisQual。したがって、距離が毎世代必ず減るという規則ではない。各runの集団内の多様性を記録し、異なるrunの解を混ぜて計算していない。

\begin{center}\small
\begin{tabular}{lrrrrrr}\toprule
問題 & 集団サイズ & TS反復数 & 摂動待ち回数 & tenure係数 & 品質重み & CIM反復数\\\midrule
@@PARAMS@@
\bottomrule\end{tabular}
\end{center}
{\footnotesize 調整済み設定を使用。CIM解は全個体を独立に新規生成し、品質・距離で選別していない。GAのseedは条件間で対応。CIM生成時間を含めた同一実時間の比較ではない。}

{\footnotesize 検証：@@POPS@@集団の距離と@@SCORES@@個体のカット値を別の計算で照合し、一致を確認。計測でGAの結果が変わらないこと等、7件のテストに成功。全世代の数値、集団の保存断面、設定、seed、コードのハッシュは本PDFと同じフォルダに保存。}
\end{document}
"""
    replacements = {"TABLE": "\n".join(table_rows), "QUALITY": "\n".join(quality_rows),
                    "PARAMS": "\n".join(param_rows), "TRIALS": str(meta["num_trials"]),
                    "GENERATIONS": str(meta["generations"]), "POPS": str(validation["checked_populations"]),
                    "SCORES": str(validation["checked_scores"]), "FINDING": finding}
    for key, value in replacements.items():
        tex = tex.replace(f"@@{key}@@", value)
    (out / "report.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    build(parser.parse_args().result_directory)
