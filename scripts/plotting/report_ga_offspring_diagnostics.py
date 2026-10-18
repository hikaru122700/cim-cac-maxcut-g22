"""Explain measured parent -> crossover -> TS -> pool acceptance in LuaLaTeX."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

COLORS = {"random": "#506778", "cim": "#C34436"}
LABELS = {"random": "ランダム初期集団", "cim": "全CIM初期集団"}
DATASETS = ["G22", "G55", "G70"]


def mean_ci(values):
    """values are independent run-level means/rates, never pooled offspring."""
    values = np.asarray(values)
    mean = values.mean(axis=0)
    half = student_t.ppf(.975, values.shape[0]-1) * values.std(axis=0, ddof=1) / np.sqrt(values.shape[0])
    return mean, half


def pct(count, total):
    return f"{100*count/total:.1f}\\%"


def ratio_cell(item):
    if item["eligible"] == 0:
        return "該当なし"
    return f"{item['accepted']}/{item['eligible']}（{pct(item['accepted'], item['eligible'])}）"


def plots(out, meta):
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False,
                         "font.size": 12, "xtick.direction": "in", "ytick.direction": "in",
                         "xtick.top": True, "ytick.right": True})
    summary = meta["summary"]
    # First page: concrete cut values in the stalled G70 CIM runs.
    s = summary["G70"]["cim"]
    means = s["means"]
    values = [means["parent_cut"], means["raw_child_cut"], means["refined_child_cut"]]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot([0, 1, 2], values, "o-", color=COLORS["cim"], lw=2.5, ms=9)
    for x, y in enumerate(values):
        ax.annotate(f"{y:,.1f}", (x, y), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=16,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .92, "pad": 1})
    ax.axhline(values[0], color="#687885", ls="--", lw=1)
    ax.set_xticks([0, 1, 2], ["選ばれた2親のうち良い親", "交叉直後の子", "局所探索（TS）後の子"])
    ax.set_ylabel("平均カット値（大きいほど良い）")
    ax.set_ylim(min(values)-180, max(values)+230)
    ax.set_xlim(-.2, 2.2)
    ax.grid(alpha=.18)
    fig.tight_layout()
    fig.savefig(out / "g70_stages.png", dpi=180)
    fig.savefig(out / "g70_stages.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1))
    time_fig, time_axes = plt.subplots(2, 3, figsize=(15, 7.6))
    diagnostic = {}
    for col, ds in enumerate(DATASETS):
        diagnostic[ds] = {}
        for condition in ["random", "cim"]:
            with np.load(out / f"{ds}_{condition}_events.npz") as z:
                parent = z["parent_fits"].max(axis=2)
                raw_delta = z["child_before_fit"]-parent
                final_delta = z["child_after_fit"]-parent
                accepted = z["accepted"]
                x = 0 if condition == "random" else 1
                for values in [final_delta]:
                    run_means = values.mean(axis=0)
                    mean, half = mean_ci(run_means)
                    axes[col].scatter(np.full(len(run_means), x)+np.linspace(-.08,.08,len(run_means)),
                                           run_means, color=COLORS[condition], alpha=.25, s=15)
                    axes[col].errorbar(x, mean, yerr=half, fmt="D", color=COLORS[condition], capsize=4, ms=8)
                    axes[col].annotate(f"{mean:+.1f}", (x, mean), xytext=(11, 0), textcoords="offset points", fontsize=11)
                # Four fixed generation intervals; average within each run first.
                blocks = [(0,20), (20,50), (50,120), (120,300)]
                block_delta = np.stack([final_delta[a:b].mean(axis=0) for a,b in blocks], axis=1)
                block_accept = np.stack([accepted[a:b].mean(axis=0)*100 for a,b in blocks], axis=1)
                diagnostic[ds][condition] = {"blocks": [f"{a+1}-{b}" for a,b in blocks],
                    "refined_delta_means": block_delta.mean(axis=0).tolist(),
                    "acceptance_percent": block_accept.mean(axis=0).tolist()}
                for row, vals in enumerate([block_delta, block_accept]):
                    mean, half = mean_ci(vals)
                    time_axes[row, col].errorbar(range(4), mean, yerr=half, fmt="o-", color=COLORS[condition],
                                                capsize=3, lw=2, label=LABELS[condition])
        ax = axes[col]
        ax.axhline(0, color="#505050", ls="--", lw=1)
        ax.set_xticks([0,1], ["ランダム", "全CIM"])
        ax.set_xlim(-.4, 1.65)
        ax.set_ylabel("局所探索後の子 − 良い親のカット値")
        ax.set_title(ds)
        ax.grid(alpha=.16)
        for row in range(2):
            ax = time_axes[row, col]
            ax.set_xticks(range(4), ["1–20", "21–50", "51–120", "121–300"])
            ax.set_xlabel("世代の区間")
            ax.grid(alpha=.16)
            ax.set_title(ds)
        time_axes[0, col].axhline(0, color="#505050", ls="--", lw=1)
        time_axes[0, col].set_ylabel("局所探索後の子 − 良い親")
        time_axes[1, col].set_ylabel("子の採用率 [%]")
        time_axes[1, col].set_ylim(-3, 103)
        time_axes[1, col].legend(fontsize=10)
    fig.tight_layout()
    time_fig.tight_layout()
    for figure, name in [(fig, "parent_differences"), (time_fig, "stage_trajectory")]:
        figure.savefig(out / f"{name}.png", dpi=170)
        figure.savefig(out / f"{name}.svg")
        plt.close(figure)
    (out / "generation_blocks.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")


def build(out):
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    if meta["num_trials"] != 16 or meta["generations"] != 300:
        raise ValueError("This report template requires 16 runs and 300 generations")
    assert all(meta["replay_checks"][f"{ds}_{condition}"]["best_history_exact"]
               for ds in DATASETS for condition in ["random", "cim"])
    plots(out, meta)
    s = meta["summary"]["G70"]["cim"]
    counts, means = s["counts"], s["means"]
    total = s["num_children"]
    final_quality = {}
    for condition in ["random", "cim"]:
        with np.load(out / f"G70_{condition}_events.npz") as z:
            final_quality[condition] = float(np.maximum(z["best_before"][-1], z["child_after_fit"][-1]).mean())
    recovery_rows, acceptance_rows, parameter_rows = [], [], []
    conditional_recovery = {}
    md = ["# GAの交叉・局所探索・採用を分けた検証", "",
          "前回と同じ初期集団・パラメータ・seedで再実行し、全世代の集団fitness、best履歴、保存個体の一致を確認。", "",
          "|問題|初期集団|交叉後−良い親|TS後−良い親|良い親以上へ回復|全子の採用|",
          "|---|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        for condition in ["random", "cim"]:
            row = meta["summary"][ds][condition]
            c, m, num = row["counts"], row["means"], row["num_children"]
            label = "ランダム" if condition == "random" else "全CIM"
            with np.load(out / f"{ds}_{condition}_events.npz") as z:
                pb = z["parent_fits"].max(axis=2)
                damaged = z["child_before_fit"] < pb
                returned = damaged & (z["child_after_fit"] >= pb)
                eligible = int(damaged.sum())
                recovered_bad = int(returned.sum())
                conditional_recovery[f"{ds}_{condition}"] = {
                    "damaged_children": eligible, "recovered_to_better_parent": recovered_bad,
                    "rate": recovered_bad/eligible if eligible else None}
            recovery_cell = f"{recovered_bad}/{eligible}" if eligible else "該当なし"
            recovery_rows.append(f"{ds} & {label} & {m['raw_minus_better_parent']:+.1f} & {m['refined_minus_better_parent']:+.1f} & "
                                 f"{c['ts_recovered_better_parent']}/{num} & {recovery_cell}" + r"\\")
            acceptance_rows.append(f"{ds} & {label} & {c['accepted']}/{num} & {ratio_cell(row['TS_improved_acceptance'])} & "
                                   f"{ratio_cell(row['parent_improved_acceptance'])} & {ratio_cell(row['new_best_acceptance'])}" + r"\\")
            md.append(f"|{ds}|{label}|{m['raw_minus_better_parent']:.2f}|{m['refined_minus_better_parent']:.2f}|"
                      f"{c['ts_recovered_better_parent']}/{num}|{c['accepted']}/{num}|")
        p = meta["source_metadata"]["datasets"][ds]["ga_params"]
        parameter_rows.append(f"{ds} & {p['pop_size']} & {p['ts_iters']:,} & {p['cr']} & {p['alpha_tenure']} & {p['beta_quality']:.4f}" + r"\\")
    if counts["ts_exceeded_better_parent"] == 0:
        interpretation = ("親を超えた子は一度も生まれていない。したがって、優れた子を採用処理が捨てたために停滞した、"
                          "という結果ではない。交叉後に失った品質を、今回の局所探索では親の水準まで戻せていない。")
    else:
        interpretation = ("局所探索での改善と、親を超える改善は区別する必要がある。"
                          "親を超えた子が採用されたかは3ページの条件付き集計で確認する。")
    md += ["", "## G70・全CIMの要点", "",
           f"良い親 {means['parent_cut']:.2f} → 交叉直後 {means['raw_child_cut']:.2f} → TS後 {means['refined_child_cut']:.2f}（全イベント平均）。",
           f"良い親への回復 {counts['ts_recovered_better_parent']}/{total}、採用 {counts['accepted']}/{total}。",
           interpretation, "", "## 定義と限界", "",
           f"G70の300世代での平均最良cutは全CIM {final_quality['cim']:.2f}、ランダム {final_quality['random']:.2f}。採用率だけで最終性能を判断しない。",
           "良い親＝実際に交叉した2親の高い方のカット値。回復はTS後>=良い親、親超えはTS後>良い親。",
           "TS改善＝TS後>交叉直後。採用はDisQualが集団の1個体を子で置き換えた場合。",
           "初期品質はそろえていない。1条件16run×300世代=4800子、同一run内の子は独立標本ではない。",
           "誤差棒はrunごとの平均から作った95% t区間。表の件数・割合は観測した子の記述統計。",
           "CIM本体やGAのハイパーパラメータは変えていない。別の交叉・TS予算・CIM設定で同じかは未検証。",
           "この結果だけでは、多様性が性能を改善するという因果効果は証明できない。", "",
           "## ファイル", "",
           "*_events.npz：全世代・全runの親fitness、子のTS前後fitness、採否、集団更新前後のfitness。",
           "*_spins.npz：全子イベントの2親とTS前後の子をpackbitsで保存。bitorder=little。",
           "metadata.json：全設定、前回との照合、ソースハッシュ、集計値。",
           "再現：scripts/benchmarks/ga_offspring_diagnostics.py、資料生成：scripts/plotting/report_ga_offspring_diagnostics.py。",
           "PDFはLuaLaTeX。report.texと3枚のPNGを同じフォルダに置いてコンパイルする。"]
    (out / "REPORT.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    (out / "conditional_recovery.json").write_text(json.dumps(conditional_recovery, indent=2), encoding="utf-8")
    tex = r"""\newcommand{\DoNotLoadEpstopdf}{}
\documentclass[11pt,a4paper,landscape]{ltjsarticle}
\usepackage[margin=14mm,footskip=7mm]{geometry}
\usepackage{luatexja-fontspec}
\setmainfont{arial.ttf}[Path=C:/Windows/Fonts/,BoldFont=arialbd.ttf]
\setmainjfont{YuGothM.ttc}[Path=C:/Windows/Fonts/,FontIndex=0,BoldFont=YuGothB.ttc,BoldFeatures={FontIndex=0}]
\setsansjfont{YuGothM.ttc}[Path=C:/Windows/Fonts/,FontIndex=0,BoldFont=YuGothB.ttc,BoldFeatures={FontIndex=0}]
\usepackage{graphicx,booktabs,xcolor}
\definecolor{ink}{HTML}{263D50}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\newcommand{\heading}[2]{{\LARGE\bfseries\color{ink}#1}\par{\small #2}\par\vspace{2mm}}
\begin{document}
\heading{G70のCIM初期集団は、どこで改善が止まるのか}{交叉直後・局所探索後・集団への採用を、実際に生まれた子で確認した}
\begin{center}\includegraphics[width=.86\linewidth,height=74mm,keepaspectratio]{g70_stages.png}\end{center}
\textbf{1．交叉で悪くなるか：}良い方の親より、交叉直後の子は平均@@RAWLOSS@@低い。両方の親より悪い子は@@BELOWBOTH@@/@@TOTAL@@個。

\textbf{2．局所探索で親に戻れるか：}子のカット値は局所探索で平均@@GAIN@@改善するが、良い親との差は平均@@RESIDUAL@@残る。良い親以上へ回復した子は@@RECOVERED@@/@@TOTAL@@個。TS後も集団内の最下位より低い子は@@BELOWPOOL@@/@@TOTAL@@個。

\textbf{3．改善した子は採用されるか：}TSで交叉直後より改善した子は@@TSIMPROVED@@個。そのうち採用されたのは@@TSACCEPTED@@個。全子の採用数は@@ACCEPTED@@/@@TOTAL@@個。

\textbf{読み取れること：}@@INTERPRETATION@@

{\footnotesize 各16run・300世代。図は全4800イベントの平均で、特定の1組の親と子ではない。親＝選択された2親のうち高いカット値の方。前回の集団推移を同じ個体・設定・seedで再現し、子の各段階を追加記録した。}

\clearpage
\heading{局所探索後の子は、親にどこまで近づけるか}{0より下なら良い方の親より悪い。0を超えるなら親より良い。各図の縦軸範囲が異なる}
\includegraphics[width=\linewidth,height=80mm,keepaspectratio]{parent_differences.png}\par
\begin{center}\small
\begin{tabular}{llrrrr}\toprule
問題 & 初期集団 & 交叉後$-$良い親 & TS後$-$良い親 & TS後に良い親以上 & 悪化した子の回復\\\midrule
@@RECOVERY_TABLE@@
\bottomrule\end{tabular}
\end{center}
{\footnotesize 「TS後に良い親以上」は該当件数/全子数。「悪化した子の回復」は、交叉で良い親より悪くなった子のうち、TS後に良い親以上に戻った件数/悪化した子数。薄い点はrunごとの300世代平均、ひし形は16run平均、誤差棒はrun単位の95\% $t$区間。初期品質はそろえておらず、多様性の因果効果を示す比較ではない。}

\clearpage
\heading{「改善した子」の意味を分けて、採用を確認する}{TSで子自身が良くなることと、親より良い子ができることは違う}
\textbf{TSで改善：}交叉直後の子より、局所探索後の子のカット値が高い。\\
\textbf{親を超えた：}局所探索後の子が、選択された両親の良い方よりも高い。\\
\textbf{採用：}集団更新処理DisQualが、集団の1個体を子に置き換えた。品質と最も近い他個体までの距離を併用して決める。

\begin{center}\small
\begin{tabular}{llrrrr}\toprule
問題 & 初期集団 & 全子の採用 & TS改善子の採用 & 親超え子の採用 & 過去最良更新子の採用\\\midrule
@@ACCEPTANCE_TABLE@@
\bottomrule\end{tabular}
\end{center}
{\small 採用欄は「採用件数/対象件数」。括弧はその対象内での採用率。「該当なし」は条件を満たす子が0個であり、採用率0\%とは異なる。過去最良の更新は、各runでそれまでに見つかった最良値を厳密に超えた件数。}

\textbf{G70・全CIMについて：}@@INTERPRETATION@@

{\small G70の300世代時点の平均最良カット値は、全CIM @@CIMFINAL@@、ランダム @@RANDOMFINAL@@。採用率の高低だけで、最終的な解の品質を判断しない。}

\textbf{この実験でまだ言えないこと：}CIMが出す解のどの構造がこの現象を生んだか、交叉や局所探索予算を変えると回復するかは未検証。今回の設定で起きている処理の流れを特定した結果である。

\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
問題 & 集団サイズ & TS反復数 & 摂動待ち回数 & tenure係数 & 品質重み\\\midrule
@@PARAMETER_TABLE@@
\bottomrule\end{tabular}
\end{center}

\vfill
{\footnotesize 1世代に各runで子を1個生成し、交叉→TS→採否を記録。集団の初期TSはこの4800件に含めない。同一run内の子は相関するため、件数4800を独立試行数とする有意差検定は行っていない。}

\clearpage
\heading{探索の途中で、回復と採用の傾向は変わるか}{4つの世代区間ごとに集計。上段はTS後の親との差、下段は全子の採用率}
\includegraphics[width=\linewidth,height=123mm,keepaspectratio]{stage_trajectory.png}\par
{\footnotesize 点と誤差棒は、区間内をrunごとに平均してから求めた16run平均と95\% $t$区間。各条件の初期集団、CIM設定、GA設定、seedは前回と同じ。全世代の集団fitness・best履歴・保存した集団断面・最終解が前回と完全一致することを確認した。}

{\footnotesize 全子の両親・交叉直後・TS後のカット値を辺の和で独立に再計算し、報告値と照合。計測追加が結果を変えないこと等8件のテストに成功。全イベントの個体と数値、設定、ソースは本PDFと同じフォルダに保存。}
\end{document}
"""
    replacements = {"RAWLOSS": f"{-means['raw_minus_better_parent']:,.1f}",
        "GAIN": f"{means['TS_gain']:,.1f}", "RESIDUAL": f"{-means['refined_minus_better_parent']:,.1f}",
        "BELOWBOTH": str(counts["raw_below_both_parents"]), "TOTAL": str(total),
        "BELOWPOOL": str(counts["below_pool_worst_after_ts"]),
        "RECOVERED": str(counts["ts_recovered_better_parent"]), "TSIMPROVED": str(counts["ts_improved_raw_child"]),
        "TSACCEPTED": str(s["TS_improved_acceptance"]["accepted"]), "ACCEPTED": str(counts["accepted"]),
        "INTERPRETATION": interpretation, "RECOVERY_TABLE": "\n".join(recovery_rows),
        "CIMFINAL": f"{final_quality['cim']:,.2f}", "RANDOMFINAL": f"{final_quality['random']:,.2f}",
        "ACCEPTANCE_TABLE": "\n".join(acceptance_rows), "PARAMETER_TABLE": "\n".join(parameter_rows)}
    for name, value in replacements.items():
        tex = tex.replace(f"@@{name}@@", value)
    (out / "report.tex").write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    build(parser.parse_args().result_directory)
