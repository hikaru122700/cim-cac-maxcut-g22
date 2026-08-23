"""plot_diversity_report.py — 多様性評価の「読ませる」ための要約図。

plot_diversity.py が出す個別図に対し、こちらは報告書 (RESULTS_visual) 用に
1 枚で結論が伝わる図を作る。

    fig1_overview.png   手法別の多様性と品質を 4 インスタンス分並べた俯瞰図
    fig2_refine.png     共通 TS で磨く前後の距離（結論「磨いても距離は残る」）
    fig3_bigvalley.png  疎 G22 と密 K2000 の big valley 構造の対比
    fig4_backbone.png   磁化プロファイル（骨格の大きさ）
    fig5_topshare.png   上位 25% を誰が占めているか

使い方:
    python scripts/plotting/plot_diversity_report.py <run_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import ALGOS
from scripts.benchmarks.diversity_metrics import (
    to_binary, distance_matrix, cross_distance_matrix, align_to_reference,
)

ORDER = ["CIM", "CAC", "SA", "SB", "PT", "GA"]


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.25)


def main():
    run_dir = Path(sys.argv[1])
    plt = setup_plt()
    raw = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ref = json.loads((run_dir / "results_ref20k.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "signs.npz")
    meta, R, RF = raw["meta"], raw["results"], ref["results"]
    datasets = [d for d in meta["datasets"] if d in R and R[d]]

    # ---------- fig1: 俯瞰（多様性の棒 + gap の棒を左右に並べる） ----------
    fig, axes = plt.subplots(len(datasets), 2, figsize=(11.5, 3.0 * len(datasets)),
                             dpi=140, squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    for k, ds in enumerate(datasets):
        algos = [a for a in ORDER if a in R[ds]]
        dist = [R[ds][a]["mean_pairwise"] for a in algos]
        gap = [R[ds][a]["gap_pct_mean"] for a in algos]
        cols = [ALGOS[a]["color"] for a in algos]
        labs = [ALGOS[a]["label"] for a in algos]
        idx = np.argsort(dist)                      # 似た解を出す順に並べる
        y = np.arange(len(algos))

        ax = axes[k][0]
        ax.barh(y, [dist[i] for i in idx], color=[cols[i] for i in idx],
                alpha=0.85, edgecolor="k", lw=0.5)
        ax.set_yticks(y); ax.set_yticklabels([labs[i] for i in idx], fontsize=9)
        ax.axvline(0.5, color="gray", ls=":", lw=1.3)
        ax.set_xlim(0, 0.54)
        ax.set_xlabel("平均ペアワイズ距離（右ほど多様）")
        ax.set_title(f"{ds}: 出力解の散らばり", fontsize=11)
        for yy, i in zip(y, idx):
            ax.text(dist[i] + 0.006, yy, f"{dist[i]:.3f}", va="center", fontsize=8)
        ax.text(0.5, 0.04, " 無相関の水準", color="gray", fontsize=8,
                transform=ax.get_xaxis_transform(), va="bottom")
        style(ax)

        ax = axes[k][1]
        ax.barh(y, [gap[i] for i in idx], color=[cols[i] for i in idx],
                alpha=0.45, edgecolor="k", lw=0.5, hatch="//")
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlabel("平均 gap [%]（左ほど高品質）")
        ax.set_title(f"{ds}: 解の品質", fontsize=11)
        for yy, i in zip(y, idx):
            ax.text(gap[i] + max(gap) * 0.02, yy, f"{gap[i]:.2f}", va="center", fontsize=8)
        ax.set_xlim(0, max(gap) * 1.25)
        style(ax)
    fig.suptitle("同じ実時間で 32 回解いたときの「散らばり」と「良さ」", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(run_dir / "fig1_overview.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- fig2: 磨く前後 ----------
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.4 * len(datasets), 4.6),
                             dpi=140, squeeze=False)
    for k, ds in enumerate(datasets):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in RF[ds]]
        for a in algos:
            b = RF[ds][a]["mean_pairwise_before"]
            af = RF[ds][a]["mean_pairwise"]
            ax.plot([0, 1], [b, af], "-o", color=ALGOS[a]["color"], lw=2.0, ms=6,
                    label=ALGOS[a]["label"] if k == 0 else None)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["磨く前", "磨いた後"])
        ax.set_xlim(-0.25, 1.25)
        ax.set_ylim(0, 0.52)
        ax.set_ylabel("平均ペアワイズ距離" if k == 0 else "")
        ax.set_title(ds, fontsize=11)
        style(ax)
    fig.legend(loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("共通の Tabu Search で磨いても、解の散らばりはほぼ変わらない", fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(run_dir / "fig2_refine.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- fig3: big valley の対比 ----------
    pair = [d for d in ["G22", "K2000"] if d in datasets]
    fig, axes = plt.subplots(1, len(pair), figsize=(5.6 * len(pair), 5.0), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(pair):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in R[ds]]
        allS = np.concatenate([to_binary(npz[f"{ds}__{a}_signs"]) for a in algos])
        allC = np.concatenate([npz[f"{ds}__{a}_cuts"] for a in algos])
        owner = np.concatenate([[i] * 32 for i in range(len(algos))])
        bks = float(meta["bks"][ds])
        ref_sol = allS[int(np.argmax(allC))]
        dref = cross_distance_matrix(allS, ref_sol[None, :]).ravel()
        gaps = (bks - allC) / bks * 100.0
        for i, a in enumerate(algos):
            sel = owner == i
            ax.scatter(dref[sel], gaps[sel], s=40, color=ALGOS[a]["color"], alpha=0.8,
                       edgecolors="k", lw=0.4, label=ALGOS[a]["label"] if k == 0 else None)
        fdc = met["per_dataset"][ds]["fdc"].get("ALL", float("nan"))
        ax.set_xlabel("見つかった最良解からの距離")
        ax.set_ylabel("BKS からの gap [%]" if k == 0 else "")
        ax.set_xlim(-0.02, 0.52)
        kind = "疎グラフ" if ds != "K2000" else "密グラフ"
        ax.set_title(f"{ds}（{kind}）  相関 = {fdc:.2f}", fontsize=11)
        style(ax)
    fig.legend(loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("良い解は 1 か所に集まるのか、あちこちに散るのか", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(run_dir / "fig3_bigvalley.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- fig4: 磁化プロファイル（骨格の大きさ） ----------
    show = [d for d in ["G22", "K2000"] if d in datasets]
    fig, axes = plt.subplots(1, len(show), figsize=(5.6 * len(show), 4.8), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(show):
        ax = axes[0][k]
        for a in [x for x in ORDER if x in R[ds]]:
            S = to_binary(npz[f"{ds}__{a}_signs"])
            c = npz[f"{ds}__{a}_cuts"]
            A = align_to_reference(S, S[int(np.argmax(c))])
            m = np.abs(2.0 * A.mean(axis=0) - 1.0)
            xs = np.arange(m.size) / m.size * 100.0
            ax.plot(xs, np.sort(m)[::-1], lw=2.0, color=ALGOS[a]["color"],
                    label=ALGOS[a]["label"] if k == 0 else None)
        ax.axhline(0.9, color="gray", ls=":", lw=1.2)
        ax.text(99, 0.915, "「固定」とみなす水準 ", color="gray", fontsize=8, ha="right")
        ax.set_xlabel("頂点（安定な順に並べた百分位）[%]")
        ax.set_ylabel("32 試行での偏り $|m_i|$" if k == 0 else "")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{ds}", fontsize=11)
        style(ax)
    fig.legend(loc="lower center", ncol=6, fontsize=9, frameon=False)
    fig.suptitle("どれだけの頂点が「毎回同じ側」に決まるか（曲線が高いほど骨格が太い）",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    fig.savefig(run_dir / "fig4_backbone.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- fig5: 上位 25% の占有 ----------
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=140)
    xs = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    for a in ORDER:
        vals = []
        for ds in datasets:
            p = met["per_dataset"][ds]["pooled_top25"].get(a, {"n_top": 0})
            vals.append(p["n_top"])
        vals = np.asarray(vals, dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=ALGOS[a]["color"], alpha=0.85,
               edgecolor="k", lw=0.5, label=ALGOS[a]["label"])
        for x, v, b in zip(xs, vals, bottom):
            if v >= 3:
                ax.text(x, b + v / 2, f"{int(v)}", ha="center", va="center", fontsize=9)
        bottom += vals
    ax.set_xticks(xs); ax.set_xticklabels(datasets)
    ax.set_ylabel("上位 25% に入った解の本数（192 本中 約 48 本）")
    ax.set_title("良い解を出しているのは誰か（全手法プールの上位 25%）")
    ax.legend(fontsize=9, ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(run_dir / "fig5_topshare.png", bbox_inches="tight")
    plt.close(fig)

    print(f"fig1..fig5 -> {run_dir}")


if __name__ == "__main__":
    main()
