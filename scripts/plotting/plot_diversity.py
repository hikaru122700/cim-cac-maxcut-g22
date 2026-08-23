"""plot_diversity.py — diversity_bench.py の出力から多様性の図表を作る。

使い方(プロジェクトルートから):
    python scripts/plotting/plot_diversity.py results/<date>/solution_diversity/v{N}_...

出力(同じ run ディレクトリ直下):
    <DS>_pairwise.png       手法別ペアワイズ距離の箱ひげ
    <DS>_mds.png            全解の 2 次元 MDS 埋め込み(色=手法)
    <DS>_frozen.png         固定頂点率とエントロピー
    <DS>_cross.png          手法間の平均距離ヒートマップ
    quality_diversity.png   品質(gap%) × 多様性(平均距離) 散布図(全データセット)
    summary.csv             指標一覧
    metrics.json            手法間距離を含む全指標
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import ALGOS, load_context
from scripts.benchmarks.diversity_metrics import (
    distance_matrix, cross_distance_matrix, classical_mds,
    align_to_reference, to_binary,
)


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


def local_opt_stats(ctx, S: np.ndarray) -> tuple[float, float]:
    """解集合 S (T, n) の 1-flip 局所最適性。

    x = 2s-1 (±1) とすると、頂点 v を反転したときのカット変化は
    gain_v = x_v * (A x)_v(A は重み付き隣接行列)。gain > 0 の頂点が
    1 つでもあれば局所最適ではない。

    Returns (局所最適だった試行の割合, 改善可能な頂点数の平均)。
    """
    from scipy.sparse import csr_matrix
    ip, ix, dt = ctx._csr
    A = csr_matrix((dt, ix, ip), shape=(ctx.n, ctx.n))
    X = (2.0 * S.astype(np.float64) - 1.0)
    G = X * (A @ X.T).T
    n_improving = (G > 1e-9).sum(axis=1)
    return float((n_improving == 0).mean()), float(n_improving.mean())


def load_run(run_dir: Path, sfx: str):
    with open(run_dir / f"results{sfx}.json", encoding="utf-8") as f:
        payload = json.load(f)
    npz = np.load(run_dir / f"signs{sfx}.npz")
    return payload, npz


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--suffix", default="",
                    help='"" = 生の出力 / "_refined" 等 = 磨いた後のファイル群')
    ap.add_argument("--title-note", default="（共通TSで磨いた後）",
                    help="suffix が空でないときに図タイトルへ添える注記")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    tag = args.suffix
    ttl = "" if not tag else args.title_note
    plt = setup_plt()
    payload, npz = load_run(run_dir, tag)
    meta, results = payload["meta"], payload["results"]
    datasets = [d for d in meta["datasets"] if d in results and results[d]]

    metrics = {"meta": meta, "per_dataset": {}}
    rows = []

    for ds in datasets:
        bks = float(meta["bks"][ds])
        ctx = load_context(ds)
        algos = [a for a in meta["algos"] if a in results[ds]]
        S = {a: to_binary(npz[f"{ds}__{a}_signs"]) for a in algos}
        C = {a: npz[f"{ds}__{a}_cuts"] for a in algos}
        labels = [ALGOS[a]["label"] for a in algos]
        colors = [ALGOS[a]["color"] for a in algos]

        # ---------- 1. ペアワイズ距離の箱ひげ ----------
        dists = []
        for a in algos:
            D = distance_matrix(S[a])
            iu = np.triu_indices(D.shape[0], k=1)
            dists.append(D[iu])
        fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=140)
        bp = ax.boxplot(dists, tick_labels=labels, patch_artist=True, widths=0.6,
                        medianprops=dict(color="k", lw=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for i, dv in enumerate(dists):
            x = np.random.default_rng(0).normal(i + 1, 0.06, size=dv.size)
            ax.plot(x, dv, ".", color="k", ms=2.0, alpha=0.25)
        ax.axhline(0.5, color="gray", ls=":", lw=1.2)
        ax.text(0.02, 0.5, "ランダム分割どうしの水準 0.5", transform=ax.get_yaxis_transform(),
                va="bottom", fontsize=8, color="gray")
        ax.set_ylabel("正規化ハミング距離（全体反転を同一視）")
        ax.set_title(f"{ds}: 独立試行どうしの解の隔たり（大きいほど多様）{ttl}")
        ax.set_ylim(-0.02, 0.55)
        style(ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{ds}_pairwise{tag}.png", bbox_inches="tight")
        plt.close(fig)

        # ---------- 2. MDS 埋め込み ----------
        allS = np.concatenate([S[a] for a in algos], axis=0)
        allC = np.concatenate([C[a] for a in algos], axis=0)
        owner = np.concatenate([[i] * S[a].shape[0] for i, a in enumerate(algos)])
        D_all = distance_matrix(allS)
        emb = classical_mds(D_all, 2)
        gaps = (bks - allC) / bks * 100.0
        g_lo, g_hi = gaps.min(), gaps.max()
        rng_g = max(g_hi - g_lo, 1e-9)
        fig, ax = plt.subplots(figsize=(7.6, 6.4), dpi=140)
        for i, a in enumerate(algos):
            sel = owner == i
            sz = 25 + 200 * (1.0 - (gaps[sel] - g_lo) / rng_g)   # 良い解ほど大きく
            ax.scatter(emb[sel, 0], emb[sel, 1], s=sz, color=ALGOS[a]["color"],
                       alpha=0.75, edgecolors="k", linewidths=0.4,
                       label=ALGOS[a]["label"])
        ax.set_xlabel("MDS 第 1 軸")
        ax.set_ylabel("MDS 第 2 軸")
        ax.set_title(f"{ds}: 解空間上の配置（距離行列の MDS, 点の大きさ=解の良さ）{ttl}")
        ax.legend(fontsize=8, ncol=2)
        style(ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{ds}_mds{tag}.png", bbox_inches="tight")
        plt.close(fig)

        # ---------- 3. 固定頂点率とエントロピー ----------
        frozen = [results[ds][a]["frozen_frac"] * 100 for a in algos]
        ent = [results[ds][a]["entropy_bits"] for a in algos]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), dpi=140)
        axes[0].bar(labels, frozen, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
        axes[0].set_ylabel("固定頂点の割合 [%]")
        axes[0].set_title(f"{ds}: 全試行でほぼ同じ側に決まる頂点")
        axes[1].bar(labels, ent, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
        axes[1].set_ylabel("頂点あたりの平均エントロピー [bit]")
        axes[1].set_title(f"{ds}: 解のばらつきの情報量（1 bit = 完全にランダム）")
        for ax in axes:
            ax.tick_params(axis="x", rotation=20)
            style(ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{ds}_frozen{tag}.png", bbox_inches="tight")
        plt.close(fig)

        # ---------- 4. 手法間の距離 ----------
        K = len(algos)
        M = np.zeros((K, K))
        Mmin = np.zeros((K, K))
        for i, a in enumerate(algos):
            for j, b in enumerate(algos):
                if i == j:
                    D = distance_matrix(S[a])
                    iu = np.triu_indices(D.shape[0], k=1)
                    M[i, j] = D[iu].mean()
                    Mmin[i, j] = D[iu].min()
                else:
                    X = cross_distance_matrix(S[a], S[b])
                    M[i, j] = X.mean()
                    Mmin[i, j] = X.min()
        fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=140)
        im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=max(0.35, M.max()))
        ax.set_xticks(range(K)); ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_yticks(range(K)); ax.set_yticklabels(labels)
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="w" if M[i, j] < 0.25 else "k", fontsize=8)
        fig.colorbar(im, ax=ax, label="平均正規化距離")
        ax.set_title(f"{ds}: 手法間の解の隔たり（対角=手法内）{ttl}")
        fig.tight_layout()
        fig.savefig(run_dir / f"{ds}_cross{tag}.png", bbox_inches="tight")
        plt.close(fig)

        # ---------- 5. 品質 × 「最良解からの距離」(big valley 構造) ----------
        best_idx = int(np.argmax(allC))
        ref = allS[best_idx]
        d_ref = cross_distance_matrix(allS, ref[None, :]).ravel()
        fdc = {}
        fig, ax = plt.subplots(figsize=(7.6, 5.6), dpi=140)
        for i, a in enumerate(algos):
            sel = owner == i
            ax.scatter(d_ref[sel], gaps[sel], s=42, color=ALGOS[a]["color"],
                       alpha=0.8, edgecolors="k", lw=0.4, label=ALGOS[a]["label"])
            if sel.sum() >= 3 and np.std(d_ref[sel]) > 1e-12 and np.std(gaps[sel]) > 1e-12:
                fdc[a] = float(np.corrcoef(d_ref[sel], gaps[sel])[0, 1])
            else:
                fdc[a] = float("nan")
        if np.std(d_ref) > 1e-12:
            fdc["ALL"] = float(np.corrcoef(d_ref, gaps)[0, 1])
        ax.set_xlabel("見つかった最良解からの距離（正規化ハミング）")
        ax.set_ylabel("BKS からの gap [%]")
        ax.set_title(f"{ds}: 解の良さと最良解からの隔たり{ttl}")
        ax.legend(fontsize=8, ncol=2)
        style(ax)
        fig.tight_layout()
        fig.savefig(run_dir / f"{ds}_fdc{tag}.png", bbox_inches="tight")
        plt.close(fig)

        # ---------- 6. 全手法プールの上位 25% に絞った多様性 ----------
        thr = np.percentile(allC, 75)
        top = allC >= thr
        pooled = {}
        for i, a in enumerate(algos):
            sel = (owner == i) & top
            k = int(sel.sum())
            if k >= 2:
                Dt = distance_matrix(allS[sel])
                iut = np.triu_indices(k, k=1)
                pooled[a] = {"n_top": k, "mean_pairwise": float(Dt[iut].mean())}
            else:
                pooled[a] = {"n_top": k, "mean_pairwise": float("nan")}
        Dtop = distance_matrix(allS[top])
        iut = np.triu_indices(int(top.sum()), k=1)

        metrics["per_dataset"][ds] = {
            "algos": algos,
            "cross_mean": M.tolist(),
            "cross_min": Mmin.tolist(),
            "fdc": fdc,
            "pooled_top25_cut_threshold": float(thr),
            "pooled_top25": pooled,
            "pooled_top25_all_mean_pairwise": float(Dtop[iut].mean()),
            "best_cut_pooled": float(allC.max()),
        }
        for a in algos:
            r = dict(results[ds][a]); r["dataset"] = ds; r["algo"] = a
            lo, ni = local_opt_stats(ctx, S[a])
            r["frac_local_opt"] = lo
            r["mean_improving_vertices"] = ni
            metrics["per_dataset"][ds].setdefault("local_opt", {})[a] = [lo, ni]
            rows.append(r)

    # ---------- 7. 品質 × 多様性(全データセット) ----------
    ncol = min(2, len(datasets))
    nrow = int(np.ceil(len(datasets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 5.0 * nrow), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(datasets):
        ax = axes[k // ncol][k % ncol]
        for a in [x for x in meta["algos"] if x in results[ds]]:
            r = results[ds][a]
            ax.scatter(r["gap_pct_mean"], r["mean_pairwise"], s=110,
                       color=ALGOS[a]["color"], edgecolors="k", lw=0.6,
                       label=ALGOS[a]["label"])
            ax.annotate(ALGOS[a]["label"], (r["gap_pct_mean"], r["mean_pairwise"]),
                        textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.set_xlabel("平均 gap [%]（小さいほど高品質）")
        ax.set_ylabel("平均ペアワイズ距離（大きいほど多様）")
        ax.set_title(f"{ds}: 品質と多様性{ttl}")
        style(ax)
    for k in range(len(datasets), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / f"quality_diversity{tag}.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- 出力 ----------
    keys = ["dataset", "algo", "budget", "time", "num_trials", "n",
            "cut_mean", "cut_max", "cut_std", "gap_pct_mean", "gap_pct_best",
            "mean_pairwise", "median_pairwise", "min_pairwise", "max_pairwise",
            "n_distinct", "n_distinct_cuts", "frozen_frac", "entropy_bits",
            "n_near", "near_mean_pairwise", "near_n_distinct",
            "gap_pct_mean_before", "mean_pairwise_before", "refine_time",
            "travel_dist", "frac_local_opt", "mean_improving_vertices"]
    with open(run_dir / f"summary{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(run_dir / f"metrics{tag}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(f"figures + summary.csv + metrics.json -> {run_dir}")


if __name__ == "__main__":
    main()
