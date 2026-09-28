"""Matched GA cold/CIM-warm ablation for G22 and G55.

The population size, GA/TS parameters, generation budget, seeds, and RNG
position are matched.  Only population member 0 is replaced by the CIM seed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import matplotlib
import numpy as np
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.GA import simulate_ga_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim


DATASETS = ["G22", "G55"]
BUDGETS = [0, 1, 3, 8, 20, 50]
NUM_TRIALS = 16
CIM_BUDGET = 600


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def stats(cuts, bks):
    cuts = np.asarray(cuts, dtype=float)
    return {
        "cut_max": float(cuts.max()),
        "cut_mean": float(cuts.mean()),
        "cut_median": float(np.median(cuts)),
        "cut_std": float(cuts.std()),
        "gap_best": float(bks - cuts.max()),
        "gap_mean": float(bks - cuts.mean()),
    }


def bootstrap_ci(delta, seed=20260726, samples=5000):
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        means[i] = rng.choice(delta, size=len(delta), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(cold, warm):
    cold = np.asarray(cold, dtype=float)
    warm = np.asarray(warm, dtype=float)
    delta = warm - cold
    p_value = (
        1.0
        if np.all(delta == 0)
        else float(wilcoxon(warm, cold, zero_method="wilcox").pvalue)
    )
    lo, hi = bootstrap_ci(delta)
    return {
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_mean_ci95": [lo, hi],
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
        "wilcoxon_p": p_value,
    }


def run_ga(ctx, budget, seeds, init_signs=None):
    p = PARAMS[ctx.name]["GA"]
    return simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        NUM_TRIALS,
        pop_size=p["pop_size"],
        max_generations=int(budget),
        ts_iters=p["ts_iters"],
        cr=p["cr"],
        alpha_tenure=p["alpha_tenure"],
        beta_quality=p["beta_quality"],
        init_signs=init_signs,
        seeds=seeds,
        align_rng_with_cold=init_signs is not None,
    )


def warmup():
    ctx = load_context("G22")
    seeds = np.arange(2, dtype=np.int64)
    _, cim_signs = run_cim(ctx, 20, 2, seeds)
    p = PARAMS["G22"]["GA"]
    common = dict(
        pop_size=min(3, p["pop_size"]),
        max_generations=1,
        ts_iters=20,
        cr=min(10, p["cr"]),
        alpha_tenure=p["alpha_tenure"],
        beta_quality=p["beta_quality"],
        seeds=seeds,
    )
    simulate_ga_batch(ctx.n, ctx.edges, ctx.weights, 2, **common)
    simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        2,
        **common,
        init_signs=cim_signs,
        align_rng_with_cold=True,
    )


def gap_for_log(value):
    return max(float(value), 0.25)


def plot_budget_aligned(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    for row, dataset in enumerate(DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        positions = np.arange(len(points))

        ax = axes[row, 0]
        cold = np.array([gap_for_log(p["cold"]["gap_best"]) for p in points])
        warm = np.array([gap_for_log(p["warm"]["gap_best"]) for p in points])
        for x, yc, yw in zip(positions, cold, warm):
            ax.plot([x, x], [yc, yw], color="#c8d1d8", lw=1.2, zorder=1)
        ax.scatter(
            positions,
            cold,
            s=75,
            facecolors="white",
            edgecolors="#666666",
            linewidths=2,
            label="cold: random member 0",
            zorder=2,
        )
        ax.scatter(
            positions,
            warm,
            s=75,
            marker="x",
            color="#d62728",
            linewidths=2.4,
            label="warm: CIM member 0",
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_xticks(positions, [str(p["budget"]) for p in points])
        ax.set_xlabel("GA generations (same column = same budget)")
        ax.set_ylabel(f"best gap to BKS={data['bks']}")
        ax.set_title(f"{dataset}: independent best-of-{NUM_TRIALS} at each budget")
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
        ax.legend()

        ax = axes[row, 1]
        means = np.array([p["paired"]["delta_mean"] for p in points])
        cis = np.array([p["paired"]["delta_mean_ci95"] for p in points])
        yerr = np.vstack([means - cis[:, 0], cis[:, 1] - means])
        point_colors = ["#147d64" if value >= 0 else "#d62728" for value in means]
        ax.errorbar(
            positions,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="#596773",
            elinewidth=1.5,
            capsize=4,
            zorder=1,
        )
        ax.scatter(positions, means, c=point_colors, s=60, zorder=2)
        ax.axhline(0, color="#333333", lw=1, ls="--")
        for x, y, point in zip(positions, means, points):
            paired = point["paired"]
            ax.annotate(
                f"{paired['wins']}/{paired['ties']}/{paired['losses']}",
                (x, y),
                xytext=(0, 9 if y >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xticks(positions, [str(p["budget"]) for p in points])
        ax.set_xlabel("GA generations")
        ax.set_ylabel("mean paired cut gain (warm - cold)")
        ax.set_title(f"{dataset}: paired effect, 95% bootstrap CI (W/T/L)")
        ax.grid(True, axis="y", ls=":", alpha=0.35)

    fig.suptitle(
        "Matched GA ablation: only population member 0 differs",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "ga_matched_budget_aligned.png", bbox_inches="tight")
    plt.close(fig)


def plot_solver_time_aligned(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180)
    for ax, dataset in zip(axes, DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        paired_x = np.array(
            [
                (p["cold"]["refiner_time"] + p["warm"]["refiner_time"]) / 2
                for p in points
            ]
        )
        cold = np.array([gap_for_log(p["cold"]["gap_best"]) for p in points])
        warm = np.array([gap_for_log(p["warm"]["gap_best"]) for p in points])
        for x, yc, yw in zip(paired_x, cold, warm):
            ax.plot([x, x], [yc, yw], color="#c8d1d8", lw=1.2, zorder=1)
        ax.scatter(
            paired_x,
            cold,
            s=70,
            facecolors="white",
            edgecolors="#666666",
            linewidths=2,
            label="cold",
            zorder=2,
        )
        ax.scatter(
            paired_x,
            warm,
            s=70,
            marker="x",
            color="#d62728",
            linewidths=2.3,
            label="CIM warm",
            zorder=3,
        )
        for x, y, point in zip(paired_x, np.maximum(cold, warm), points):
            ax.annotate(
                f"{point['budget']} gen",
                (x, y),
                xytext=(4, 5),
                textcoords="offset points",
                fontsize=7.5,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("paired mean GA wall time [s] (CIM time excluded)")
        ax.set_ylabel(f"best gap to BKS={data['bks']}")
        ax.set_title(f"{dataset}: same budget shares one x column")
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend()

    fig.suptitle(
        "Matched GA: solver time only; independent budget points are not a trajectory",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "ga_matched_solver_time_aligned.png", bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = (
        ROOT
        / "results"
        / date.today().isoformat()
        / "cim_warmstart"
        / "ga_matched_v1_G22_G55_nt16"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    warmup()
    seeds = np.arange(NUM_TRIALS, dtype=np.int64)
    results = {
        "meta": {
            "datasets": DATASETS,
            "budgets": BUDGETS,
            "num_trials": NUM_TRIALS,
            "cim_budget": CIM_BUDGET,
            "comparison": (
                "population size, GA/TS parameters, generations, seeds, and "
                "post-initialization RNG position are matched; only member 0 differs"
            ),
            "plot_note": (
                "each budget is an independent experiment, not one cumulative trajectory"
            ),
        },
        "datasets": {},
    }
    raw = {}

    for dataset in DATASETS:
        ctx = load_context(dataset)
        p = PARAMS[dataset]["GA"]
        cim_result, cim_time = timed(
            lambda: run_cim(ctx, CIM_BUDGET, NUM_TRIALS, seeds)
        )
        _, cim_signs = cim_result
        cim_cuts = ctx.score(cim_signs)
        log(
            f"{dataset} CIM: {cim_time:.3f}s "
            f"best={cim_cuts.max():.0f} mean={cim_cuts.mean():.1f}"
        )
        points = []
        raw[f"{dataset}_cim"] = cim_cuts

        for budget in BUDGETS:
            cold_result, cold_time = timed(
                lambda b=budget: run_ga(ctx, b, seeds)
            )
            warm_result, warm_time = timed(
                lambda b=budget: run_ga(ctx, b, seeds, cim_signs)
            )
            cold_cuts = np.asarray(cold_result[0])
            warm_cuts = np.asarray(warm_result[0])
            paired = paired_stats(cold_cuts, warm_cuts)
            point = {
                "budget": budget,
                "cold": {
                    "refiner_time": cold_time,
                    "total_time": cold_time,
                    **stats(cold_cuts, ctx.bks),
                },
                "warm": {
                    "refiner_time": warm_time,
                    "cim_time": cim_time,
                    "total_time": cim_time + warm_time,
                    **stats(warm_cuts, ctx.bks),
                },
                "paired": paired,
            }
            points.append(point)
            raw[f"{dataset}_{budget}_cold"] = cold_cuts
            raw[f"{dataset}_{budget}_warm"] = warm_cuts
            log(
                f"{dataset} g={budget}: cold={cold_cuts.max():.0f}, "
                f"warm={warm_cuts.max():.0f}, mean_delta={paired['delta_mean']:+.1f}, "
                f"W/T/L={paired['wins']}/{paired['ties']}/{paired['losses']}, "
                f"p={paired['wilcoxon_p']:.4g}"
            )

        results["datasets"][dataset] = {
            "bks": ctx.bks,
            "params": p,
            "cim": {"time": cim_time, **stats(cim_cuts, ctx.bks)},
            "points": points,
        }

    with (out_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "cuts.npz", **raw)
    plot_budget_aligned(results, out_dir)
    plot_solver_time_aligned(results, out_dir)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
