"""Matched SA cold/warm ablation.

Cold and CIM-warm SA use exactly the same dataset-specific temperature
schedule, iteration budget, seeds, and post-initialization RNG stream.  The
only intended difference is the initial spin configuration.
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

from modules.SA import simulate_sa_batch, simulate_sa_warm
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim


DATASETS = ["G22", "G55"]
BUDGETS = [
    100_000,
    300_000,
    1_000_000,
    3_000_000,
    10_000_000,
    30_000_000,
    100_000_000,
    300_000_000,
]
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


def bootstrap_ci(delta, seed=20260724, samples=5000):
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
    if np.all(delta == 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(warm, cold, zero_method="wilcox").pvalue)
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


def warmup():
    ctx = load_context("G22")
    seeds = np.arange(2, dtype=np.int64)
    _, cim_signs = run_cim(ctx, 20, 2, seeds)
    p = PARAMS["G22"]["SA"]
    simulate_sa_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        20,
        2,
        t_start=p["t_start"],
        t_end=p["t_end"],
        seeds=seeds,
    )
    simulate_sa_warm(
        ctx.n,
        ctx.edges,
        ctx.weights,
        cim_signs,
        20,
        t_start=p["t_start"],
        t_end=p["t_end"],
        seeds=seeds,
        align_rng_with_cold=True,
    )


def plot_results(results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    for row, dataset in enumerate(DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        bks = data["bks"]
        temperature = data["temperature"]

        ax = axes[row, 0]
        cold_x = [x["cold"]["total_time"] for x in points]
        warm_x = [x["warm"]["total_time"] for x in points]
        cold_y = [max(x["cold"]["gap_best"], 0.25) for x in points]
        warm_y = [max(x["warm"]["gap_best"], 0.25) for x in points]
        ax.plot(cold_x, cold_y, "-o", color="#777777", lw=2, label="cold: random init")
        ax.plot(warm_x, warm_y, "-o", color="#d62728", lw=2, label="warm: CIM init")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("end-to-end wall time [s]")
        ax.set_ylabel(f"best gap to BKS={bks}")
        ax.set_title(
            f"{dataset}: best of {NUM_TRIALS} "
            f"(same T: {temperature['t_start']:.4g} to {temperature['t_end']:.4g})"
        )
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend()

        ax = axes[row, 1]
        x = np.array([point["budget"] for point in points], dtype=float)
        means = np.array([point["paired"]["delta_mean"] for point in points])
        cis = np.array([point["paired"]["delta_mean_ci95"] for point in points])
        yerr = np.vstack([means - cis[:, 0], cis[:, 1] - means])
        colors = ["#147d64" if value >= 0 else "#d62728" for value in means]
        ax.errorbar(
            x,
            means,
            yerr=yerr,
            fmt="none",
            ecolor="#596773",
            elinewidth=1.5,
            capsize=4,
            zorder=1,
        )
        ax.scatter(x, means, c=colors, s=55, zorder=2)
        ax.axhline(0, color="#333333", lw=1, ls="--")
        for xi, yi, point in zip(x, means, points):
            paired = point["paired"]
            ax.annotate(
                f"{paired['wins']}/{NUM_TRIALS} wins",
                (xi, yi),
                xytext=(0, 9 if yi >= 0 else -15),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
        ax.set_xscale("log")
        ax.set_xlabel("SA iterations")
        ax.set_ylabel("mean paired cut gain (warm - cold)")
        ax.set_title(f"{dataset}: paired effect, 95% bootstrap CI")
        ax.grid(True, which="both", ls=":", alpha=0.35)

    fig.suptitle(
        "Matched SA ablation: only the initial spin configuration differs",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "sa_matched_cold_vs_cim.png", bbox_inches="tight")
    plt.close(fig)


def plot_refiner_time_only(results, out_dir):
    """Plot SA quality against SA runtime, excluding CIM seed generation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=180, sharey=False)
    for ax, dataset in zip(axes, DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        bks = data["bks"]
        temperature = data["temperature"]

        cold_x = [point["cold"]["refiner_time"] for point in points]
        warm_x = [point["warm"]["refiner_time"] for point in points]
        cold_y = [max(point["cold"]["gap_best"], 0.25) for point in points]
        warm_y = [max(point["warm"]["gap_best"], 0.25) for point in points]

        ax.plot(
            cold_x,
            cold_y,
            "-o",
            color="#777777",
            lw=2.2,
            ms=6,
            label="cold: random init",
        )
        ax.plot(
            warm_x,
            warm_y,
            "-o",
            color="#d62728",
            lw=2.2,
            ms=6,
            label="warm: CIM init",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("SA wall time only [s] (CIM time excluded)")
        ax.set_ylabel(f"best gap to BKS={bks}")
        ax.set_title(
            f"{dataset}: best of {NUM_TRIALS}\n"
            f"same T: {temperature['t_start']:.4g} to {temperature['t_end']:.4g}"
        )
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend(loc="best")

    fig.suptitle(
        "Matched SA ablation: solver time only (CIM seed generation excluded)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "sa_matched_refiner_time_only.png", bbox_inches="tight")
    plt.close(fig)


def plot_budget_aligned(results, out_dir):
    """Plot independent SA budgets in shared cold/warm x columns."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160)
    for row, dataset in enumerate(DATASETS):
        data = results["datasets"][dataset]
        points = data["points"]
        positions = np.arange(len(points))
        labels = [
            (
                f"{point['budget'] // 1_000_000}m"
                if point["budget"] >= 1_000_000
                else f"{point['budget'] // 1_000}k"
            )
            for point in points
        ]

        ax = axes[row, 0]
        cold = np.array([max(point["cold"]["gap_best"], 0.25) for point in points])
        warm = np.array([max(point["warm"]["gap_best"], 0.25) for point in points])
        for x, yc, yw in zip(positions, cold, warm):
            ax.plot([x, x], [yc, yw], color="#c8d1d8", lw=1.2, zorder=1)
        ax.scatter(
            positions,
            cold,
            s=75,
            facecolors="white",
            edgecolors="#666666",
            linewidths=2,
            label="cold: random init",
            zorder=2,
        )
        ax.scatter(
            positions,
            warm,
            s=75,
            marker="x",
            color="#d62728",
            linewidths=2.4,
            label="warm: CIM init",
            zorder=3,
        )
        ax.set_yscale("log")
        ax.set_xticks(positions, labels)
        ax.set_xlabel("SA iterations (same column = same budget)")
        ax.set_ylabel(f"best gap to BKS={data['bks']}")
        ax.set_title(f"{dataset}: independent best-of-{NUM_TRIALS} at each budget")
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
        ax.legend()

        ax = axes[row, 1]
        means = np.array([point["paired"]["delta_mean"] for point in points])
        cis = np.array([point["paired"]["delta_mean_ci95"] for point in points])
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
        ax.set_xticks(positions, labels)
        ax.set_xlabel("SA iterations")
        ax.set_ylabel("mean paired cut gain (warm - cold)")
        ax.set_title(f"{dataset}: paired effect, 95% bootstrap CI (W/T/L)")
        ax.grid(True, axis="y", ls=":", alpha=0.35)

    fig.suptitle(
        "Matched SA ablation: independent budgets aligned by iteration count",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "sa_matched_budget_aligned.png", bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = (
        ROOT
        / "results"
        / date.today().isoformat()
        / "cim_warmstart"
        / "sa_matched_v2_G22_G55_nt16_to300m"
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
            "rng_alignment": (
                "warm consumes n initialization uniforms so proposal RNG "
                "starts at the same position as cold"
            ),
            "comparison": (
                "temperature, iterations, seeds, and RNG position are matched; "
                "only initial spins differ"
            ),
        },
        "datasets": {},
    }
    raw = {}

    for dataset in DATASETS:
        ctx = load_context(dataset)
        p = PARAMS[dataset]["SA"]
        (cim_result, cim_time) = timed(
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
            (cold_result, cold_time) = timed(
                lambda b=budget: simulate_sa_batch(
                    ctx.n,
                    ctx.edges,
                    ctx.weights,
                    b,
                    NUM_TRIALS,
                    t_start=p["t_start"],
                    t_end=p["t_end"],
                    seeds=seeds,
                )
            )
            (warm_result, warm_time) = timed(
                lambda b=budget: simulate_sa_warm(
                    ctx.n,
                    ctx.edges,
                    ctx.weights,
                    cim_signs,
                    b,
                    t_start=p["t_start"],
                    t_end=p["t_end"],
                    seeds=seeds,
                    align_rng_with_cold=True,
                )
            )
            cold_cuts = ctx.score(cold_result[1])
            warm_cuts = ctx.score(warm_result[1])
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
                f"{dataset} b={budget}: cold={cold_cuts.max():.0f}, "
                f"warm={warm_cuts.max():.0f}, mean_delta={paired['delta_mean']:+.1f}, "
                f"W/T/L={paired['wins']}/{paired['ties']}/{paired['losses']}, "
                f"p={paired['wilcoxon_p']:.4g}"
            )

        results["datasets"][dataset] = {
            "bks": ctx.bks,
            "temperature": {
                "t_start": p["t_start"],
                "t_end": p["t_end"],
            },
            "cim": {
                "time": cim_time,
                **stats(cim_cuts, ctx.bks),
            },
            "points": points,
        }

    with (out_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "cuts.npz", **raw)
    plot_results(results, out_dir)
    plot_refiner_time_only(results, out_dir)
    plot_budget_aligned(results, out_dir)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
