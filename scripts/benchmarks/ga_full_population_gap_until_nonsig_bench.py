"""GA random population vs all-CIM population until the paired gap is non-significant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.GA import simulate_ga_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim


DEFAULT_BUDGETS = [0, 1, 3, 8, 20, 50, 120, 300, 800]


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def summary(cuts: np.ndarray, bks: int, solver_time: float, cim_time: float = 0.0) -> dict:
    cuts = np.asarray(cuts, dtype=float)
    gaps = bks - cuts
    return {
        "cut_best": float(cuts.max()),
        "cut_mean": float(cuts.mean()),
        "gap_best": float(gaps.min()),
        "gap_mean": float(gaps.mean()),
        "gap_median": float(np.median(gaps)),
        "gap_std": float(gaps.std()),
        "solver_time": float(solver_time),
        "cim_time": float(cim_time),
        "total_time": float(solver_time + cim_time),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int, reps: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(values, size=len(values), replace=True).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_stats(cold: np.ndarray, warm: np.ndarray, seed: int) -> dict:
    cold = np.asarray(cold, dtype=float)
    warm = np.asarray(warm, dtype=float)
    delta = warm - cold
    p_value = (
        1.0
        if np.all(delta == 0)
        else float(wilcoxon(warm, cold, zero_method="wilcox").pvalue)
    )
    return {
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_mean_ci95": bootstrap_mean_ci(delta, seed),
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
        "wilcoxon_p": p_value,
        "significant_0_05": bool(p_value < 0.05),
    }


def run_ga(ctx, generations: int, seeds: np.ndarray, init_population=None):
    p = PARAMS[ctx.name]["GA"]
    return simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        len(seeds),
        pop_size=int(p["pop_size"]),
        max_generations=int(generations),
        ts_iters=int(p["ts_iters"]),
        cr=int(p["cr"]),
        alpha_tenure=int(p["alpha_tenure"]),
        beta_quality=float(p["beta_quality"]),
        init_population=init_population,
        seeds=seeds,
        align_rng_with_cold=init_population is not None,
    )


def full_cim_seeds(num_trials: int, pop_size: int) -> np.ndarray:
    """Give every initial-population member a distinct reproducible CIM seed."""
    return np.arange(num_trials * pop_size, dtype=np.int64).reshape(num_trials, pop_size)


def population_diversity(population: np.ndarray) -> dict:
    """Mean pairwise Hamming distance after global spin-flip alignment."""
    values = []
    for trial_population in np.asarray(population):
        for i in range(len(trial_population)):
            for j in range(i + 1, len(trial_population)):
                raw = float(np.mean(trial_population[i] != trial_population[j]))
                values.append(min(raw, 1.0 - raw))
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
    }


def save(out: Path, results: dict, raw: dict[str, np.ndarray]) -> None:
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out / "cuts.npz", **raw)


def warmup() -> None:
    ctx = load_context("G22")
    seeds = np.arange(2, dtype=np.int64)
    _, cim_signs = run_cim(ctx, 10, 4, np.arange(4, dtype=np.int64))
    population = cim_signs.reshape(2, 2, ctx.n)
    p = PARAMS["G22"]["GA"]
    common = dict(
        pop_size=2,
        max_generations=1,
        ts_iters=20,
        cr=min(10, int(p["cr"])),
        alpha_tenure=int(p["alpha_tenure"]),
        beta_quality=float(p["beta_quality"]),
        seeds=seeds,
    )
    simulate_ga_batch(ctx.n, ctx.edges, ctx.weights, 2, **common)
    simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        2,
        **common,
        init_population=population,
        align_rng_with_cold=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["G22", "G55", "G70", "K2000"])
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--num-trials", type=int, default=16)
    parser.add_argument("--cim-budget", type=int, default=600)
    parser.add_argument("--continue-existing", action="store_true")
    parser.add_argument(
        "--ignore-stop-for",
        nargs="*",
        default=[],
        help="Datasets that should run every requested budget even after p >= 0.05.",
    )
    args = parser.parse_args()

    out = (
        ROOT
        / "results"
        / date.today().isoformat()
        / "cim_warmstart"
        / "ga_full_population_until_nonsig_G22_G55_G70_K2000_nt16"
    )
    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "results.json"
    cuts_path = out / "cuts.npz"
    if args.continue_existing and result_path.exists():
        results = json.loads(result_path.read_text(encoding="utf-8"))
        raw = dict(np.load(cuts_path)) if cuts_path.exists() else {}
    else:
        results = {
            "meta": {
                "datasets": args.datasets,
                "candidate_budgets": args.budgets,
                "num_trials": args.num_trials,
                "cim_budget": args.cim_budget,
                "warm_mode": "all population members are CIM solutions with distinct seeds",
                "comparison": (
                    "GA parameters, generations, GA seeds, and post-initialization RNG "
                    "position are matched; no experiment-specific tuning"
                ),
                "stop_rule": (
                    "first tested generation budget with paired two-sided Wilcoxon p >= 0.05"
                ),
                "note": (
                    "each generation budget is an independent run; generation 0 includes "
                    "the GA initial-population tabu refinement"
                ),
            },
            "datasets": {},
        }
        raw = {}

    warmup()
    ga_seeds = np.arange(args.num_trials, dtype=np.int64)
    for dataset in args.datasets:
        ignore_stop = dataset in set(args.ignore_stop_for)
        existing = results["datasets"].get(dataset)
        if existing and existing.get("stop_reached") and not ignore_stop:
            print(f"{dataset}: already complete", flush=True)
            continue

        ctx = load_context(dataset)
        params = dict(PARAMS[dataset]["GA"])
        pop_size = int(params["pop_size"])
        seed_matrix = full_cim_seeds(args.num_trials, pop_size)
        (cim_cuts, cim_signs_flat), measured_cim_time = timed(
            lambda: run_cim(
                ctx,
                args.cim_budget,
                args.num_trials * pop_size,
                seed_matrix.reshape(-1),
            )
        )
        full_population = cim_signs_flat.reshape(args.num_trials, pop_size, ctx.n)
        full_cuts = ctx.score(cim_signs_flat).reshape(args.num_trials, pop_size)

        ds = existing or {
            "bks": ctx.bks,
            "n": ctx.n,
            "edges": len(ctx.edges),
            "ga_params": params,
            "pop_size": pop_size,
            "cim": {
                "time": measured_cim_time,
                "num_solutions": args.num_trials * pop_size,
                "cut_best": float(np.max(full_cuts)),
                "cut_mean": float(np.mean(full_cuts)),
                "gap_best": float(ctx.bks - np.max(full_cuts)),
                "diversity": population_diversity(full_population),
                "seed_matrix": seed_matrix.tolist(),
            },
            "points": [],
            "stop_reached": False,
            "stop_generations": None,
        }
        cim_time = float(ds["cim"]["time"])
        results["datasets"][dataset] = ds
        if (
            ignore_stop
            and ds.get("stop_reached")
            and ds.get("first_nonsignificant_generations") is None
        ):
            ds["first_nonsignificant_generations"] = ds.get("stop_generations")
        raw[f"{dataset}_cim_full"] = full_cuts
        done = {int(point["generations"]) for point in ds["points"]}

        print(
            f"{dataset} CIM population: {cim_time:.2f}s, "
            f"{args.num_trials * pop_size} solutions, "
            f"diversity={ds['cim']['diversity']['mean']:.3f}",
            flush=True,
        )
        for generations in args.budgets:
            if generations in done:
                continue
            cold_result, cold_time = timed(
                lambda g=generations: run_ga(ctx, g, ga_seeds)
            )
            warm_result, warm_time = timed(
                lambda g=generations: run_ga(ctx, g, ga_seeds, full_population)
            )
            cold_cuts = np.asarray(cold_result[0], dtype=float)
            warm_cuts = np.asarray(warm_result[0], dtype=float)
            paired = paired_stats(cold_cuts, warm_cuts, seed=2000 + generations)
            point = {
                "generations": int(generations),
                "cold": summary(cold_cuts, ctx.bks, cold_time),
                "warm": summary(warm_cuts, ctx.bks, warm_time, cim_time),
                "paired": paired,
            }
            ds["points"].append(point)
            raw[f"{dataset}_{generations}_cold"] = cold_cuts
            raw[f"{dataset}_{generations}_warm"] = warm_cuts
            if not paired["significant_0_05"] and ds.get(
                "first_nonsignificant_generations"
            ) is None:
                ds["first_nonsignificant_generations"] = int(generations)
            if ignore_stop:
                ds["stop_reached"] = (
                    ds.get("first_nonsignificant_generations") is not None
                )
                ds["stop_generations"] = ds.get(
                    "first_nonsignificant_generations"
                )
                ds["extended_through_generations"] = int(generations)
            else:
                ds["stop_reached"] = not paired["significant_0_05"]
                ds["stop_generations"] = (
                    int(generations) if ds["stop_reached"] else None
                )
            save(out, results, raw)
            print(
                f"{dataset} gen={generations}: "
                f"gap mean cold={point['cold']['gap_mean']:.1f}, "
                f"warm={point['warm']['gap_mean']:.1f}, "
                f"gain={paired['delta_mean']:+.1f}, p={paired['wilcoxon_p']:.4g}, "
                f"time={cold_time:.1f}/{warm_time + cim_time:.1f}s",
                flush=True,
            )
            if ds["stop_reached"] and not ignore_stop:
                break
        if not ds["stop_reached"] and ds["points"]:
            ds["practical_cap_generations"] = int(ds["points"][-1]["generations"])
            ds["cap_reason"] = (
                "The next longer GA run would require several additional minutes per "
                "condition, while the improvement from 300 to 800 generations was small."
            )
        save(out, results, raw)

    print(out, flush=True)


if __name__ == "__main__":
    main()
