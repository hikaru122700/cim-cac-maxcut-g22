"""Compare random, one-CIM, and all-CIM initial GA populations."""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.GA import simulate_ga_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim
from scripts.benchmarks.ga_matched_warmstart_bench import paired_stats, stats, timed


DATASETS = ["G22", "G55"]
BUDGETS = [0, 1, 3, 8, 20, 50]
NUM_TRIALS = 16
CIM_BUDGET = 600


def run_ga(ctx, budget, seeds, *, init_signs=None, init_population=None):
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
        init_population=init_population,
        seeds=seeds,
        align_rng_with_cold=(
            init_signs is not None or init_population is not None
        ),
    )


def full_cim_seed_matrix(pop_size):
    """Member 0 uses trial seed t; all other members use unique later seeds."""
    seeds = np.empty((NUM_TRIALS, pop_size), dtype=np.int64)
    seeds[:, 0] = np.arange(NUM_TRIALS, dtype=np.int64)
    if pop_size > 1:
        rest = np.arange(
            NUM_TRIALS,
            NUM_TRIALS + NUM_TRIALS * (pop_size - 1),
            dtype=np.int64,
        )
        seeds[:, 1:] = rest.reshape(NUM_TRIALS, pop_size - 1)
    return seeds


def aligned_population_diversity(population):
    """Mean normalized Hamming distance with global spin-flip alignment."""
    population = np.asarray(population)
    trials, pop_size, n = population.shape
    values = []
    for trial in range(trials):
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                raw = np.mean(population[trial, i] != population[trial, j])
                values.append(min(raw, 1.0 - raw))
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values)) if values else 0.0,
        "min": float(np.min(values)) if values else 0.0,
        "max": float(np.max(values)) if values else 0.0,
    }


def warmup():
    ctx = load_context("G22")
    seeds = np.arange(2, dtype=np.int64)
    _, cim = run_cim(ctx, 20, 4, np.arange(4, dtype=np.int64))
    population = cim.reshape(2, 2, ctx.n)
    p = PARAMS["G22"]["GA"]
    common = dict(
        pop_size=2,
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
        init_signs=population[:, 0],
        align_rng_with_cold=True,
    )
    simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        2,
        **common,
        init_population=population,
        align_rng_with_cold=True,
    )


def main():
    out_dir = (
        ROOT
        / "results"
        / date.today().isoformat()
        / "cim_warmstart"
        / "ga_full_cim_population_v1_G22_G55_nt16"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    warmup()
    ga_seeds = np.arange(NUM_TRIALS, dtype=np.int64)
    results = {
        "meta": {
            "datasets": DATASETS,
            "budgets": BUDGETS,
            "num_trials": NUM_TRIALS,
            "cim_budget": CIM_BUDGET,
            "variants": {
                "cold": "all population members are random",
                "single": "member 0 is CIM; remaining members are random",
                "full": "all population members are CIM with distinct CIM seeds",
            },
            "comparison": (
                "GA/TS parameters, generations, GA seeds, and post-initialization "
                "RNG position are matched"
            ),
        },
        "datasets": {},
    }
    raw = {}

    for dataset in DATASETS:
        ctx = load_context(dataset)
        p = PARAMS[dataset]["GA"]
        pop_size = int(p["pop_size"])

        single_cim_result, single_cim_time = timed(
            lambda: run_cim(
                ctx,
                CIM_BUDGET,
                NUM_TRIALS,
                np.arange(NUM_TRIALS, dtype=np.int64),
            )
        )
        _, single_signs = single_cim_result
        single_cuts = ctx.score(single_signs)

        cim_seed_matrix = full_cim_seed_matrix(pop_size)
        full_cim_result, full_cim_time = timed(
            lambda: run_cim(
                ctx,
                CIM_BUDGET,
                NUM_TRIALS * pop_size,
                cim_seed_matrix.reshape(-1),
            )
        )
        _, full_signs_flat = full_cim_result
        full_population = full_signs_flat.reshape(
            NUM_TRIALS, pop_size, ctx.n
        )
        full_cuts = ctx.score(full_signs_flat).reshape(NUM_TRIALS, pop_size)
        diversity = aligned_population_diversity(full_population)
        log(
            f"{dataset} CIM: single={single_cim_time:.3f}s/{NUM_TRIALS}, "
            f"full={full_cim_time:.3f}s/{NUM_TRIALS * pop_size}, "
            f"full_diversity={diversity['mean']:.3f}"
        )

        points = []
        raw[f"{dataset}_cim_single"] = single_cuts
        raw[f"{dataset}_cim_full"] = full_cuts

        for budget in BUDGETS:
            cold_result, cold_time = timed(
                lambda b=budget: run_ga(ctx, b, ga_seeds)
            )
            single_result, single_time = timed(
                lambda b=budget: run_ga(
                    ctx, b, ga_seeds, init_signs=single_signs
                )
            )
            full_result, full_time = timed(
                lambda b=budget: run_ga(
                    ctx, b, ga_seeds, init_population=full_population
                )
            )
            cold_cuts = np.asarray(cold_result[0])
            one_cuts = np.asarray(single_result[0])
            all_cuts = np.asarray(full_result[0])
            point = {
                "budget": budget,
                "cold": {
                    "refiner_time": cold_time,
                    "total_time": cold_time,
                    **stats(cold_cuts, ctx.bks),
                },
                "single": {
                    "refiner_time": single_time,
                    "cim_time": single_cim_time,
                    "total_time": single_cim_time + single_time,
                    **stats(one_cuts, ctx.bks),
                },
                "full": {
                    "refiner_time": full_time,
                    "cim_time": full_cim_time,
                    "total_time": full_cim_time + full_time,
                    **stats(all_cuts, ctx.bks),
                },
                "paired": {
                    "single_vs_cold": paired_stats(cold_cuts, one_cuts),
                    "full_vs_cold": paired_stats(cold_cuts, all_cuts),
                    "full_vs_single": paired_stats(one_cuts, all_cuts),
                },
            }
            points.append(point)
            raw[f"{dataset}_{budget}_cold"] = cold_cuts
            raw[f"{dataset}_{budget}_single"] = one_cuts
            raw[f"{dataset}_{budget}_full"] = all_cuts
            full_pair = point["paired"]["full_vs_cold"]
            fs_pair = point["paired"]["full_vs_single"]
            log(
                f"{dataset} g={budget}: cold={cold_cuts.max():.0f}, "
                f"single={one_cuts.max():.0f}, full={all_cuts.max():.0f}; "
                f"full-cold={full_pair['delta_mean']:+.1f} "
                f"({full_pair['wins']}/{full_pair['ties']}/{full_pair['losses']}), "
                f"full-single={fs_pair['delta_mean']:+.1f}"
            )

        results["datasets"][dataset] = {
            "bks": ctx.bks,
            "ga_params": p,
            "pop_size": pop_size,
            "cim_single": {
                "num_solutions": NUM_TRIALS,
                "time": single_cim_time,
                **stats(single_cuts, ctx.bks),
            },
            "cim_full": {
                "num_solutions": NUM_TRIALS * pop_size,
                "time": full_cim_time,
                "cut_max": float(full_cuts.max()),
                "cut_mean": float(full_cuts.mean()),
                "cut_std": float(full_cuts.std()),
                "diversity": diversity,
                "seed_matrix": cim_seed_matrix.tolist(),
            },
            "points": points,
        }

    with (out_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(out_dir / "cuts.npz", **raw)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
