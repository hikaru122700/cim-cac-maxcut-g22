"""PT-ICM cold vs CIM-ladder warm start until the paired difference is non-significant."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.PT_ICM import simulate_pticm_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim

DEFAULT_BUDGETS = [10, 30, 80, 200, 600, 1800, 5000]


def timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def pt(ctx, sweeps, seeds, init_signs=None):
    p = PARAMS[ctx.name]["PT"]
    return simulate_pticm_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        len(seeds),
        num_sweeps=int(sweeps),
        num_temps=p["num_temps"],
        t_min=p["t_min"],
        t_max=p["t_max"],
        swap_interval=p["swap_interval"],
        icm_interval=p["icm_interval"],
        seeds=seeds,
        init_signs=init_signs,
        init_mode="ladder",
        perturb_max=0.5,
    )


def bootstrap_mean_ci(values: np.ndarray, seed: int, reps: int = 4000) -> list[float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = rng.choice(values, size=values.size, replace=True).mean()
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def paired_stats(cold: np.ndarray, warm: np.ndarray, seed: int) -> dict:
    delta = np.asarray(warm, dtype=float) - np.asarray(cold, dtype=float)
    if np.all(delta == 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(delta, zero_method="wilcox", alternative="two-sided").pvalue)
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


def save(out: Path, results: dict, raw: dict[str, np.ndarray]) -> None:
    (out / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out / "cuts.npz", **raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["G22", "G55", "G70", "K2000"])
    parser.add_argument("--num-trials", type=int, default=16)
    parser.add_argument("--cim-budget", type=int, default=600)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--continue-existing", action="store_true")
    args = parser.parse_args()

    out = args.output or (
        ROOT
        / "results"
        / date.today().isoformat()
        / "cim_warmstart"
        / f"pticm_gap_until_nonsig_G22_G55_G70_K2000_nt{args.num_trials}"
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
                "warm_mode": "ladder",
                "perturb_max": 0.5,
                "stop_rule": "first tested sweep budget with paired two-sided Wilcoxon p >= 0.05",
                "note": "each sweep budget is an independent run; CIM generation time is measured once per dataset",
            },
            "datasets": {},
        }
        raw = {}

    # Compile before timing.
    warm_ctx = load_context("G22")
    warm_seeds = np.arange(2, dtype=np.int64)
    _, warm_init = run_cim(warm_ctx, 10, 2, warm_seeds)
    pt(warm_ctx, 1, warm_seeds)
    pt(warm_ctx, 1, warm_seeds, warm_init)

    seeds = np.arange(args.num_trials, dtype=np.int64)
    for dataset in args.datasets:
        existing = results["datasets"].get(dataset)
        if existing and existing.get("stop_reached"):
            print(f"{dataset}: already complete", flush=True)
            continue
        ctx = load_context(dataset)
        (cim_cuts, cim_signs), cim_time = timed(
            lambda: run_cim(ctx, args.cim_budget, args.num_trials, seeds)
        )
        ds = existing or {
            "bks": ctx.bks,
            "n": ctx.n,
            "edges": len(ctx.edges),
            "pt_params": PARAMS[dataset]["PT"],
            "cim": {
                "time": cim_time,
                "cut_best": float(np.max(cim_cuts)),
                "cut_mean": float(np.mean(cim_cuts)),
                "gap_best": float(ctx.bks - np.max(cim_cuts)),
            },
            "points": [],
            "stop_reached": False,
        }
        if existing:
            cim_time = float(ds["cim"]["time"])
        results["datasets"][dataset] = ds
        done = {int(point["sweeps"]) for point in ds["points"]}

        for sweeps in args.budgets:
            if sweeps in done:
                continue
            cold_result, cold_time = timed(lambda: pt(ctx, sweeps, seeds))
            warm_result, warm_time = timed(lambda: pt(ctx, sweeps, seeds, cim_signs))
            cold_cuts = np.asarray(cold_result[0], dtype=float)
            warm_cuts = np.asarray(warm_result[0], dtype=float)
            stats = paired_stats(cold_cuts, warm_cuts, seed=1000 + sweeps)
            point = {
                "sweeps": int(sweeps),
                "cold": summary(cold_cuts, ctx.bks, cold_time),
                "warm": summary(warm_cuts, ctx.bks, warm_time, cim_time),
                "paired": stats,
            }
            ds["points"].append(point)
            raw[f"{dataset}_{sweeps}_cold"] = cold_cuts
            raw[f"{dataset}_{sweeps}_warm"] = warm_cuts
            ds["stop_reached"] = not stats["significant_0_05"]
            ds["stop_sweeps"] = int(sweeps) if ds["stop_reached"] else None
            save(out, results, raw)
            print(
                f"{dataset} sweeps={sweeps}: "
                f"gap mean cold={point['cold']['gap_mean']:.1f}, "
                f"warm={point['warm']['gap_mean']:.1f}, "
                f"gain={stats['delta_mean']:+.1f}, p={stats['wilcoxon_p']:.4g}, "
                f"time={cold_time:.1f}/{warm_time + cim_time:.1f}s",
                flush=True,
            )
            if ds["stop_reached"]:
                break
        save(out, results, raw)

    print(out, flush=True)


if __name__ == "__main__":
    main()
