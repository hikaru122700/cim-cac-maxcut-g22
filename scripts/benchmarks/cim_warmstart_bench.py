"""End-to-end CIM warm-start benchmark for GA, SA, and PT-ICM.

The benchmark compares random (cold) initialization with a shared batch of
CIM solutions.  Warm timings always include the measured CIM generation cost.

Examples
--------
Smoke test:
    uv run python scripts/benchmarks/cim_warmstart_bench.py --smoke

G22 experiment:
    uv run python scripts/benchmarks/cim_warmstart_bench.py \
        --dataset G22 --num-trials 16 --tag main
"""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.GA import simulate_ga_batch
from modules.PT_ICM import simulate_pticm_batch
from modules.SA import simulate_sa_batch, simulate_sa_warm
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim


DEFAULT_BUDGETS = {
    "GA": [3, 8, 20],
    "SA": [1_000_000, 3_000_000, 10_000_000],
    "PT": [30, 80, 200],
}


def _out_dir(dataset: str, num_trials: int, tag: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / "cim_warmstart"
    root.mkdir(parents=True, exist_ok=True)
    versions = [
        int(p.name.split("_", 1)[0][1:])
        for p in root.iterdir()
        if p.is_dir()
        and p.name.startswith("v")
        and p.name.split("_", 1)[0][1:].isdigit()
    ]
    version = max(versions, default=0) + 1
    suffix = f"_{tag}" if tag else ""
    out = root / f"v{version}_{dataset}_nt{num_trials}{suffix}"
    out.mkdir()
    return out


def _stats(cuts: np.ndarray, bks: int) -> dict:
    cuts = np.asarray(cuts, dtype=np.float64)
    return {
        "cut_max": float(cuts.max()),
        "cut_mean": float(cuts.mean()),
        "cut_median": float(np.median(cuts)),
        "cut_std": float(cuts.std()),
        "gap_best": float(bks - cuts.max()),
        "success_bks": int(np.sum(cuts >= bks)),
    }


def _comparison(cold: np.ndarray, warm: np.ndarray) -> dict:
    delta = np.asarray(warm) - np.asarray(cold)
    return {
        "warm_minus_cold_mean": float(delta.mean()),
        "warm_minus_cold_median": float(np.median(delta)),
        "wins": int(np.sum(delta > 0)),
        "ties": int(np.sum(delta == 0)),
        "losses": int(np.sum(delta < 0)),
    }


def _timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def _run_ga(ctx, budget, num_trials, seeds, init_signs=None):
    p = PARAMS[ctx.name]["GA"]
    return simulate_ga_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        num_trials,
        pop_size=p["pop_size"],
        max_generations=int(budget),
        ts_iters=p["ts_iters"],
        cr=p["cr"],
        alpha_tenure=p["alpha_tenure"],
        beta_quality=p["beta_quality"],
        init_signs=init_signs,
        seeds=seeds,
    )


def _run_sa(ctx, budget, num_trials, seeds, init_signs=None):
    if init_signs is None:
        p = PARAMS[ctx.name]["SA"]
        return simulate_sa_batch(
            ctx.n,
            ctx.edges,
            ctx.weights,
            int(budget),
            num_trials,
            t_start=p["t_start"],
            t_end=p["t_end"],
            seeds=seeds,
        )
    return simulate_sa_warm(
        ctx.n,
        ctx.edges,
        ctx.weights,
        init_signs,
        int(budget),
        t_start=0.5,
        t_end=0.001,
        seeds=seeds,
    )


def _run_pt(
    ctx,
    budget,
    num_trials,
    seeds,
    init_signs=None,
    init_mode="single",
    perturb_max=0.5,
):
    p = PARAMS[ctx.name]["PT"]
    return simulate_pticm_batch(
        ctx.n,
        ctx.edges,
        ctx.weights,
        num_trials,
        num_sweeps=int(budget),
        num_temps=p["num_temps"],
        t_min=p["t_min"],
        t_max=p["t_max"],
        swap_interval=p["swap_interval"],
        icm_interval=p["icm_interval"],
        seeds=seeds,
        init_signs=init_signs,
        init_mode=init_mode,
        perturb_max=perturb_max,
    )


def _warm_up():
    """Compile all Numba paths before measurements."""
    ctx = load_context("G22")
    seeds = np.arange(2, dtype=np.int64)
    _, init = run_cim(ctx, 20, 2, seeds)
    _run_ga(ctx, 1, 2, seeds)
    _run_ga(ctx, 1, 2, seeds, init)
    _run_sa(ctx, 20, 2, seeds)
    _run_sa(ctx, 20, 2, seeds, init)
    _run_pt(ctx, 1, 2, seeds)
    _run_pt(ctx, 1, 2, seeds, init, "single")
    _run_pt(ctx, 1, 2, seeds, init, "ladder", 0.5)


def _record_pair(
    *,
    bks,
    budget,
    cold_result,
    cold_time,
    warm_result,
    warm_time,
    cim_time,
):
    cold_cuts = np.asarray(cold_result[0])
    warm_cuts = np.asarray(warm_result[0])
    return {
        "budget": int(budget),
        "cold": {
            "refiner_time": cold_time,
            "total_time": cold_time,
            **_stats(cold_cuts, bks),
        },
        "warm": {
            "refiner_time": warm_time,
            "cim_time": cim_time,
            "total_time": cim_time + warm_time,
            **_stats(warm_cuts, bks),
        },
        "paired": _comparison(cold_cuts, warm_cuts),
        "_cold_cuts": cold_cuts,
        "_warm_cuts": warm_cuts,
    }


def _json_ready(record):
    return {k: v for k, v in record.items() if not k.startswith("_")}


def _plot(results: dict, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "cold": "#777777",
        "warm": "#d62728",
        "single": "#9467bd",
        "ladder": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=140)
    bks = results["meta"]["bks"]
    for ax, solver in zip(axes, ("GA", "SA", "PT")):
        series = results["solvers"][solver]
        if solver != "PT":
            variants = [("cold", "cold"), ("warm", "CIM warm")]
            for key, label in variants:
                xs = [point[key]["total_time"] for point in series]
                ys = [max(point[key]["gap_best"], 0.25) for point in series]
                ax.plot(xs, ys, "-o", color=colors[key], label=label)
        else:
            xs = [point["cold"]["total_time"] for point in series["single"]]
            ys = [
                max(point["cold"]["gap_best"], 0.25)
                for point in series["single"]
            ]
            ax.plot(xs, ys, "-o", color=colors["cold"], label="cold")
            for mode in ("single", "ladder"):
                points = series[mode]
                xs = [point["warm"]["total_time"] for point in points]
                ys = [max(point["warm"]["gap_best"], 0.25) for point in points]
                ax.plot(
                    xs,
                    ys,
                    "-o",
                    color=colors[mode],
                    label=f"CIM {mode}",
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("end-to-end wall time [s]")
        ax.set_ylabel(f"gap to BKS={bks}")
        ax.set_title(solver if solver != "PT" else "PT-ICM")
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend()
    fig.suptitle(
        f"{results['meta']['dataset']}: CIM warm start (best of "
        f"{results['meta']['num_trials']})"
    )
    fig.tight_layout()
    fig.savefig(out / "gap_vs_total_time.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="G22")
    parser.add_argument("--num-trials", type=int, default=16)
    parser.add_argument("--cim-budget", type=int, default=600)
    parser.add_argument("--pt-perturb-max", type=float, default=0.5)
    parser.add_argument("--tag", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    budgets = (
        {"GA": [1], "SA": [10_000], "PT": [2]}
        if args.smoke
        else DEFAULT_BUDGETS
    )
    if args.smoke and args.num_trials == 16:
        args.num_trials = 2
    if args.smoke and args.cim_budget == 600:
        args.cim_budget = 30

    out = _out_dir(args.dataset, args.num_trials, args.tag or ("smoke" if args.smoke else "main"))

    def log(message):
        print(message, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    log("JIT warm-up")
    _warm_up()

    ctx = load_context(args.dataset)
    seeds = np.arange(args.num_trials, dtype=np.int64)
    (cim_result, cim_time) = _timed(
        lambda: run_cim(ctx, args.cim_budget, args.num_trials, seeds)
    )
    cim_cuts, cim_signs = cim_result
    cim_cuts = ctx.score(cim_signs)
    log(
        f"CIM budget={args.cim_budget} time={cim_time:.3f}s "
        f"best={cim_cuts.max():.0f} mean={cim_cuts.mean():.1f}"
    )

    results = {
        "meta": {
            "dataset": args.dataset,
            "n": ctx.n,
            "bks": ctx.bks,
            "num_trials": args.num_trials,
            "numba_threads": int(os.environ["NUMBA_NUM_THREADS"]),
            "cim_budget": args.cim_budget,
            "cim_time": cim_time,
            "cim": _stats(cim_cuts, ctx.bks),
            "timing_definition": "warm total_time = measured CIM time + refiner time",
            "pt_ladder_perturb_max": args.pt_perturb_max,
        },
        "solvers": {"GA": [], "SA": [], "PT": {"single": [], "ladder": []}},
    }
    raw = {"cim": np.asarray(cim_cuts)}

    for solver, runner in (("GA", _run_ga), ("SA", _run_sa)):
        for budget in budgets[solver]:
            cold_result, cold_time = _timed(
                lambda b=budget: runner(ctx, b, args.num_trials, seeds)
            )
            warm_result, warm_time = _timed(
                lambda b=budget: runner(
                    ctx, b, args.num_trials, seeds, cim_signs
                )
            )
            record = _record_pair(
                bks=ctx.bks,
                budget=budget,
                cold_result=cold_result,
                cold_time=cold_time,
                warm_result=warm_result,
                warm_time=warm_time,
                cim_time=cim_time,
            )
            results["solvers"][solver].append(_json_ready(record))
            raw[f"{solver}_{budget}_cold"] = record["_cold_cuts"]
            raw[f"{solver}_{budget}_warm"] = record["_warm_cuts"]
            log(
                f"{solver} b={budget}: cold={record['cold']['cut_max']:.0f} "
                f"({cold_time:.3f}s), warm={record['warm']['cut_max']:.0f} "
                f"(total {record['warm']['total_time']:.3f}s), "
                f"delta_mean={record['paired']['warm_minus_cold_mean']:+.1f}"
            )

    for budget in budgets["PT"]:
        cold_result, cold_time = _timed(
            lambda b=budget: _run_pt(ctx, b, args.num_trials, seeds)
        )
        for mode in ("single", "ladder"):
            warm_result, warm_time = _timed(
                lambda b=budget, m=mode: _run_pt(
                    ctx,
                    b,
                    args.num_trials,
                    seeds,
                    cim_signs,
                    m,
                    args.pt_perturb_max,
                )
            )
            record = _record_pair(
                bks=ctx.bks,
                budget=budget,
                cold_result=cold_result,
                cold_time=cold_time,
                warm_result=warm_result,
                warm_time=warm_time,
                cim_time=cim_time,
            )
            results["solvers"]["PT"][mode].append(_json_ready(record))
            raw[f"PT_{budget}_cold"] = record["_cold_cuts"]
            raw[f"PT_{budget}_{mode}"] = record["_warm_cuts"]
            log(
                f"PT-{mode} b={budget}: cold={record['cold']['cut_max']:.0f} "
                f"({cold_time:.3f}s), warm={record['warm']['cut_max']:.0f} "
                f"(total {record['warm']['total_time']:.3f}s), "
                f"delta_mean={record['paired']['warm_minus_cold_mean']:+.1f}"
            )

    with (out / "results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    np.savez_compressed(out / "cuts.npz", **raw)
    _plot(results, out)
    log(f"saved: {out}")
    print(f"OUT_DIR={out}")


if __name__ == "__main__":
    main()
