"""Measure within-population diversity during the existing memetic GA.

Run from the repository root. Each run gets its own fresh CIM population;
no selection by quality or distance is performed. Generation = one offspring.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from numba import njit
from modules.GA import simulate_ga_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim

EXPERIMENT_KIND = "ga_diversity_trajectory"
CIM_ROUNDS = {"G22": 4800, "G55": 9600, "G70": 4800}
CHECKPOINTS = {-1, 0, 1, 3, 8, 20, 50, 120, 300, 800}
LABELS = {"random": "ランダム初期集団", "cim": "全CIM初期集団"}
COLORS = {"random": "#506778", "cim": "#C34436"}


@njit(cache=True)
def population_metrics(populations):
    """Return mean distance, mean nearest distance, min distance, unique count.

    Compute within each run, never across runs. Global complements are equal.
    """
    runs, size, n = populations.shape
    result = np.zeros((runs, 4))
    for t in range(runs):
        nearest = np.full(size, 0.5)
        duplicates = np.zeros(size, dtype=np.bool_)
        total = 0.0
        smallest = 0.5
        for i in range(size):
            for j in range(i + 1, size):
                h = 0
                for v in range(n):
                    h += populations[t, i, v] != populations[t, j, v]
                d = min(h, n - h) / n
                total += d
                nearest[i] = min(nearest[i], d)
                nearest[j] = min(nearest[j], d)
                smallest = min(smallest, d)
                if d == 0:
                    duplicates[j] = True
        result[t, 0] = total / (size * (size - 1) / 2)
        result[t, 1] = nearest.mean()
        result[t, 2] = smallest
        result[t, 3] = size - duplicates.sum()
    return result


def create_output(args):
    kind = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name.split("_")[0][1:]) for p in kind.iterdir()
                if p.is_dir() and p.name.split("_")[0][1:].isdigit()]
    desc = f"{'_'.join(args.datasets)}_nt{args.num_trials}_gen{args.generations}"
    if args.tag:
        desc += f"_{args.tag}"
    out = kind / f"v{max(versions, default=0) + 1}_{desc}"
    out.mkdir(exist_ok=False)
    return out


def plot_results(out, datasets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import t as student_t

    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False,
                         "font.size": 11, "xtick.direction": "in",
                         "ytick.direction": "in", "xtick.top": True, "ytick.right": True})
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(3, len(datasets), figsize=(6 * len(datasets), 11), squeeze=False)
    init_fig, init_axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 4.7), squeeze=False)
    detail_fig, detail_axes = plt.subplots(2, len(datasets), figsize=(6 * len(datasets), 7.6), squeeze=False)
    summaries = {}
    for col, ds in enumerate(datasets):
        summaries[ds] = {}
        bks = meta["datasets"][ds]["bks"]
        for condition, label in LABELS.items():
            with np.load(out / f"{ds}_{condition}_trajectory.npz") as z:
                g = z["generation"]
                keep = g >= 0
                metrics = z["metrics"]
                gaps = bks - z["best_ever"]
                fit_mean = z["population_fits"].mean(axis=2)
                summaries[ds][condition] = {
                    "raw_D": float(metrics[0, :, 0].mean()),
                    "gen0_D": float(metrics[1, :, 0].mean()),
                    "final_D": float(metrics[-1, :, 0].mean()),
                    "final_min_D_across_runs": float(metrics[-1, :, 0].min()),
                    "gen0_gap": float(gaps[1].mean()),
                    "final_gap": float(gaps[-1].mean()),
                    "gen0_unique": float(metrics[1, :, 3].mean()),
                    "final_unique": float(metrics[-1, :, 3].mean()),
                    "D_decreased_runs": int((metrics[-1, :, 0] < metrics[1, :, 0]).sum()),
                    "num_trials": int(metrics.shape[1]),
                    "generations": int(g[-1]),
                    "checkpoints": {str(int(g[k])): {
                        "D": float(metrics[k, :, 0].mean()),
                        "gap": float(gaps[k].mean())}
                        for k in range(len(g)) if int(g[k]) in CHECKPOINTS},
                }
                for row, values in enumerate([gaps, metrics[:, :, 0], metrics[:, :, 3]]):
                    y = values[keep]
                    mean = y.mean(axis=1)
                    band = student_t.ppf(0.975, y.shape[1] - 1) * y.std(axis=1, ddof=1) / np.sqrt(y.shape[1])
                    ax = axes[row, col]
                    ax.plot(g[keep], mean, color=COLORS[condition], label=label, lw=2)
                    ax.fill_between(g[keep], mean - band, mean + band, color=COLORS[condition], alpha=.14)
                ax = init_axes[0, col]
                for trial in range(metrics.shape[1]):
                    ax.plot([0, 1], metrics[:2, trial, 0], color=COLORS[condition], alpha=.16, lw=.8)
                ax.plot([0, 1], metrics[:2, :, 0].mean(axis=1), "o-", color=COLORS[condition], label=label, lw=2)
                for row in range(2):
                    vals = metrics[keep, :, 0] if row == 0 else gaps[keep]
                    detail_axes[row, col].plot(g[keep], vals, color=COLORS[condition], alpha=.28, lw=.8)
                    detail_axes[row, col].plot(g[keep], vals.mean(axis=1), color=COLORS[condition], lw=2, label=label)
        for row in range(3):
            ax = axes[row, col]
            ax.set_xscale("symlog", linthresh=1)
            ticks = [i for i in [0, 1, 3, 8, 20, 50, 120, 300, 800] if i <= meta["generations"]]
            ax.set_xticks(ticks, labels=[str(i) for i in ticks])
            ax.grid(alpha=.18)
            ax.set_xlabel("世代（1世代で子を1個生成）")
        axes[0, col].set_title(f"{ds}  集団サイズ{meta['datasets'][ds]['ga_params']['pop_size']}")
        axes[0, col].set_ylabel("各runの最良値とBKSの差\n小さいほど良い")
        axes[1, col].set_ylabel("集団内の平均ペア距離 D\n大きいほど多様")
        axes[1, col].set_ylim(0, .5)
        axes[2, col].set_ylabel("異なる分割の個数\n全反転は同一と数える")
        axes[2, col].set_ylim(0, meta["datasets"][ds]["ga_params"]["pop_size"] + 1)
        axes[0, col].legend(fontsize=10)
        init_axes[0, col].set(title=ds, xticks=[0, 1], xticklabels=["初期TS前", "初期TS後（世代0）"],
                                 ylabel="集団内の平均ペア距離 D", ylim=(0, .5))
        init_axes[0, col].legend(fontsize=10)
        init_axes[0, col].grid(alpha=.18)
        for row in range(2):
            detail_axes[row, col].set_title(ds)
            detail_axes[row, col].set_xlabel("世代")
            detail_axes[row, col].grid(alpha=.18)
        detail_axes[0, col].set_ylabel("集団内の平均ペア距離 D")
        detail_axes[0, col].set_ylim(0, .5)
        detail_axes[1, col].set_ylabel("各runの最良値とBKSの差")
    fig.suptitle("GAの改善と、集団内の多様性を同じ世代で追跡", fontsize=19)
    fig.text(.5, .012, "線：独立runの平均　帯：run単位の平均の95% t区間（点ごと）　世代0：初期集団のTS後\n"
             "CIMとランダムの初期品質はそろえていない。この比較だけで多様性の因果効果は判断しない。",
             ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .06, 1, .96))
    init_fig.suptitle("初期の局所探索で多様性はどう変わるか（細線：各run、太線：平均）", fontsize=16)
    init_fig.tight_layout(rect=(0, 0, 1, .93))
    detail_fig.suptitle("各runの推移（細線）と平均（太線）", fontsize=17)
    detail_fig.tight_layout(rect=(0, 0, 1, .95))
    for figure, name in [(fig, "trajectory"), (init_fig, "initial_refinement"), (detail_fig, "individual_runs")]:
        figure.savefig(out / f"{name}.png", dpi=160)
        figure.savefig(out / f"{name}.svg")
        plt.close(figure)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", choices=list(CIM_ROUNDS), default=list(CIM_ROUNDS))
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--generations", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=10000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.num_trials < 2 or args.generations < 1:
        ap.error("at least two trials and one generation are required")
    if args.tag and not all(c.isascii() and (c.isalnum() or c == "_") for c in args.tag):
        ap.error("tag must use ASCII letters, digits or underscores")
    out = create_output(args)

    def log(message):
        print(message, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(message + "\n")

    meta = {"datasets": {}, "num_trials": args.num_trials, "generations": args.generations,
            "seed_offset": args.seed_offset, "threads": os.environ["NUMBA_NUM_THREADS"],
            "definition": "within-run mean min(H,N-H)/N; generation=-1 before initial TS, 0 after TS",
            "snapshot_bitorder": "little", "metric_columns": ["mean_pairwise", "mean_nearest", "min_pairwise", "unique_partitions"],
            "initial_quality_matched": False, "screenshot_reproduction": False}
    source_files = [Path(__file__), ROOT / "modules/GA.py", ROOT / "modules/CIM.py",
                    ROOT / "scripts/benchmarks/algo_registry.py", ROOT / "results/anytime_tuned_params.json"]
    meta["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    log(f"OUTPUT={out}")
    for ds in args.datasets:
        ctx = load_context(ds)
        gp = PARAMS[ds]["GA"].copy()
        size = gp["pop_size"]
        ga_seeds = np.arange(args.num_trials, dtype=np.int64) + args.seed_offset
        cim_seeds = np.arange(args.num_trials * size, dtype=np.int64) + args.seed_offset + 1000000
        meta["sha256"][f"input/{ds}.txt"] = hashlib.sha256((ROOT / f"input/{ds}.txt").read_bytes()).hexdigest()
        meta["datasets"][ds] = {"n": ctx.n, "bks": ctx.bks, "ga_params": gp,
                                  "cim_params": PARAMS[ds]["CIM"], "cim_rounds": CIM_ROUNDS[ds],
                                  "ga_seeds": ga_seeds.tolist(), "cim_seeds": cim_seeds.tolist()}
        log(f"{ds} CIM: {len(cim_seeds)} independent solutions, {CIM_ROUNDS[ds]} rounds")
        started = time.perf_counter()
        cim_cuts, cim_signs = run_cim(ctx, CIM_ROUNDS[ds], len(cim_seeds), cim_seeds)
        np.testing.assert_allclose(ctx.score(cim_signs), cim_cuts, rtol=0, atol=1e-7)
        meta["datasets"][ds]["cim_batch_seconds"] = time.perf_counter() - started
        for condition in LABELS:
            init = None if condition == "random" else cim_signs.reshape(args.num_trials, size, ctx.n)
            records, fit_records, generations, snapshots = [], [], [], {}
            started = time.perf_counter()
            last_log = started

            def observe(generation, populations, fits):
                nonlocal last_log
                records.append(population_metrics(populations))
                fit_records.append(fits)
                generations.append(generation)
                if generation in CHECKPOINTS or generation == args.generations:
                    recomputed = ctx.score(populations.reshape(-1, ctx.n)).reshape(fits.shape)
                    np.testing.assert_allclose(recomputed, fits, rtol=0, atol=1e-7)
                    snapshots[f"gen_{generation}"] = np.packbits(populations, axis=-1, bitorder="little")
                if generation in {0, args.generations} or time.perf_counter() - last_log > 25:
                    log(f"{ds} {condition}: gen={generation}, D={records[-1][:,0].mean():.4f}, "
                        f"pool_best_gap={(ctx.bks - fits.max(axis=1)).mean():.3f}, "
                        f"elapsed={time.perf_counter()-started:.1f}s")
                    last_log = time.perf_counter()

            best, signs, history = simulate_ga_batch(
                ctx.n, ctx.edges, ctx.weights, args.num_trials,
                **gp, seeds=ga_seeds, max_generations=args.generations,
                init_population=init, population_observer=observe, return_history=True)
            np.testing.assert_allclose(ctx.score(signs), best, rtol=0, atol=1e-7)
            best_ever = np.vstack([fit_records[0].max(axis=1), history])
            np.savez_compressed(out / f"{ds}_{condition}_trajectory.npz", generation=generations,
                                metrics=np.asarray(records), population_fits=np.asarray(fit_records),
                                best_ever=best_ever, best_signs=signs, seeds=ga_seeds)
            np.savez_compressed(out / f"{ds}_{condition}_snapshots.npz", **snapshots)
            meta["datasets"][ds][f"{condition}_ga_batch_seconds"] = time.perf_counter() - started
        (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        plot_results(out, list(meta["datasets"]))
    shutil.copy2(__file__, out / "runner_source.py")
    shutil.copy2(ROOT / "modules/GA.py", out / "ga_source.py")
    log("COMPLETE: all snapshot scores and final scores verified; plots saved")


if __name__ == "__main__":
    main()
