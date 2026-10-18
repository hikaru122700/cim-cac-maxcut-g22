"""Replay recorded GA runs and trace actual crossover, local search and acceptance.

Initial populations, parameters and seeds come from the previous experiment.
Every generation's population fitness and best history must exactly reproduce it.
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from modules.GA import load_graph, simulate_ga_batch

EXPERIMENT_KIND = "ga_offspring_diagnostics"
DEFAULT_SOURCE = ROOT / "results/2026-09-14/ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_measured"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(events):
    """All event arrays use (generation, run, ...); runs are statistical units."""
    parent_best = events["parent_fits"].max(axis=2)
    parent_worst = events["parent_fits"].min(axis=2)
    before = events["child_before_fit"]
    after = events["child_after_fit"]
    accepted = events["accepted"]
    recovered = after >= parent_best
    improved = after > parent_best
    ts_improved = after > before
    best_improved = after > events["best_before"]
    flags = {"raw_below_better_parent": before < parent_best,
             "raw_below_both_parents": before < parent_worst,
             "ts_recovered_better_parent": recovered,
             "ts_recovered_worse_parent": after >= parent_worst,
             "ts_exceeded_better_parent": improved,
             "ts_improved_raw_child": ts_improved,
             "accepted": accepted, "new_best_found": best_improved,
             "below_pool_worst_after_ts": after < events["pool_fits_before"].min(axis=2)}
    out = {"num_children": int(before.size), "generations": int(before.shape[0]),
           "runs": int(before.shape[1]), "counts": {}, "per_run_counts": {}}
    for name, flag in flags.items():
        out["counts"][name] = int(flag.sum())
        out["per_run_counts"][name] = flag.sum(axis=0).tolist()
    for name, flag in [("parent_improved", improved), ("parent_recovered", recovered),
                       ("TS_improved", ts_improved), ("new_best", best_improved)]:
        out[name + "_acceptance"] = {
            "eligible": int(flag.sum()), "accepted": int((flag & accepted).sum()),
            "rejected": int((flag & ~accepted).sum()),
            "rate": float(accepted[flag].mean()) if flag.any() else None,
        }
    out["means"] = {
        "raw_minus_better_parent": float((before-parent_best).mean()),
        "refined_minus_better_parent": float((after-parent_best).mean()),
        "TS_gain": float((after-before).mean()),
        "parent_cut": float(parent_best.mean()), "raw_child_cut": float(before.mean()),
        "refined_child_cut": float(after.mean()),
    }
    out["refined_minus_better_parent_quantiles"] = np.quantile(after-parent_best, [0, .05, .5, .95, 1]).tolist()
    out["per_run_means"] = {
        "raw_minus_better_parent": (before-parent_best).mean(axis=0).tolist(),
        "refined_minus_better_parent": (after-parent_best).mean(axis=0).tolist(),
        "TS_gain": (after-before).mean(axis=0).tolist(),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--datasets", nargs="+", default=["G70", "G55", "G22"], choices=["G22", "G55", "G70"])
    parser.add_argument("--tag", default="replay")
    args = parser.parse_args()
    if args.tag and not all(c.isascii() and (c.isalnum() or c == "_") for c in args.tag):
        parser.error("tag must use ASCII letters, digits and underscores")
    old_meta = json.loads((args.source / "metadata.json").read_text(encoding="utf-8"))
    trials, generations = old_meta["num_trials"], old_meta["generations"]
    kind = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name.split("_")[0][1:]) for p in kind.iterdir()
                if p.is_dir() and p.name.split("_")[0][1:].isdigit()]
    desc = f"{'_'.join(args.datasets)}_nt{trials}_gen{generations}_{args.tag}"
    out = kind / f"v{max(versions, default=0)+1}_{desc}"
    out.mkdir(exist_ok=False)

    def log(msg):
        print(msg, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as stream:
            stream.write(msg + "\n")

    meta = {"source": str(args.source.relative_to(ROOT)), "source_metadata": old_meta,
            "datasets": args.datasets, "num_trials": trials, "generations": generations,
            "spins_order": ["parent_a", "parent_b", "child_before_ts", "child_after_ts"],
            "bitorder": "little", "sha256": {"source_metadata": sha256(args.source / "metadata.json"),
            "modules/GA.py": sha256(ROOT / "modules/GA.py"), "runner": sha256(Path(__file__))},
            "replay_checks": {}, "summary": {}}
    log(f"OUTPUT={out}")
    for ds in args.datasets:
        info = old_meta["datasets"][ds]
        n, edges, weights = load_graph(str(ROOT / f"input/{ds}.txt"), return_weights=True)
        assert n == info["n"]
        graph_path = ROOT / f"input/{ds}.txt"
        assert sha256(graph_path) == old_meta["sha256"][f"input/{ds}.txt"]
        edge_array = np.asarray(edges, dtype=np.int64)
        a, b = edge_array.T
        w = np.asarray(weights)

        def score(spins):
            # Independent edge sum, not GA's incremental fitness or CSR scorer.
            return ((spins[..., a] != spins[..., b]) * w).sum(axis=-1)

        meta["summary"][ds] = {}
        for condition in ["cim", "random"]:
            key = f"{ds}_{condition}"
            old_path = args.source / f"{key}_trajectory.npz"
            snap_path = args.source / f"{key}_snapshots.npz"
            meta["sha256"][old_path.name] = sha256(old_path)
            meta["sha256"][snap_path.name] = sha256(snap_path)
            with np.load(old_path) as archive:
                old = {name: archive[name] for name in archive.files}
            with np.load(snap_path) as archive:
                old_snapshots = {name: archive[name] for name in archive.files}
            initial = np.unpackbits(old_snapshots["gen_-1"], axis=-1, count=n, bitorder="little")
            rows, packed = [], []
            started = last_log = time.perf_counter()

            def observe_population(g, pop, fits):
                index = g + 1
                np.testing.assert_array_equal(fits, old["population_fits"][index])
                name = f"gen_{g}"
                if name in old_snapshots:
                    np.testing.assert_array_equal(np.packbits(pop, axis=-1, bitorder="little"), old_snapshots[name])

            def observe_child(g, event):
                nonlocal last_log
                parent_fits = score(event["parents"])
                np.testing.assert_array_equal(parent_fits, event["parent_fits"])
                raw = score(event["child_before_ts"])
                refined = score(event["child_after_ts"])
                np.testing.assert_array_equal(refined, event["child_after_fit"])
                assert np.all(refined >= raw)
                row = {name: event[name] for name in ["parent_indices", "parent_fits", "pool_fits_before",
                                                     "pool_fits_after", "best_before", "child_after_fit", "accepted"]}
                row["child_before_fit"] = raw
                rows.append(row)
                spins = np.concatenate([event["parents"], event["child_before_ts"][:, None, :],
                                        event["child_after_ts"][:, None, :]], axis=1)
                packed.append(np.packbits(spins, axis=-1, bitorder="little"))
                if g == 1 or g == generations or time.perf_counter()-last_log >= 25:
                    pb = parent_fits.max(axis=1)
                    log(f"{key} gen={g}: crossover_delta={(raw-pb).mean():.1f}, "
                        f"after_TS_delta={(refined-pb).mean():.1f}, accepted={event['accepted'].sum()}/{trials}, "
                        f"elapsed={time.perf_counter()-started:.1f}s")
                    last_log = time.perf_counter()

            best, signs, history = simulate_ga_batch(
                n, edges, weights, trials, **info["ga_params"],
                seeds=np.asarray(info["ga_seeds"]), init_population=initial,
                max_generations=generations, return_history=True,
                population_observer=observe_population, offspring_observer=observe_child)
            np.testing.assert_array_equal(history, old["best_ever"][1:])
            np.testing.assert_array_equal(signs, old["best_signs"])
            np.testing.assert_array_equal(score(signs), best)
            events = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
            np.savez_compressed(out / f"{key}_events.npz", **events)
            np.savez_compressed(out / f"{key}_spins.npz", spins=np.asarray(packed), n=n)
            meta["summary"][ds][condition] = summarize(events)
            meta["replay_checks"][key] = {"all_generation_fitness_exact": True,
                "saved_population_snapshots_exact": True, "best_history_exact": True,
                "final_solutions_exact": True, "independent_stage_scores_checked": trials*generations*4,
                "elapsed_seconds": time.perf_counter()-started}
            (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            log(f"{key} COMPLETE: exact replay confirmed; {generations*trials} offspring recorded")
    shutil.copy2(__file__, out / "runner_source.py")
    shutil.copy2(ROOT / "modules/GA.py", out / "ga_source.py")
    log("COMPLETE: all conditions and all stage scores verified")


if __name__ == "__main__":
    main()
