"""Independently recompute saved GA population distances and edge-cut scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def mean_ci(values):
    mean = float(values.mean())
    width = float(student_t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "ci95": [mean - width, mean + width]}


def validate(out):
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    report = {"checked_populations": 0, "checked_scores": 0, "datasets": {}, "sha256": {}}
    for ds, info in meta["datasets"].items():
        edges = np.loadtxt(ROOT / f"input/{ds}.txt", skiprows=1)
        a = edges[:, 0].astype(int) - 1
        b = edges[:, 1].astype(int) - 1
        weights = edges[:, 2] if edges.shape[1] > 2 else np.ones(len(edges))
        report["datasets"][ds] = {}
        for condition in ["random", "cim"]:
            path = out / f"{ds}_{condition}_trajectory.npz"
            snap_path = out / f"{ds}_{condition}_snapshots.npz"
            with np.load(path) as z, np.load(snap_path) as snapshots:
                unchanged = np.ones(meta["num_trials"], dtype=bool)
                saved_post_ts_generations = []
                for key in snapshots.files:
                    generation = int(key.removeprefix("gen_"))
                    row = int(np.flatnonzero(z["generation"] == generation)[0])
                    pop = np.unpackbits(snapshots[key], axis=-1, count=info["n"], bitorder="little")
                    if generation >= 0:
                        unchanged &= np.all(snapshots[key] == snapshots["gen_0"], axis=(1, 2))
                        saved_post_ts_generations.append(generation)
                    for run, population in enumerate(pop):
                        size, n = population.shape
                        distances = np.zeros((size, size))
                        for i in range(size):
                            for j in range(i + 1, size):
                                h = np.count_nonzero(population[i] != population[j])
                                distances[i, j] = distances[j, i] = min(h, n-h) / n
                        dv = distances[np.triu_indices(size, 1)]
                        np.fill_diagonal(distances, np.inf)
                        canonical = population ^ population[:, :1]
                        computed = [dv.mean(), distances.min(axis=1).mean(), dv.min(),
                                    len(np.unique(canonical, axis=0))]
                        np.testing.assert_allclose(z["metrics"][row, run], computed, rtol=0, atol=1e-12)
                        cuts = ((population[:, a] != population[:, b]) * weights).sum(axis=1)
                        np.testing.assert_allclose(cuts, z["population_fits"][row, run], rtol=0, atol=1e-7)
                        report["checked_populations"] += 1
                        report["checked_scores"] += size
                final_cuts = ((z["best_signs"][:, a] != z["best_signs"][:, b]) * weights).sum(axis=1)
                np.testing.assert_allclose(final_cuts, z["best_ever"][-1], rtol=0, atol=1e-7)
                assert np.all(np.diff(z["best_ever"], axis=0) >= -1e-7)
                d = z["metrics"][:, :, 0]
                delta = d[-1] - d[1]
                report["datasets"][ds][condition] = {
                    "initial_TS_D_change": mean_ci(d[1] - d[0]),
                    "gen0_to_final_D_change": mean_ci(delta),
                    "gen0_to_final_cut_gain": mean_ci(z["best_ever"][-1] - z["best_ever"][1]),
                    "increases_in_D_across_all_run_generation_steps": int((np.diff(d[1:], axis=0) > 1e-12).sum()),
                    "generation_steps_total": int((d.shape[0] - 2) * d.shape[1]),
                    "runs_with_constant_D_all_generations": int(np.all(np.abs(d[1:] - d[1]) < 1e-12, axis=0).sum()),
                    "runs_identical_to_gen0_at_all_saved_checkpoints": int(unchanged.sum()),
                    "saved_post_ts_generations": saved_post_ts_generations,
                }
            for p in [path, snap_path]:
                report["sha256"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    report["all_checks_passed"] = True
    dest = out / "independent_validation.json"
    with dest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_directory", type=Path)
    validate(parser.parse_args().result_directory)
