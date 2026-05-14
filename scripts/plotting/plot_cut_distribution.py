"""100 試行を回し、best_cut の分布をヒストグラムとして保存する。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules.CIM import load_graph, build_coupling_matrix, simulate_cim_batch


def main():
    config = {
        "kappa": 130.0,
        "L": 0.05,
        "gamma": 42.09,
        "loss_dB": 11.0,
        "bandwidth": 1.0e9,
        "photon_energy_J": 1.28e-19,
        "dP_per_round": 0.05e-3,
        "coupling": -0.03,
        "num_rounds": 1500,
        "num_trials": 100,
        "base_seed": 42,
    }
    eta = 10.0 ** (-config["loss_dB"] / 10.0)

    n, k_edges, adj, edges = load_graph("input/G22.txt")
    print(f"N={n}, K={k_edges}, eta={eta:.4f}")

    J = build_coupling_matrix(n, edges, config["coupling"])

    rng = np.random.default_rng(config["base_seed"])
    seeds = rng.integers(0, 2**63 - 1, size=config["num_trials"]).astype(np.int64)

    print(f"Running {config['num_trials']} trials...")
    t0 = time.perf_counter()
    best_cuts, best_signs = simulate_cim_batch(
        n=n,
        J=J,
        edges=edges,
        num_rounds=config["num_rounds"],
        num_trials=config["num_trials"],
        kappa=config["kappa"],
        L=config["L"],
        gamma=config["gamma"],
        eta=eta,
        bandwidth=config["bandwidth"],
        photon_energy=config["photon_energy_J"],
        dP_per_round=config["dP_per_round"],
        seeds=seeds,
    )
    elapsed = time.perf_counter() - t0

    mean_c = float(np.mean(best_cuts))
    std_c = float(np.std(best_cuts))
    best_c = int(np.max(best_cuts))
    worst_c = int(np.min(best_cuts))
    median_c = float(np.median(best_cuts))

    print(f"Elapsed: {elapsed:.2f} s ({elapsed/config['num_trials']*1000:.0f} ms/trial)")
    print(f"mean={mean_c:.2f}, std={std_c:.2f}, best={best_c}, worst={worst_c}, median={median_c:.1f}")

    paper_mean = 13275
    paper_best = 13321
    known_best = 13359

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

    bins = np.arange(best_cuts.min() - 1, best_cuts.max() + 2) - 0.5
    counts, _, _ = ax.hist(
        best_cuts,
        bins=bins,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
        label=f"This run ({config['num_trials']} trials)",
    )
    ymax = counts.max() * 1.25

    ax.axvline(mean_c, color="#C44E52", linestyle="--", linewidth=1.8,
               label=f"This run mean = {mean_c:.1f}")
    ax.axvline(best_c, color="#55A868", linestyle="--", linewidth=1.8,
               label=f"This run best = {best_c}")
    ax.axvline(paper_mean, color="black", linestyle=":", linewidth=1.5,
               label=f"Paper Fig.8 mean = {paper_mean}")
    ax.axvline(paper_best, color="dimgray", linestyle=":", linewidth=1.5,
               label=f"Paper Fig.8 best = {paper_best}")
    ax.axvline(known_best, color="goldenrod", linestyle="-", linewidth=1.5,
               label=f"Known best = {known_best}")

    ax.set_xlabel("best_cut")
    ax.set_ylabel("frequency")
    ax.set_title(
        f"CIM on G22 — distribution of best_cut over {config['num_trials']} trials\n"
        f"(rounds={config['num_rounds']}, mean={mean_c:.1f}, std={std_c:.2f}, "
        f"best={best_c}, worst={worst_c})"
    )
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)

    out_path = "results/cut_distribution_100trials.png"
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
