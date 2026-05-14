"""wandb なしで CIM を1回実行する確認用スクリプト。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np

from modules.CIM import (
    load_graph,
    build_coupling_matrix,
    simulate_cim,
)
from modules.verify import run_all_checks


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
        "seed": 42,
    }
    eta = 10.0 ** (-config["loss_dB"] / 10.0)
    rng = np.random.default_rng(config["seed"])

    n, k_edges, adj, edges = load_graph("input/G22.txt")
    print(f"N={n}, K={k_edges}, eta={eta:.4f}")

    J = build_coupling_matrix(n, edges, config["coupling"])

    print("Running CIM simulation (wandb off, JIT fast path)...")
    t0 = time.perf_counter()
    c_final, best_cut, best_x = simulate_cim(
        n=n,
        J=J,
        edges=edges,
        num_rounds=config["num_rounds"],
        kappa=config["kappa"],
        L=config["L"],
        gamma=config["gamma"],
        eta=eta,
        bandwidth=config["bandwidth"],
        photon_energy=config["photon_energy_J"],
        dP_per_round=config["dP_per_round"],
        rng=rng,
        wandb_log=False,
    )
    elapsed = time.perf_counter() - t0

    print(f"Best cut: {best_cut}")
    print(f"Paper (Fig.8 G22): mean=13275, best=13321")
    print(f"Known best: 13359")
    print(f"Ratio to known best: {best_cut / 13359:.4f}")
    print(f"Elapsed: {elapsed:.2f} s")

    run_all_checks(best_x, n, k_edges, adj, edges, best_cut)


if __name__ == "__main__":
    main()
