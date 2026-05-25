import time
import numpy as np
from modules.CIM_2026_05_25 import (
    load_graph, build_coupling_matrix, simulate_cim_batch_polished,
)

n, k, adj, edges = load_graph("input/G22.txt")
J = build_coupling_matrix(n, edges, -0.03)
eta = 10.0 ** (-1.1)
seeds = np.arange(0, 100, dtype=np.int64)
base = dict(
    n=n, J=J, edges=edges, num_rounds=1500, num_trials=100,
    kappa=130.0, L=0.05, gamma=42.09, eta=eta,
    bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    seeds=seeds, tabu_tenure=150,
)
_ = simulate_cim_batch_polished(**{**base, "num_trials": 1, "seeds": np.array([0],dtype=np.int64),
                                    "tabu_iters": 1000, "ils_outer": 1, "ils_perturb": 100})

configs = [
    (100_000, 15, 600),
    (100_000, 15, 1000),
    (100_000, 15, 1500),
    (100_000, 20, 1000),
    (150_000, 15, 600),
    (50_000, 30, 600),
    (50_000, 30, 1000),
]
print(f"{'iters':>8}{'out':>5}{'pert':>5}{'mean':>10}{'best':>8}{'worst':>8}{'std':>7}{'wall':>8}")
for ti, ou, pe in configs:
    t0 = time.perf_counter()
    cuts, _ = simulate_cim_batch_polished(**base, tabu_iters=ti, ils_outer=ou, ils_perturb=pe)
    dt = time.perf_counter() - t0
    print(f"{ti:>8}{ou:>5}{pe:>5}{cuts.mean():>10.1f}{int(cuts.max()):>8}{int(cuts.min()):>8}{cuts.std(ddof=1):>7.2f}{dt:>8.1f}")
