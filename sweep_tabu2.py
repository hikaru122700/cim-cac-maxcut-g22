import time
import numpy as np
from modules.CIM_2026_05_25 import (
    load_graph, build_coupling_matrix, simulate_cim_batch_polished,
)

n, k, adj, edges = load_graph("input/G22.txt")
J = build_coupling_matrix(n, edges, -0.03)
eta = 10.0 ** (-1.1)
seeds = np.arange(0, 20, dtype=np.int64)

base = dict(
    n=n, J=J, edges=edges, num_rounds=1500, num_trials=20,
    kappa=130.0, L=0.05, gamma=42.09, eta=eta,
    bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    seeds=seeds,
)
_ = simulate_cim_batch_polished(**{**base, "num_trials": 1, "seeds": np.array([0],dtype=np.int64),
                                    "tabu_iters": 100, "tabu_tenure": 50})

configs = [
    (5000, 80), (5000, 100), (5000, 150), (5000, 200),
    (10000, 80), (10000, 100), (10000, 150),
    (20000, 80), (20000, 100), (20000, 150),
    (50000, 100), (50000, 150),
]
print(f"{'iters':>8}{'tenure':>8}{'mean':>10}{'best':>8}{'worst':>8}{'wall':>8}")
for it, te in configs:
    t0 = time.perf_counter()
    cuts, _ = simulate_cim_batch_polished(**base, tabu_iters=it, tabu_tenure=te)
    dt = time.perf_counter() - t0
    print(f"{it:>8}{te:>8}{cuts.mean():>10.1f}{int(cuts.max()):>8}{int(cuts.min()):>8}{dt:>8.2f}")
