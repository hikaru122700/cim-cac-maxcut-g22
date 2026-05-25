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
    seeds=seeds, kl_passes=4,
)
_ = simulate_cim_batch_polished(**{**base, "num_trials": 1, "seeds": np.array([0],dtype=np.int64),
                                    "ils_iters": 5, "ils_perturb": 20})

configs = [
    (50, 500), (50, 800), (50, 1000),
    (100, 300), (100, 500), (100, 800),
    (200, 300), (200, 500),
    (300, 300),
]
print(f"{'iters':>6}{'perturb':>10}{'mean':>10}{'best':>8}{'wall':>10}")
for it, pe in configs:
    t0 = time.perf_counter()
    cuts, _ = simulate_cim_batch_polished(**base, ils_iters=it, ils_perturb=pe)
    dt = time.perf_counter() - t0
    print(f"{it:>6}{pe:>10}{cuts.mean():>10.1f}{int(cuts.max()):>8}{dt:>10.2f}")
