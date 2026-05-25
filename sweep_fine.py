import time
import numpy as np
from modules.CIM_2026_05_25 import (
    load_graph, build_coupling_matrix, simulate_cim_batch_polished,
)

n, k, adj, edges = load_graph("input/G22.txt")
J = build_coupling_matrix(n, edges, -0.03)
eta = 10.0 ** (-1.1)
# 100 trial で測る
seeds = np.arange(0, 100, dtype=np.int64)
base = dict(
    n=n, J=J, edges=edges, num_rounds=1500, num_trials=100,
    kappa=130.0, L=0.05, gamma=42.09, eta=eta,
    bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    seeds=seeds,
)
_ = simulate_cim_batch_polished(**{**base, "num_trials": 1, "seeds": np.array([0],dtype=np.int64),
                                    "tabu_iters": 1000, "tabu_tenure":50, "ils_outer": 1, "ils_perturb": 50})

configs = [
    # (tabu_iters, tenure, outer, perturb)
    (150_000, 100, 20, 50),
    (150_000, 150, 20, 50),
    (150_000, 200, 20, 50),
    (150_000, 100, 30, 30),
    (150_000, 150, 30, 30),
    (200_000, 100, 20, 50),
    (200_000, 200, 15, 80),
]
print(f"{'iters':>8}{'ten':>5}{'out':>5}{'pert':>5}{'mean':>10}{'best':>8}{'worst':>8}{'std':>7}{'wall':>8}")
for ti, te, ou, pe in configs:
    t0 = time.perf_counter()
    cuts, _ = simulate_cim_batch_polished(**base, tabu_iters=ti, tabu_tenure=te, ils_outer=ou, ils_perturb=pe)
    dt = time.perf_counter() - t0
    print(f"{ti:>8}{te:>5}{ou:>5}{pe:>5}{cuts.mean():>10.1f}{int(cuts.max()):>8}{int(cuts.min()):>8}{cuts.std(ddof=1):>7.2f}{dt:>8.1f}")
