"""ILS-KL の (perturb_size, iters) スイープ。20 trial で速攻測定。"""
import time
import numpy as np
from modules.CIM_2026_05_25 import (
    load_graph, build_coupling_matrix, simulate_cim_batch_polished,
)

n, k, adj, edges = load_graph("input/G22.txt")
J = build_coupling_matrix(n, edges, -0.03)
eta = 10.0 ** (-1.1)
seeds = np.arange(0, 20, dtype=np.int64)

base_kwargs = dict(
    n=n, J=J, edges=edges, num_rounds=1500, num_trials=20,
    kappa=130.0, L=0.05, gamma=42.09, eta=eta,
    bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    seeds=seeds, kl_passes=4,
)

# warmup
_ = simulate_cim_batch_polished(**{**base_kwargs, "num_trials": 1, "seeds": np.array([0],dtype=np.int64),
                                    "ils_iters": 5, "ils_perturb": 20})

configs = [
    (20, 40), (50, 40), (100, 40),
    (50, 80), (50, 150), (50, 300),
    (100, 80), (100, 150),
    (200, 80),
]
print(f"{'iters':>6}{'perturb':>10}{'mean':>10}{'best':>8}{'wall':>10}")
for ils_iters, ils_perturb in configs:
    t0 = time.perf_counter()
    cuts, _ = simulate_cim_batch_polished(**base_kwargs, ils_iters=ils_iters, ils_perturb=ils_perturb)
    dt = time.perf_counter() - t0
    print(f"{ils_iters:>6}{ils_perturb:>10}{cuts.mean():>10.1f}{int(cuts.max()):>8}{dt:>10.2f}")
