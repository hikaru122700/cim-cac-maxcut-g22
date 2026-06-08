#!/usr/bin/env python
"""Smoke driver for the cim-cac-maxcut-g22 solvers.

Drives the actual CIM simulator (no wandb, no plotting) on a G-Set graph,
then independently re-verifies the returned cut. This is the programmatic
handle a future agent uses to confirm the solver core works after a change.

Run from the project root:
    uv run python .claude/skills/run-cim-cac-maxcut-g22/driver.py
    uv run python .claude/skills/run-cim-cac-maxcut-g22/driver.py --graph G15 --trials 8

Exit code 0 = solver ran AND the cut re-verified AND ratio>=threshold.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

# modules/verify.py prints Japanese; on Windows the default cp932 console would
# raise UnicodeEncodeError. Force UTF-8 so the driver is self-contained.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# skill is at <root>/.claude/skills/run-cim-cac-maxcut-g22/driver.py → root = parents[3]
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
from modules.verify import compute_cut_from_edges, verify_solution

# G22 is the repo's central benchmark; known-best values from README.
KNOWN_BEST = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}

# Physical params: paper values (Inoue & Yoshida 2022), identical to modules/CIM.py main().
CIM_PARAMS = dict(
    kappa=130.0, L=0.05, gamma=42.09, eta=10.0 ** (-1.1),
    bandwidth=1.0e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="G22", help="G-Set name in input/ (default G22)")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--min-ratio", type=float, default=0.99,
                    help="fail if best_cut/known_best is below this")
    args = ap.parse_args()

    graph_path = ROOT / "input" / f"{args.graph}.txt"
    if not graph_path.exists():
        print(f"[FAIL] graph not found: {graph_path}")
        return 1

    n, k, adj, edges = load_graph(str(graph_path))
    J = build_coupling_matrix(n, edges, -0.03)  # G22: antiferromagnetic, cut-promoting
    seeds = np.arange(args.trials, dtype=np.int64)

    print(f"[run] CIM {args.graph} N={n} K={k}  rounds={args.rounds} trials={args.trials}")
    print("[run] first call includes Numba JIT compile (a few seconds)...")
    t0 = time.time()
    best_cuts, best_signs = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=args.rounds, num_trials=args.trials,
        seeds=seeds, **CIM_PARAMS,
    )
    dt = time.time() - t0

    bi = int(np.argmax(best_cuts))
    best_cut = int(best_cuts[bi])
    x = best_signs[bi].astype(np.int64).tolist()

    # Independent re-verification (does NOT trust the kernel's own cut count).
    recut = compute_cut_from_edges(x, edges)
    bipartite_ok = verify_solution(x, n)  # all assignments are 0/1
    known = KNOWN_BEST.get(args.graph)
    ratio = best_cut / known if known else float("nan")

    print(f"[result] best_cut={best_cut}  independent_recut={recut}  "
          f"ratio_to_known_best={ratio:.4f}  wall={dt:.1f}s")

    ok = (recut == best_cut) and bipartite_ok and (known is None or ratio >= args.min_ratio)
    print("[PASS] CIM solver verified" if ok else "[FAIL] solver/verification mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
