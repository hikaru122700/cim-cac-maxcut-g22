"""SB が決定論的であることを確認する:同じ seed なら完全に同じ結果。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from modules.CIM import build_coupling_matrix, load_graph
from modules.SB import simulate_sb_batch


def main():
    n, k_edges, _, edges = load_graph("input/G22.txt")
    J = build_coupling_matrix(n, edges, -1.0)

    NUM_TRIALS = 5
    NUM_STEPS = 1000

    for variant in ["aSB", "bSB", "dSB"]:
        # 同じ seed 配列で 2 回呼ぶ
        seeds = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        cuts_a, signs_a = simulate_sb_batch(
            n=n, J=J, edges=edges,
            num_steps=NUM_STEPS, num_trials=NUM_TRIALS,
            variant=variant, seeds=seeds,
        )
        cuts_b, signs_b = simulate_sb_batch(
            n=n, J=J, edges=edges,
            num_steps=NUM_STEPS, num_trials=NUM_TRIALS,
            variant=variant, seeds=seeds,
        )
        identical_cuts = np.array_equal(cuts_a, cuts_b)
        identical_signs = np.array_equal(signs_a, signs_b)
        print(f"[{variant}]")
        print(f"  Run A cuts: {cuts_a.tolist()}")
        print(f"  Run B cuts: {cuts_b.tolist()}")
        print(f"  cuts identical?  {identical_cuts}")
        print(f"  signs identical? {identical_signs}")

        # 異なる seed では違う結果になる
        seeds_diff = np.array([100, 101, 102, 103, 104], dtype=np.int64)
        cuts_c, _ = simulate_sb_batch(
            n=n, J=J, edges=edges,
            num_steps=NUM_STEPS, num_trials=NUM_TRIALS,
            variant=variant, seeds=seeds_diff,
        )
        print(f"  diff seed cuts: {cuts_c.tolist()}  (異なるはず)")
        print()


if __name__ == "__main__":
    main()
