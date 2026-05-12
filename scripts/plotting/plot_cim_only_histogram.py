"""compare_histogram.png の CIM パネルだけを単独画像として出力する。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch


def main():
    NUM_TRIALS = 100
    SEED_BASE = 0
    CIM_NUM_ROUNDS = 1500
    CIM_COUPLING = -0.03
    CIM_PARAMS = dict(
        kappa=130.0, L=0.05, gamma=42.09,
        eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
    )
    KNOWN_BEST = 13359

    n, k_edges, adj, edges = load_graph("input/G22.txt")
    print(f"Graph: N={n}, K={k_edges}")

    seeds = np.arange(SEED_BASE, SEED_BASE + NUM_TRIALS, dtype=np.int64)

    print(f"\n[CIM] {NUM_TRIALS} trials (num_rounds={CIM_NUM_ROUNDS})...")
    J = build_coupling_matrix(n, edges, CIM_COUPLING)
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n,
        J=J,
        edges=edges,
        num_rounds=CIM_NUM_ROUNDS,
        num_trials=NUM_TRIALS,
        seeds=seeds,
        **CIM_PARAMS,
    )
    cim_time = time.time() - t0
    print(f"  time: {cim_time:.2f} sec  ({cim_time / NUM_TRIALS * 1000:.1f} ms/trial)")
    print(f"  mean={cim_cuts.mean():.1f}  best={cim_cuts.max()}  worst={cim_cuts.min()}  std={cim_cuts.std():.2f}")

    os.makedirs("results", exist_ok=True)

    # compare_histogram.png の CIM パネルと同一スタイル
    fig, ax = plt.subplots(figsize=(5, 4.8))

    x_min = int(cim_cuts.min()) - 20
    x_max = max(int(cim_cuts.max()) + 20, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 35)

    color = "#1f77b4"
    ax.hist(
        cim_cuts, bins=bins, color=color, alpha=0.75,
        edgecolor="black", linewidth=0.5,
    )
    ax.axvline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3,
               label=f"既知最良値 {KNOWN_BEST}")
    ax.axvline(cim_cuts.mean(), color="black", linestyle=":", linewidth=1.3,
               label=f"平均 {cim_cuts.mean():.0f}")
    ax.set_title(
        f"CIM\n総時間: {cim_time:.1f}s  "
        f"平均: {cim_cuts.mean():.0f}  最良: {int(cim_cuts.max())}",
        fontsize=11,
    )
    ax.set_xlabel("カット値")
    ax.set_ylabel("頻度")
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"CIM — G22 の MAX-CUT ({NUM_TRIALS} 試行)",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = os.path.join("results", "compare_histogram_cim_only.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
