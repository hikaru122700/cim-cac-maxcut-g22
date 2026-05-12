"""SB 全 5 バリアント (aSB, bSB, dSB, HbSB, HdSB) を G22 で比較。"""

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

from modules.CIM import build_coupling_matrix, load_graph
from modules.SB import simulate_sb_batch, auto_c0


NUM_TRIALS = 100
NUM_STEPS = 1000
SEED_BASE = 0
KNOWN_BEST = 13359

# Kanao-Goto 2022 の論文推奨値(K2000 で最適化済み)
GAMMA = {"HbSB": 0.5, "HdSB": 0.06}


def main():
    n, k_edges, _, edges = load_graph("input/G22.txt")
    print(f"N={n}, K={k_edges}")

    J = build_coupling_matrix(n, edges, -1.0)
    c0 = auto_c0(J, n)
    print(f"c0 (auto) = {c0:.4f}")

    seeds = np.arange(SEED_BASE, SEED_BASE + NUM_TRIALS, dtype=np.int64)

    results = {}
    for variant in ["aSB", "bSB", "dSB", "HbSB", "HdSB"]:
        gamma = GAMMA.get(variant, 0.0)
        print(f"\n[{variant}] {NUM_TRIALS} trials, {NUM_STEPS} steps, γ={gamma}...")
        t0 = time.time()
        cuts, _ = simulate_sb_batch(
            n=n, J=J, edges=edges,
            num_steps=NUM_STEPS, num_trials=NUM_TRIALS,
            variant=variant,
            gamma_heat=gamma,
            seeds=seeds,
        )
        elapsed = time.time() - t0
        results[variant] = {"cuts": cuts, "time": elapsed, "gamma": gamma}
        print(f"  mean={cuts.mean():.2f}  std={cuts.std():.2f}  "
              f"best={int(cuts.max())}  worst={int(cuts.min())}  "
              f"time={elapsed:.2f}s ({elapsed/NUM_TRIALS*1000:.1f} ms/trial)")

    # ヒストグラム 5 パネル
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.8), dpi=120)
    colors = {
        "aSB":  "#9467bd",
        "bSB":  "#1f77b4",
        "dSB":  "#d62728",
        "HbSB": "#2ca02c",
        "HdSB": "#ff7f0e",
    }

    all_cuts = np.concatenate([r["cuts"] for r in results.values()])
    x_min = int(all_cuts.min()) - 30
    x_max = max(int(all_cuts.max()) + 30, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 40)

    for ax, variant in zip(axes, ["aSB", "bSB", "dSB", "HbSB", "HdSB"]):
        cuts = results[variant]["cuts"]
        t = results[variant]["time"]
        gamma = results[variant]["gamma"]
        ax.hist(cuts, bins=bins, color=colors[variant], alpha=0.75,
                edgecolor="black", linewidth=0.4)
        ax.axvline(cuts.mean(), color="black", linestyle=":", linewidth=1.5,
                   label=f"平均 {cuts.mean():.0f}")
        ax.axvline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
                   label=f"既知最良値 {KNOWN_BEST}")
        title_suffix = f"  γ={gamma}" if gamma > 0 else ""
        ax.set_title(
            f"{variant}{title_suffix}\n総時間: {t:.1f}s  "
            f"平均: {cuts.mean():.0f}  最良: {int(cuts.max())}",
            fontsize=10,
        )
        ax.set_xlabel("カット値")
        ax.set_ylabel("頻度")
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"SB 全 5 バリアント — G22 ({NUM_TRIALS} 試行, num_steps={NUM_STEPS})",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    out = "results/v1_sb_all_variants.png"
    i = 1
    while os.path.exists(out):
        i += 1
        out = f"results/v{i}_sb_all_variants.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")

    print("\n" + "=" * 78)
    print(f"{'method':<6} {'gamma':>6} {'mean':>10} {'best':>8} {'worst':>8} {'std':>8} {'time[s]':>10}")
    print("-" * 78)
    for variant in ["aSB", "bSB", "dSB", "HbSB", "HdSB"]:
        r = results[variant]
        cuts = r["cuts"]
        print(f"{variant:<6} {r['gamma']:>6.2f} {cuts.mean():>10.1f} "
              f"{int(cuts.max()):>8d} {int(cuts.min()):>8d} "
              f"{cuts.std():>8.2f} {r['time']:>10.2f}")
    print("=" * 78)
    print(f"Reference: Known best={KNOWN_BEST}")


if __name__ == "__main__":
    main()
