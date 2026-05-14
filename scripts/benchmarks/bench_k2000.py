"""K2000 (WK2000) で CIM, SB 全 5 バリアント, SA を回し性能を比較する。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
from modules.SB import simulate_sb_batch, auto_c0
from modules.SA import simulate_sa_batch


NUM_TRIALS = 100
NUM_STEPS_SB = 1000
NUM_ROUNDS_CIM = 1500
NUM_ITERS_SA = 200_000   # SA は内部反復が多いほど良い、ベンチで時間と相談
SEED_BASE = 0
GAMMA = {"HbSB": 0.5, "HdSB": 0.06}


def main():
    print("Loading K2000 ...")
    t0 = time.time()
    n, k_edges, _, edges, weights = load_graph("input/K2000.txt", return_weights=True)
    print(f"  N={n}, K={k_edges}, load_time={time.time()-t0:.1f}s")

    seeds = np.arange(SEED_BASE, SEED_BASE + NUM_TRIALS, dtype=np.int64)

    results = {}

    # ===== SB 5 バリアント =====
    J_sb = build_coupling_matrix(n, edges, coupling=-1.0, weights=weights)
    c0 = auto_c0(J_sb, n)
    print(f"\nSB c0 (auto) = {c0:.5f}")

    for variant in ["aSB", "bSB", "dSB", "HbSB", "HdSB"]:
        gamma = GAMMA.get(variant, 0.0)
        print(f"\n[{variant}] gamma={gamma} num_steps={NUM_STEPS_SB} ...")
        t0 = time.time()
        cuts, _ = simulate_sb_batch(
            n=n, J=J_sb, edges=edges,
            num_steps=NUM_STEPS_SB, num_trials=NUM_TRIALS,
            variant=variant, gamma_heat=gamma,
            weights=weights,
            seeds=seeds,
        )
        elapsed = time.time() - t0
        results[variant] = {"cuts": cuts, "time": elapsed, "extra": f"γ={gamma}"}
        print(f"  mean={cuts.mean():.1f}  std={cuts.std():.1f}  "
              f"best={cuts.max():.0f}  worst={cuts.min():.0f}  "
              f"time={elapsed:.2f}s ({elapsed/NUM_TRIALS*1000:.0f} ms/trial)")

    # ===== CIM =====
    # CIM は論文パラメータ。J は coupling = -0.03 × weight
    print(f"\n[CIM] paper params, num_rounds={NUM_ROUNDS_CIM} ...")
    J_cim = build_coupling_matrix(n, edges, coupling=-0.03, weights=weights)
    cim_params = dict(
        kappa=130.0, L=0.05, gamma=42.09,
        eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
    )
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n, J=J_cim, edges=edges,
        num_rounds=NUM_ROUNDS_CIM, num_trials=NUM_TRIALS,
        seeds=seeds, weights=weights,
        **cim_params,
    )
    elapsed = time.time() - t0
    results["CIM"] = {"cuts": cim_cuts, "time": elapsed, "extra": f"rounds={NUM_ROUNDS_CIM}"}
    print(f"  mean={cim_cuts.mean():.1f}  std={cim_cuts.std():.1f}  "
          f"best={cim_cuts.max():.0f}  worst={cim_cuts.min():.0f}  "
          f"time={elapsed:.2f}s ({elapsed/NUM_TRIALS*1000:.0f} ms/trial)")

    # ===== SA =====
    print(f"\n[SA]  iters={NUM_ITERS_SA} ...")
    t0 = time.time()
    sa_cuts, _ = simulate_sa_batch(
        n=n, edges=edges, weights=weights,
        num_iters=NUM_ITERS_SA, num_trials=NUM_TRIALS,
        t_start=2.0, t_end=0.001,
        seeds=seeds,
    )
    elapsed = time.time() - t0
    results["SA"] = {"cuts": sa_cuts, "time": elapsed, "extra": f"iters={NUM_ITERS_SA}"}
    print(f"  mean={sa_cuts.mean():.1f}  std={sa_cuts.std():.1f}  "
          f"best={sa_cuts.max():.0f}  worst={sa_cuts.min():.0f}  "
          f"time={elapsed:.2f}s ({elapsed/NUM_TRIALS*1000:.0f} ms/trial)")

    # ===== サマリー =====
    print("\n" + "=" * 90)
    print(f"{'method':<7} {'param':<14} {'mean':>10} {'best':>10} {'worst':>10} {'std':>8} {'time[s]':>10}")
    print("-" * 90)
    method_order = ["aSB", "bSB", "dSB", "HbSB", "HdSB", "CIM", "SA"]
    for name in method_order:
        r = results[name]
        cuts = r["cuts"]
        print(f"{name:<7} {r['extra']:<14} {cuts.mean():>10.1f} "
              f"{cuts.max():>10.0f} {cuts.min():>10.0f} "
              f"{cuts.std():>8.2f} {r['time']:>10.2f}")
    print("=" * 90)
    print("cut 値は重み付き MAX-CUT 値 (sum w * indicator), w in +-1.")

    # ===== ヒストグラム =====
    fig, axes = plt.subplots(1, 7, figsize=(28, 4.8), dpi=110)
    colors = {
        "aSB":  "#9467bd", "bSB":  "#1f77b4", "dSB":  "#d62728",
        "HbSB": "#2ca02c", "HdSB": "#ff7f0e",
        "CIM":  "#17becf", "SA":   "#8c564b",
    }
    all_cuts = np.concatenate([r["cuts"] for r in results.values()])
    x_min = int(all_cuts.min()) - 50
    x_max = int(all_cuts.max()) + 50
    bins = np.linspace(x_min, x_max, 35)
    best_overall = float(all_cuts.max())

    for ax, name in zip(axes, method_order):
        cuts = results[name]["cuts"]
        t = results[name]["time"]
        ax.hist(cuts, bins=bins, color=colors[name], alpha=0.75,
                edgecolor="black", linewidth=0.4)
        ax.axvline(cuts.mean(), color="black", linestyle=":", linewidth=1.3,
                   label=f"平均 {cuts.mean():.0f}")
        ax.axvline(best_overall, color="goldenrod", linestyle="--", linewidth=1.2,
                   label=f"全体最良 {best_overall:.0f}")
        ax.set_title(
            f"{name}  ({results[name]['extra']})\n"
            f"時間 {t:.1f}s  平均 {cuts.mean():.0f}  最良 {cuts.max():.0f}",
            fontsize=10,
        )
        ax.set_xlabel("重み付き cut 値")
        ax.set_ylabel("頻度")
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"K2000 (N=2000 all-to-all, J∈±1, {k_edges} 辺) — CIM/SB/SA 性能比較 "
        f"({NUM_TRIALS} 試行)",
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = Path("results") / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "v1_k2000_full_benchmark.png"
    i = 1
    while out.exists():
        i += 1
        out = out_dir / f"v{i}_k2000_full_benchmark.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
