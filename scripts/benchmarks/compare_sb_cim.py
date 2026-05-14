"""SB (aSB/bSB/dSB) と CIM の G22 性能を比較する。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from modules.verify import compute_cut_from_edges

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import build_coupling_matrix, load_graph
from modules.SB import simulate_sb_batch, auto_c0


NUM_TRIALS = 100
NUM_STEPS = 1000  # SB の論文設定
SEED_BASE = 0
KNOWN_BEST = 13359


def main():
    n, k_edges, _, edges = load_graph("input/G22.txt")
    print(f"N={n}, K={k_edges}")

    # SB は J_ij = -1 で c0 正規化が標準
    J_sb = build_coupling_matrix(n, edges, -1.0)
    c0_default = auto_c0(J_sb, n)
    print(f"c0 (auto)         = {c0_default:.4f}")

    seeds = np.arange(SEED_BASE, SEED_BASE + NUM_TRIALS, dtype=np.int64)

    results = {}
    for variant in ["aSB", "bSB", "dSB"]:
        print(f"\n[{variant}] {NUM_TRIALS} trials, {NUM_STEPS} steps...")
        t0 = time.time()
        cuts, _ = simulate_sb_batch(
            n=n, J=J_sb, edges=edges,
            num_steps=NUM_STEPS, num_trials=NUM_TRIALS,
            variant=variant,
            seeds=seeds,
        )
        elapsed = time.time() - t0
        results[variant] = {"cuts": cuts, "time": elapsed}
        print(f"  mean={cuts.mean():.2f}  std={cuts.std():.2f}  "
              f"best={int(cuts.max())}  worst={int(cuts.min())}  "
              f"time={elapsed:.2f}s ({elapsed/NUM_TRIALS*1000:.1f} ms/trial)")

    # 比較用に既存の CIM 結果も読み込み(別途実行済みの v1 ヒストグラム)
    # 直接 CIM を回さず、これまでの 100 trial 平均値だけ参照

    # --- ヒストグラム 3 パネル ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=130)
    colors = {"aSB": "#9467bd", "bSB": "#1f77b4", "dSB": "#d62728"}
    all_cuts = np.concatenate([r["cuts"] for r in results.values()])
    x_min = int(all_cuts.min()) - 30
    x_max = max(int(all_cuts.max()) + 30, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 40)

    for ax, variant in zip(axes, ["aSB", "bSB", "dSB"]):
        cuts = results[variant]["cuts"]
        t = results[variant]["time"]
        ax.hist(cuts, bins=bins, color=colors[variant], alpha=0.75,
                edgecolor="black", linewidth=0.4)
        ax.axvline(cuts.mean(), color="black", linestyle=":", linewidth=1.5,
                   label=f"平均 {cuts.mean():.0f}")
        ax.axvline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
                   label=f"既知最良値 {KNOWN_BEST}")
        ax.set_title(
            f"{variant}\n総時間: {t:.1f}s  "
            f"平均: {cuts.mean():.0f}  最良: {int(cuts.max())}",
            fontsize=11,
        )
        ax.set_xlabel("カット値")
        ax.set_ylabel("頻度")
        ax.set_xlim(x_min, x_max)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"Simulated Bifurcation 3 バリアント — G22 ({NUM_TRIALS} 試行, "
        f"num_steps={NUM_STEPS})",
        fontsize=13,
    )
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    out = "results/v1_sb_three_variants.png"
    i = 1
    while os.path.exists(out):
        i += 1
        out = f"results/v{i}_sb_three_variants.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")

    # --- まとめ表 ---
    print("\n" + "=" * 66)
    print(f"{'method':<8} {'mean':>10} {'best':>8} {'worst':>8} {'std':>8} {'time[s]':>10}")
    print("-" * 66)
    for variant in ["aSB", "bSB", "dSB"]:
        cuts = results[variant]["cuts"]
        t = results[variant]["time"]
        print(f"{variant:<8} {cuts.mean():>10.1f} {int(cuts.max()):>8d} "
              f"{int(cuts.min()):>8d} {cuts.std():>8.2f} {t:>10.2f}")
    print("=" * 66)
    print(f"Reference — Known best:      {KNOWN_BEST}")
    print(f"Reference — Paper CIM mean:  13275, best: 13321")
    print(f"Reference — CIM (Optuna):    mean 13297, best 13340 (held-out)")


if __name__ == "__main__":
    main()
