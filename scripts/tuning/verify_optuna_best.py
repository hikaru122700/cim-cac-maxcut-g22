"""Optuna ベストパラメータと論文パラメータを seed 100..199 (held-out) で再評価して比較する。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch


PAPER_PARAMS = dict(
    kappa=130.0, L=0.05, gamma=42.09, loss_dB=11.0,
    bandwidth=1.0e9, photon_energy=1.28e-19,
    dP_per_round=0.05e-3, coupling=-0.03,
)

# v2_optuna_best_params.json から
OPTUNA_PARAMS = dict(
    kappa=253.831, L=0.0278291, gamma=7.21026, loss_dB=10.3693,
    bandwidth=1.64003e9, photon_energy=6.33518e-20,
    dP_per_round=1.55374e-05, coupling=-0.0494281,
)

NUM_ROUNDS = 1500
N_TRIALS = 100
SEED_START = 100  # held-out (Optuna 中は 0..19 を使った)


def run(params: dict, n, edges, seeds):
    eta = 10.0 ** (-params["loss_dB"] / 10.0)
    J = build_coupling_matrix(n, edges, params["coupling"])
    best_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges,
        num_rounds=NUM_ROUNDS, num_trials=len(seeds),
        kappa=params["kappa"], L=params["L"], gamma=params["gamma"], eta=eta,
        bandwidth=params["bandwidth"], photon_energy=params["photon_energy"],
        dP_per_round=params["dP_per_round"],
        seeds=seeds,
    )
    return best_cuts


def main():
    n, k_edges, _, edges = load_graph("input/G22.txt")
    print(f"N={n}, K={k_edges}, num_rounds={NUM_ROUNDS}, n_trials={N_TRIALS}, seeds={SEED_START}..{SEED_START+N_TRIALS-1}")

    seeds = np.arange(SEED_START, SEED_START + N_TRIALS, dtype=np.int64)

    t0 = time.time()
    paper_cuts = run(PAPER_PARAMS, n, edges, seeds)
    paper_t = time.time() - t0

    t0 = time.time()
    optuna_cuts = run(OPTUNA_PARAMS, n, edges, seeds)
    optuna_t = time.time() - t0

    print()
    print(f"{'paper':10s}  mean={paper_cuts.mean():.2f}  std={paper_cuts.std():.2f}  "
          f"best={paper_cuts.max()}  worst={paper_cuts.min()}  median={np.median(paper_cuts):.1f}  "
          f"time={paper_t:.2f}s")
    print(f"{'optuna':10s}  mean={optuna_cuts.mean():.2f}  std={optuna_cuts.std():.2f}  "
          f"best={optuna_cuts.max()}  worst={optuna_cuts.min()}  median={np.median(optuna_cuts):.1f}  "
          f"time={optuna_t:.2f}s")
    diff = optuna_cuts.mean() - paper_cuts.mean()
    print(f"\ndiff (optuna - paper): {diff:+.2f}")

    # Welch t-test (不等分散) — scipy 不要、手書き
    m1, m2 = optuna_cuts.mean(), paper_cuts.mean()
    v1, v2 = optuna_cuts.var(ddof=1), paper_cuts.var(ddof=1)
    n1 = n2 = len(seeds)
    t_stat = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    print(f"Welch t-statistic: {t_stat:.3f}  (|t| > 2 で 95%, > 2.6 で 99% 有意)")

    # --- 比較ヒストグラム ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

    all_cuts = np.concatenate([paper_cuts, optuna_cuts])
    x_min = int(all_cuts.min()) - 10
    x_max = int(all_cuts.max()) + 10
    bins = np.linspace(x_min, x_max, 35)

    ax.hist(paper_cuts, bins=bins, color="#1f77b4", alpha=0.6,
            edgecolor="black", linewidth=0.4,
            label=f"論文パラメータ (mean={paper_cuts.mean():.1f}, best={paper_cuts.max()})")
    ax.hist(optuna_cuts, bins=bins, color="#d62728", alpha=0.6,
            edgecolor="black", linewidth=0.4,
            label=f"Optuna 最適 (mean={optuna_cuts.mean():.1f}, best={optuna_cuts.max()})")

    ax.axvline(paper_cuts.mean(), color="#1f77b4", linestyle=":", linewidth=1.5)
    ax.axvline(optuna_cuts.mean(), color="#d62728", linestyle=":", linewidth=1.5)
    ax.axvline(13359, color="goldenrod", linestyle="--", linewidth=1.3,
               label="既知最良値 13359")

    ax.set_xlabel("best_cut")
    ax.set_ylabel("頻度")
    ax.set_title(
        f"論文 vs Optuna 最適パラメータ: held-out seeds {SEED_START}..{SEED_START+N_TRIALS-1} "
        f"({N_TRIALS} 試行)\n"
        f"差 = {diff:+.2f} (Welch t = {t_stat:.2f})"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    out = "results/v1_optuna_vs_paper_heldout.png"
    i = 1
    while os.path.exists(out):
        i += 1
        out = f"results/v{i}_optuna_vs_paper_heldout.png"
    fig.savefig(out)
    print(f"Saved: {out}")

    # 結果 JSON
    results = {
        "n_trials": N_TRIALS,
        "seed_range": [int(seeds[0]), int(seeds[-1])],
        "num_rounds": NUM_ROUNDS,
        "paper": {
            "params": PAPER_PARAMS,
            "mean": float(paper_cuts.mean()),
            "std": float(paper_cuts.std()),
            "best": int(paper_cuts.max()),
            "worst": int(paper_cuts.min()),
        },
        "optuna": {
            "params": OPTUNA_PARAMS,
            "mean": float(optuna_cuts.mean()),
            "std": float(optuna_cuts.std()),
            "best": int(optuna_cuts.max()),
            "worst": int(optuna_cuts.min()),
        },
        "diff_mean": float(diff),
        "welch_t": float(t_stat),
    }
    out_json = "results/v1_optuna_vs_paper_heldout.json"
    i = 1
    while os.path.exists(out_json):
        i += 1
        out_json = f"results/v{i}_optuna_vs_paper_heldout.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
