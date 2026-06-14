"""CAC を 1500 step × 100 trial(開ループの 1500 rounds と同予算)で測り直し、
5方式を comparison_best と同形式で並べ直す(iso-step 公平比較)。

開ループ4方式は既存 cuts.npz をそのまま流用し、CAC のみ num_outer_steps=1500 で
再実行する(seed・パラメータは run_pumpbench_real_cim_g22.py と同一)。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/runners/run_cac_fair_1500step_g22.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters, simulate_cac_batch  # noqa: E402

EXPERIMENT_KIND = "pumpbench_fair_1500step_g22"
KNOWN_BEST = 13359
STEPS = 1500
TRIALS = 100
SRC = ROOT / "results" / "2026-06-14" / "pumpbench_real_cim_g22" / "v1_5cond_100trial_real"

LABELS = {"cac": "CAC(閉ループ)", "gain_linear": "線形利得ランプ\n(最良開ループ)",
          "linear_power": "線形電力ランプ\n(現行)", "sigmoid": "シグモイド",
          "power_early_p05": "べき乗 早上げ p=0.5"}
COLORS = {"cac": "#c0392b", "gain_linear": "#16a085", "linear_power": "#7f8c8d",
          "sigmoid": "#2c5f8a", "power_early_p05": "#d35400"}


def main() -> None:
    import time

    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J_cac = build_coupling_matrix(n, edges, -1.0)
    params = compute_gset_parameters(J_cac, n)
    params.pop("d_0", None); params.pop("d_1", None)
    seeds = np.arange(TRIALS, dtype=np.int64)

    print(f"G22 N={n} E={k_edges}  CAC を {STEPS} step × {TRIALS} trial で再実行")
    t0 = time.time()
    cac_cuts, _ = simulate_cac_batch(
        n=n, J=J_cac, edges=edges, num_outer_steps=STEPS, num_trials=TRIALS,
        seeds=seeds, **params)
    dt = time.time() - t0
    cac_cuts = cac_cuts.astype(float)
    print(f"  CAC@{STEPS}step: mean={cac_cuts.mean():.1f} best={cac_cuts.max():.0f} "
          f"std={cac_cuts.std():.1f} ({cac_cuts.max()/KNOWN_BEST*100:.2f}%) [{dt:.1f}s]")

    # 開ループ4方式は既存結果を流用、CAC は新しい 1500step 結果に差し替え
    src = np.load(SRC / "cuts.npz")
    cuts = {k: src[k].astype(float) for k in src.files if k != "cac"}
    cuts["cac"] = cac_cuts

    # ---- 出力(CLAUDE.md 規約) ----
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in kind_root.iterdir():
        if q.is_dir() and q.name.startswith("v") and q.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(q.name.split("_", 1)[0][1:]))
    out_dir = kind_root / f"v{v + 1}_steps{STEPS}_{TRIALS}trial_isostep"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "cuts.npz", **cuts)

    # ---- comparison_best 同形式 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    order = sorted(cuts.keys(), key=lambda k: -cuts[k].max())
    xs = range(len(order))
    means = [float(cuts[k].mean()) for k in order]
    stds = [float(cuts[k].std()) for k in order]
    bests = [float(cuts[k].max()) for k in order]
    cols = [COLORS[k] for k in order]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(xs, means, yerr=stds, color=cols, alpha=0.45, capsize=4,
           label="平均 ± 標準偏差")
    ax.scatter(xs, bests, marker="*", s=320, color=cols, edgecolor="black",
               linewidth=0.8, zorder=5, label="最良カット(100 trial 中)")
    for x, b in zip(xs, bests):
        ax.annotate(f"{b:.0f}", (x, b), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold")
    ax.axhline(KNOWN_BEST, color="red", ls="--", lw=1.6, label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=9)
    ax.set_ylabel("カット値", fontsize=13)
    ax.set_ylim(min(means) - 70, KNOWN_BEST + 18)
    ax.set_title(f"同予算(1500 step/round)での平均と最良 — G22 ({TRIALS} trial)\n"
                 "棒=平均±std / 星=最良。CAC も開ループと同じ 1500 step に揃えた", fontsize=12)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = out_dir / "comparison_best_1500step.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")

    # ---- サマリ表 ----
    print("\n" + "=" * 64)
    print(f"{'方式':<22}{'平均':>9}{'最良':>8}{'std':>7}{'%既知':>8}")
    for k in order:
        c = cuts[k]
        lab = LABELS[k].replace("\n", "")
        print(f"{lab:<22}{c.mean():>9.1f}{c.max():>8.0f}{c.std():>7.1f}"
              f"{c.max()/KNOWN_BEST*100:>7.2f}%")
    print("=" * 64)


if __name__ == "__main__":
    main()
