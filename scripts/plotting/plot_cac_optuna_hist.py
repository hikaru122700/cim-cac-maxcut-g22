"""CAC Optuna チューニング結果のカット値分布(100 trial)を描く。

`scripts/tuning/tune_cac_optuna.py` の run ディレクトリ(results.json)から
既定(GSET)パラメータと Optuna best パラメータを読み、両者を **同一 seed・full budget
(既定 100 trial × 50000 step)で再実行**して 100 試行のカット分布を比較する。

出力(run ディレクトリ直下に追加):
  cuts.npz          : {"baseline": ..., "tuned": ...} の生カット配列
  hist.png          : 参照様式(1 条件 = 1 パネル)の多パネルヒストグラム
  hist_overlay.png  : 2 条件を重ねた分布(平均シフトと BKS 到達が一目で分かる)

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_cac_optuna_hist.py
  .venv/Scripts/python.exe scripts/plotting/plot_cac_optuna_hist.py <run_dir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters, simulate_cac_batch  # noqa: E402

KNOWN_BEST = 13359
SEARCH_PARAMS = ("p", "alpha", "rho", "delta", "beta0_error", "gamma_growth", "tau")
# baseline=既定(GSET) / tuned=Optuna best の表示設定
COND = {
    "baseline": ("既定(GSET, Leleu 2021)", "#7f8c8d"),
    "tuned":    ("Optuna チューニング後", "#27ae60"),
}


def _latest_run_dir() -> Path:
    """results/*/cac_optuna/v*_*/results.json のうち最新(mtime)の run ディレクトリ。"""
    candidates = sorted(
        ROOT.glob("results/*/cac_optuna/v*_*/results.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError("cac_optuna の run ディレクトリが見つかりません")
    return candidates[-1].parent


def _run_cac(n, J, edges, params, fixed, steps, trials, seeds) -> np.ndarray:
    cuts, _ = simulate_cac_batch(
        n=n, J=J, edges=edges, num_outer_steps=steps, num_trials=trials,
        p=params["p"], alpha=params["alpha"], rho=params["rho"],
        delta=params["delta"], beta0_error=params["beta0_error"],
        gamma_growth=params["gamma_growth"], tau=params["tau"],
        n_x_inner=fixed["n_x_inner"], n_e_inner=fixed["n_e_inner"],
        dt_x=fixed["dt_x"], dt_e=fixed["dt_e"], e_max=fixed["e_max"],
        seeds=seeds,
    )
    return cuts.astype(float)


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else _latest_run_dir()
    cfg = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    steps = int(cfg.get("final_steps", 50000))
    trials = int(cfg.get("final_trials", 100))
    graph = cfg.get("graph", "G22.txt")
    baseline_params = {k: cfg["baseline_params"][k] for k in SEARCH_PARAMS}
    tuned_params = {k: cfg["best_params"][k] for k in SEARCH_PARAMS}
    print(f"run_dir : {run_dir}")
    print(f"graph={graph}  budget={trials} trial x {steps} step")

    # ---- グラフ & 固定(離散化)パラメータ ----
    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / graph))
    J = build_coupling_matrix(n, edges, -1.0)
    base = compute_gset_parameters(J, n)
    fixed = {k: base[k] for k in ("n_x_inner", "n_e_inner", "dt_x", "dt_e", "e_max")}
    seeds = np.arange(cfg.get("seed_base", 0), cfg.get("seed_base", 0) + trials, dtype=np.int64)

    # ---- 100 trial 再実行(seed 揃え) ----
    cuts = {}
    for key, params in (("baseline", baseline_params), ("tuned", tuned_params)):
        t = time.time()
        cuts[key] = _run_cac(n, J, edges, params, fixed, steps, trials, seeds)
        c = cuts[key]
        print(f"  {COND[key][0]:<24} mean={c.mean():.1f} best={c.max():.0f} "
              f"std={c.std():.1f} opt={int((c == KNOWN_BEST).sum())}/{trials} [{time.time()-t:.1f}s]")

    np.savez(run_dir / "cuts.npz", **cuts)

    # ---- 描画 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 13

    order = ["baseline", "tuned"]  # 平均昇順(左=既定, 右=チューニング後)
    all_cuts = np.concatenate([cuts[k] for k in order])
    x_min = float(all_cuts.min()) - 20
    x_max = max(float(all_cuts.max()) + 20, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 36)

    # --- Fig1: 多パネル(参照 hist.png 様式) ---
    fig, axes = plt.subplots(1, len(order), figsize=(4.6 * len(order), 4.6), sharex=True)
    for ax, key in zip(axes, order):
        c = cuts[key]
        label, color = COND[key]
        pct = c.max() / KNOWN_BEST * 100
        ax.hist(c, bins=bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axvline(c.mean(), color="black", linestyle=":", linewidth=1.3,
                   label=f"平均 {c.mean():.0f}")
        ax.axvline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3,
                   label=f"既知ベスト {KNOWN_BEST}")
        ax.set_title(f"{label}\n平均:{c.mean():.0f} 最良:{c.max():.0f} "
                     f"({pct:.2f}%) std:{c.std():.1f}", fontsize=10)
        ax.set_xlabel("カット値", fontsize=LABEL_FS)
        ax.set_ylabel("頻度", fontsize=LABEL_FS)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"CAC カット値分布 — G22 ({trials} trial × {steps} step)  "
                 f"既定 vs Optuna チューニング後", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(run_dir / "hist.png", dpi=150)
    plt.close(fig)

    # --- Fig2: 重ね描き(平均シフトと BKS 到達を一望) ---
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for key in order:
        c = cuts[key]
        label, color = COND[key]
        ax.hist(c, bins=bins, color=color, alpha=0.55, edgecolor="black", linewidth=0.4,
                label=f"{label}(平均 {c.mean():.0f} / 最良 {c.max():.0f} / std {c.std():.1f})")
        ax.axvline(c.mean(), color=color, linestyle=":", linewidth=1.8)
    ax.axvline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.6,
               label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xlabel("カット値", fontsize=LABEL_FS)
    ax.set_ylabel("頻度", fontsize=LABEL_FS)
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"CAC: 既定 → Optuna チューニングによる分布シフト — G22 ({trials} trial)\n"
                 f"点線=各平均 / 平均 {cuts['baseline'].mean():.0f} → {cuts['tuned'].mean():.0f} "
                 f"(+{cuts['tuned'].mean()-cuts['baseline'].mean():.0f}), "
                 f"std {cuts['baseline'].std():.1f} → {cuts['tuned'].std():.1f}", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(run_dir / "hist_overlay.png", dpi=150)
    plt.close(fig)

    print(f"saved: {run_dir/'hist.png'}")
    print(f"saved: {run_dir/'hist_overlay.png'}")
    print(f"saved: {run_dir/'cuts.npz'}")


if __name__ == "__main__":
    main()
