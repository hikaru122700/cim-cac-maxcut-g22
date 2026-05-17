"""Optuna による CIM パラメータの縮約版チューニング(G22)。

v1 (8 パラメータ全探索) の重要度分析で寄与がほぼ無かった
  - kappa
  - bandwidth
  - photon_energy
を v1 best の値に固定し、残り 5 パラメータのみを探索する。

仮説: 探索空間の次元削減で TPE がより細かい山を見つけやすくなり、
      同じ 1000 trial 予算で v1 best (13307.25) を超える。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


N_OPTUNA_TRIALS: int = int(os.environ.get("N_OPTUNA_TRIALS", 1000))
N_CIM_TRIALS_PER_OPTUNA_TRIAL: int = int(os.environ.get("N_CIM_TRIALS", 20))
NUM_ROUNDS: int = int(os.environ.get("NUM_ROUNDS", 1500))
SEED_BASE: int = 0
KNOWN_BEST: int = 13359

# --- 固定値(論文値: 物理/装置パラメータは文献値のまま) ---
# 論文の物理セットアップを保持した上でアルゴリズム側を最適化する、という建付け。
FIXED_KAPPA: float = 130.0
FIXED_BANDWIDTH: float = 1.0e9
FIXED_PHOTON_ENERGY: float = 1.28e-19

# --- warm start 用(論文値の 5 パラ) ---
# 固定パラメータと整合させた論文ベースラインを最初の trial として評価する。
PAPER_WARM_START: dict[str, float] = {
    "L": 0.05,
    "gamma": 42.09,
    "loss_dB": 11.0,
    "dP_per_round": 5.0e-5,
    "abs_coupling": 0.03,
}

OUT_DIR: Path = Path(os.environ.get("OUT_DIR", "results/2026-05-17"))
TAG: str = os.environ.get("TAG", "reduced")


print("Loading G22...")
N, K_EDGES, _, EDGES = load_graph("input/G22.txt")
print(f"  N={N}, K={K_EDGES}")
print(
    f"  Fixed: kappa={FIXED_KAPPA:.4g}, bandwidth={FIXED_BANDWIDTH:.4g}, "
    f"photon_energy={FIXED_PHOTON_ENERGY:.4g}"
)
print(f"  Searching: L, gamma, loss_dB, dP_per_round, abs_coupling")

SEEDS = np.arange(SEED_BASE, SEED_BASE + N_CIM_TRIALS_PER_OPTUNA_TRIAL, dtype=np.int64)


def objective(trial: optuna.Trial) -> float:
    L = trial.suggest_float("L", 0.01, 0.20, log=True)
    gamma = trial.suggest_float("gamma", 5.0, 200.0, log=True)
    loss_dB = trial.suggest_float("loss_dB", 3.0, 25.0)
    dP_per_round = trial.suggest_float("dP_per_round", 1e-6, 5e-4, log=True)
    coupling = -trial.suggest_float("abs_coupling", 1e-3, 0.2, log=True)

    eta = 10.0 ** (-loss_dB / 10.0)
    J = build_coupling_matrix(N, EDGES, coupling)

    try:
        best_cuts, _ = simulate_cim_batch(
            n=N,
            J=J,
            edges=EDGES,
            num_rounds=NUM_ROUNDS,
            num_trials=N_CIM_TRIALS_PER_OPTUNA_TRIAL,
            kappa=FIXED_KAPPA,
            L=L,
            gamma=gamma,
            eta=eta,
            bandwidth=FIXED_BANDWIDTH,
            photon_energy=FIXED_PHOTON_ENERGY,
            dP_per_round=dP_per_round,
            seeds=SEEDS,
        )
    except Exception as exc:
        print(f"  [trial {trial.number}] sim error: {exc}")
        return 0.0

    mean_cut = float(np.mean(best_cuts))
    trial.set_user_attr("std_cut", float(np.std(best_cuts)))
    trial.set_user_attr("max_cut", int(np.max(best_cuts)))
    trial.set_user_attr("min_cut", int(np.min(best_cuts)))
    return mean_cut


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=0)
    storage_url = "sqlite:///results/optuna_cim_study.db"
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"cim_g22_{TAG}_nr{NUM_ROUNDS}",
        storage=storage_url,
        load_if_exists=True,
    )

    # warm start: v1 best を最初に enqueue
    study.enqueue_trial(V1_BEST_PARAMS)

    t0 = time.time()
    log_every = 50

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % log_every == 0:
            now = time.time()
            elapsed = now - t0
            rate = (trial.number + 1) / elapsed
            print(
                f"[{trial.number + 1:4d}/{N_OPTUNA_TRIALS}] "
                f"best mean_cut = {study.best_value:.2f}  "
                f"({rate:.2f} trial/s, elapsed {elapsed:.1f}s)"
            )

    print(
        f"Starting Optuna (reduced): {N_OPTUNA_TRIALS} trials, "
        f"{N_CIM_TRIALS_PER_OPTUNA_TRIAL} CIM trials each..."
    )
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[cb])
    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f} s "
        f"({elapsed/N_OPTUNA_TRIALS*1000:.0f} ms/optuna_trial)"
    )

    best = study.best_trial
    print("=" * 60)
    print(f"best mean_cut    : {best.value:.3f}")
    print(f"  std            : {best.user_attrs.get('std_cut'):.3f}")
    print(f"  best of 20     : {best.user_attrs.get('max_cut')}")
    print(f"  worst of 20    : {best.user_attrs.get('min_cut')}")
    print("best params (searched):")
    for k, v in best.params.items():
        print(f"  {k} = {v:.6g}")
    print("fixed params:")
    print(f"  kappa = {FIXED_KAPPA:.6g}")
    print(f"  bandwidth = {FIXED_BANDWIDTH:.6g}")
    print(f"  photon_energy = {FIXED_PHOTON_ENERGY:.6g}")
    print("=" * 60)

    results = {
        "best_value_mean_cut": best.value,
        "best_params": best.params,
        "fixed_params": {
            "kappa": FIXED_KAPPA,
            "bandwidth": FIXED_BANDWIDTH,
            "photon_energy": FIXED_PHOTON_ENERGY,
        },
        "best_std": best.user_attrs.get("std_cut"),
        "best_max": best.user_attrs.get("max_cut"),
        "best_min": best.user_attrs.get("min_cut"),
        "known_best": KNOWN_BEST,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_cim_trials_per_optuna": N_CIM_TRIALS_PER_OPTUNA_TRIAL,
        "num_rounds": NUM_ROUNDS,
        "elapsed_sec": elapsed,
        "tag": TAG,
    }
    out_json = OUT_DIR / f"{TAG}_optuna_best_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # --- 探索履歴 ---
    values = np.array(
        [t.value if t.value is not None else 0.0 for t in study.trials]
    )
    running_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    trial_idx = np.arange(1, len(values) + 1)
    ax.scatter(
        trial_idx, values, s=8, color="#1f77b4", alpha=0.35,
        label="各試行の mean_cut",
    )
    ax.plot(
        trial_idx, running_best, color="#d62728", linewidth=2.2,
        label="これまでの最良 mean_cut",
    )
    ax.axhline(13275, color="black", linestyle=":", linewidth=1.3,
               label="論文 Fig.8 平均 13275")
    ax.axhline(13307.25, color="green", linestyle="-.", linewidth=1.3,
               label="v1 (8 パラ全探索) best 13307.25")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
               label=f"既知最良値 {KNOWN_BEST}")
    ax.set_xlabel("Optuna 試行番号")
    ax.set_ylabel("mean best_cut (CIM 20 試行平均)")
    ax.set_title(
        f"Optuna 縮約版 (5 パラメータ探索, 3 パラメータ固定)\n"
        f"{N_OPTUNA_TRIALS} 試行 × CIM {N_CIM_TRIALS_PER_OPTUNA_TRIAL} 試行 "
        f"/ num_rounds={NUM_ROUNDS} / 最終 best = {best.value:.2f}"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    out_png = OUT_DIR / f"{TAG}_optuna_history.png"
    fig.savefig(out_png)
    print(f"Saved: {out_png}")

    # --- 重要度 (残り 5 パラの相対) ---
    try:
        importance = optuna.importance.get_param_importances(study)
        fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=130)
        names = list(importance.keys())
        vals = list(importance.values())
        ax2.barh(names, vals, color="#1f77b4")
        ax2.set_xlabel("パラメータ重要度 (fANOVA 推定)")
        ax2.set_title("縮約版: 探索 5 パラメータの相対重要度")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
        ax2.tick_params(direction="in", which="both", top=True, right=True)
        fig2.tight_layout()
        out_imp = OUT_DIR / f"{TAG}_optuna_importance.png"
        fig2.savefig(out_imp)
        print(f"Saved: {out_imp}")
    except Exception as exc:
        print(f"importance plot skipped: {exc}")


if __name__ == "__main__":
    main()
