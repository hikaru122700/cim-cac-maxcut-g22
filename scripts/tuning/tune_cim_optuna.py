"""Optuna で CIM のパラメータを全数チューニングする。

設定:
- 試行数: 1000 (Optuna trials)
- 1 trial あたり CIM を 20 回走らせて mean best_cut を最大化
- num_rounds = 1500 固定(計算予算を揃えるため)
- それ以外の物理/アルゴリズムパラメータを全部探索
"""

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
import optuna

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch


N_OPTUNA_TRIALS = 1000
N_CIM_TRIALS_PER_OPTUNA_TRIAL = 20
NUM_ROUNDS = 1500
SEED_BASE = 0
KNOWN_BEST = 13359

print("Loading G22...")
N, K_EDGES, _, EDGES = load_graph("input/G22.txt")
print(f"  N={N}, K={K_EDGES}")

SEEDS = np.arange(SEED_BASE, SEED_BASE + N_CIM_TRIALS_PER_OPTUNA_TRIAL, dtype=np.int64)


def objective(trial: optuna.Trial) -> float:
    # 物理 / アルゴリズムパラメータの探索範囲
    kappa = trial.suggest_float("kappa", 30.0, 300.0, log=True)
    L = trial.suggest_float("L", 0.01, 0.20, log=True)
    gamma = trial.suggest_float("gamma", 5.0, 200.0, log=True)
    loss_dB = trial.suggest_float("loss_dB", 3.0, 25.0)
    bandwidth = trial.suggest_float("bandwidth", 1e8, 5e9, log=True)
    photon_energy = trial.suggest_float("photon_energy", 0.5e-19, 5e-19, log=True)
    dP_per_round = trial.suggest_float("dP_per_round", 1e-6, 5e-4, log=True)
    coupling = -trial.suggest_float("abs_coupling", 1e-3, 0.2, log=True)

    eta = 10.0 ** (-loss_dB / 10.0)
    J = build_coupling_matrix(N, EDGES, coupling)

    try:
        best_cuts, _ = simulate_cim_batch(
            n=N, J=J, edges=EDGES,
            num_rounds=NUM_ROUNDS,
            num_trials=N_CIM_TRIALS_PER_OPTUNA_TRIAL,
            kappa=kappa, L=L, gamma=gamma, eta=eta,
            bandwidth=bandwidth, photon_energy=photon_energy,
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


def main():
    os.makedirs("results", exist_ok=True)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=0)
    storage_url = "sqlite:///results/optuna_cim_study.db"
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name="cim_g22_full_tuning",
        storage=storage_url, load_if_exists=True,
    )

    # ベースライン(論文値)を最初に enqueue して比較しやすく
    baseline = dict(
        kappa=130.0, L=0.05, gamma=42.09, loss_dB=11.0,
        bandwidth=1.0e9, photon_energy=1.28e-19,
        dP_per_round=0.05e-3, abs_coupling=0.03,
    )
    study.enqueue_trial(baseline)

    t0 = time.time()
    last_log_time = t0
    log_every = 50  # trials

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        nonlocal last_log_time
        if (trial.number + 1) % log_every == 0:
            now = time.time()
            elapsed = now - t0
            rate = (trial.number + 1) / elapsed
            best = study.best_value
            best_params = study.best_params
            print(
                f"[{trial.number + 1:4d}/{N_OPTUNA_TRIALS}] "
                f"best mean_cut = {best:.2f}  "
                f"({rate:.2f} trial/s, "
                f"elapsed {elapsed:.1f}s)"
            )
            last_log_time = now

    print(f"Starting Optuna: {N_OPTUNA_TRIALS} trials, {N_CIM_TRIALS_PER_OPTUNA_TRIAL} CIM trials each...")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, callbacks=[cb])
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s ({elapsed/N_OPTUNA_TRIALS*1000:.0f} ms/optuna_trial)")

    best = study.best_trial
    print("=" * 60)
    print(f"best mean_cut    : {best.value:.3f}")
    print(f"  std            : {best.user_attrs.get('std_cut'):.3f}")
    print(f"  best of 20     : {best.user_attrs.get('max_cut')}")
    print(f"  worst of 20    : {best.user_attrs.get('min_cut')}")
    print("best params:")
    for k, v in best.params.items():
        print(f"  {k} = {v:.6g}")
    print("=" * 60)

    # 結果保存
    results = {
        "best_value_mean_cut": best.value,
        "best_params": best.params,
        "best_std": best.user_attrs.get("std_cut"),
        "best_max": best.user_attrs.get("max_cut"),
        "best_min": best.user_attrs.get("min_cut"),
        "known_best": KNOWN_BEST,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "n_cim_trials_per_optuna": N_CIM_TRIALS_PER_OPTUNA_TRIAL,
        "num_rounds": NUM_ROUNDS,
        "elapsed_sec": elapsed,
    }
    out_json = "results/v1_optuna_best_params.json"
    i = 1
    while os.path.exists(out_json):
        i += 1
        out_json = f"results/v{i}_optuna_best_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")

    # --- 探索履歴グラフ ---
    values = np.array([t.value if t.value is not None else 0.0 for t in study.trials])
    running_best = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    trial_idx = np.arange(1, len(values) + 1)
    ax.scatter(trial_idx, values, s=8, color="#1f77b4", alpha=0.35,
               label="各試行の mean_cut")
    ax.plot(trial_idx, running_best, color="#d62728", linewidth=2.2,
            label="これまでの最良 mean_cut")
    ax.axhline(13275, color="black", linestyle=":", linewidth=1.3,
               label="論文 Fig.8 平均 13275")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
               label=f"既知最良値 {KNOWN_BEST}")
    ax.set_xlabel("Optuna 試行番号")
    ax.set_ylabel("mean best_cut(CIM 20 試行平均)")
    ax.set_title(
        f"Optuna による CIM パラメータ最適化 "
        f"({N_OPTUNA_TRIALS} 試行 × CIM {N_CIM_TRIALS_PER_OPTUNA_TRIAL} 試行, "
        f"num_rounds={NUM_ROUNDS})\n"
        f"最終 best = {best.value:.2f}"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    out_png = "results/v1_optuna_history.png"
    i = 1
    while os.path.exists(out_png):
        i += 1
        out_png = f"results/v{i}_optuna_history.png"
    fig.savefig(out_png)
    print(f"Saved: {out_png}")

    # --- パラメータ重要度グラフ ---
    try:
        importance = optuna.importance.get_param_importances(study)
        fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=130)
        names = list(importance.keys())
        vals = list(importance.values())
        ax2.barh(names, vals, color="#1f77b4")
        ax2.set_xlabel("パラメータ重要度 (fANOVA 推定)")
        ax2.set_title("各パラメータの相対重要度")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
        ax2.tick_params(direction="in", which="both", top=True, right=True)
        fig2.tight_layout()
        out_imp = "results/v1_optuna_importance.png"
        i = 1
        while os.path.exists(out_imp):
            i += 1
            out_imp = f"results/v{i}_optuna_importance.png"
        fig2.savefig(out_imp)
        print(f"Saved: {out_imp}")
    except Exception as exc:
        print(f"importance plot skipped: {exc}")


if __name__ == "__main__":
    main()
