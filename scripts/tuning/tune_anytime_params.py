"""tune_anytime_params.py — anytime ベンチ用に CIM / CAC をデータセット別に Optuna 調整。

G22 物理パラメータは他データセット(K2000 密・±1, G55/G70 大規模)に転移しないため、
ここで per-dataset に tune し、results/anytime_tuned_params.json に書き込む。
algo_registry.load_overrides() がそれを読み込む。

目的関数: 固定予算で num_trials バッチを走らせ、統一の重み付きカットで再採点した
平均カット(mean)を最大化(分散も見るので mean が頑健)。

実行:
  python scripts/tuning/tune_anytime_params.py --dataset K2000 --algo CIM --trials 50
  python scripts/tuning/tune_anytime_params.py --dataset K2000 --algo CAC --trials 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import optuna

from modules.CIM import simulate_cim_batch
from modules.CAC import simulate_cac_batch
from scripts.benchmarks.algo_registry import load_context, PARAMS

OVERRIDE_PATH = os.path.join("results", "anytime_tuned_params.json")


def _load_override():
    if os.path.exists(OVERRIDE_PATH):
        with open(OVERRIDE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_override(ov):
    os.makedirs("results", exist_ok=True)
    with open(OVERRIDE_PATH, "w", encoding="utf-8") as f:
        json.dump(ov, f, ensure_ascii=False, indent=1)


def tune_cim(ctx, budget, num_trials, n_trials, seed=0):
    seeds = np.arange(num_trials, dtype=np.int64)

    def objective(tr):
        kappa = tr.suggest_float("kappa", 30.0, 400.0, log=True)
        L = tr.suggest_float("L", 0.005, 0.1, log=True)
        gamma = tr.suggest_float("gamma", 0.5, 60.0, log=True)
        loss_dB = tr.suggest_float("loss_dB", 3.0, 25.0)
        dP = tr.suggest_float("dP_per_round", 1e-6, 5e-3, log=True)
        abs_c = tr.suggest_float("abs_coupling", 1e-4, 0.3, log=True)
        eta = 10.0 ** (-loss_dB / 10.0)
        from modules.CIM import build_coupling_matrix
        J = build_coupling_matrix(ctx.n, ctx.edges, -abs_c, weights=ctx.weights)
        _, signs = simulate_cim_batch(
            ctx.n, J, ctx.edges, int(budget), num_trials,
            kappa, L, gamma, eta, 1.0e9, 6.6e-20, dP, seeds, weights=ctx.weights,
        )
        sc = ctx.score(signs)
        return float(sc.mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    out = dict(
        kappa=bp["kappa"], L=bp["L"], gamma=bp["gamma"],
        eta=10.0 ** (-bp["loss_dB"] / 10.0), bandwidth=1.0e9,
        photon_energy=6.6e-20, dP_per_round=bp["dP_per_round"],
        coupling=-bp["abs_coupling"],
    )
    return out, study.best_value


def tune_cac(ctx, budget, num_trials, n_trials, seed=0):
    seeds = np.arange(num_trials, dtype=np.int64)
    from modules.CAC import compute_gset_parameters
    base = compute_gset_parameters(ctx.J_cac, ctx.n)

    def objective(tr):
        p = tr.suggest_float("p", 0.2, 0.99)
        alpha = tr.suggest_float("alpha", 1.0, 8.0)
        rho = tr.suggest_float("rho", 0.3, 3.0)
        delta = tr.suggest_float("delta", 1e-4, 1e-2, log=True)
        beta0 = tr.suggest_float("beta0_error", 0.01, 1.0, log=True)
        gg = tr.suggest_float("gamma_growth", 1e-3, 0.1, log=True)
        tau = tr.suggest_float("tau", 2.0 * ctx.n, 20.0 * ctx.n)
        _, signs = simulate_cac_batch(
            ctx.n, ctx.J_cac, ctx.edges, int(budget), num_trials,
            p, alpha, rho, delta, beta0, gg, tau,
            6, 4, 2.0 ** -6, 2.0 ** -4, 32.0, seeds,
        )
        sc = ctx.score(signs)
        return float(sc.mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    out = dict(
        p=bp["p"], alpha=bp["alpha"], rho=bp["rho"], delta=bp["delta"],
        beta0_error=bp["beta0_error"], gamma_growth=bp["gamma_growth"],
        tau=bp["tau"], n_x_inner=6, n_e_inner=4,
        dt_x=2.0 ** -6, dt_e=2.0 ** -4, e_max=32.0,
    )
    return out, study.best_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--algo", required=True, choices=["CIM", "CAC"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=8)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = load_context(args.dataset)
    budget = args.budget or (2000 if args.algo == "CIM" else 15000)
    print(f"[tune] {args.dataset}/{args.algo} budget={budget} "
          f"num_trials={args.num_trials} optuna_trials={args.trials}", flush=True)

    if args.algo == "CIM":
        best, val = tune_cim(ctx, budget, args.num_trials, args.trials)
    else:
        best, val = tune_cac(ctx, budget, args.num_trials, args.trials)

    print(f"[tune] best mean weighted cut = {val:.1f} (bks={ctx.bks}, "
          f"gap_mean={ctx.bks - val:.1f})", flush=True)
    print(f"[tune] best params: {json.dumps(best, indent=1)}", flush=True)

    ov = _load_override()
    ov.setdefault(args.dataset, {})[args.algo] = best
    _save_override(ov)
    print(f"[tune] saved → {OVERRIDE_PATH}", flush=True)


if __name__ == "__main__":
    main()
