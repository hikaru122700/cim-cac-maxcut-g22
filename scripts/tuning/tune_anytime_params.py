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
from modules.SA import simulate_sa_batch
from modules.SB import simulate_sb_batch
from modules.PT_ICM import simulate_pticm_batch
from modules.GA import simulate_ga_batch
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


def tune_sa(ctx, budget, num_trials, n_trials, seed=0):
    """SA: 指数冷却の開始/終了温度を調整。"""
    seeds = np.arange(num_trials, dtype=np.int64)

    def objective(tr):
        t_start = tr.suggest_float("t_start", 0.5, 6.0)
        t_end = tr.suggest_float("t_end", 1e-4, 0.1, log=True)
        _, signs = simulate_sa_batch(
            ctx.n, ctx.edges, ctx.weights, int(budget), num_trials,
            t_start=t_start, t_end=t_end, seeds=seeds,
        )
        return float(ctx.score(signs).mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return dict(t_start=bp["t_start"], t_end=bp["t_end"]), study.best_value


def tune_sb(ctx, budget, num_trials, n_trials, seed=0):
    """SB: variant(bSB/dSB)・時間刻み dt・初期振幅係数 a0 を調整。"""
    seeds = np.arange(num_trials, dtype=np.int64)

    def objective(tr):
        variant = tr.suggest_categorical("variant", ["bSB", "dSB"])
        dt = tr.suggest_float("dt", 0.1, 1.0)
        a0 = tr.suggest_float("a0", 0.5, 2.0)
        _, signs = simulate_sb_batch(
            ctx.n, ctx.J_sb, ctx.edges, int(budget), num_trials,
            variant=variant, a0=a0, dt=dt, weights=ctx.weights, seeds=seeds,
        )
        return float(ctx.score(signs).mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return dict(variant=bp["variant"], dt=bp["dt"], a0=bp["a0"]), study.best_value


def tune_pt(ctx, budget, num_trials, n_trials, seed=0):
    """PT-ICM: 温度ラダー(本数・範囲)と swap/ICM 間隔を調整。"""
    seeds = np.arange(num_trials, dtype=np.int64)

    def objective(tr):
        num_temps = tr.suggest_int("num_temps", 8, 28)
        t_min = tr.suggest_float("t_min", 0.02, 0.4, log=True)
        t_max = tr.suggest_float("t_max", 1.5, 5.0)
        swap_interval = tr.suggest_int("swap_interval", 1, 4)
        icm_interval = tr.suggest_int("icm_interval", 1, 10)
        _, signs, _ = simulate_pticm_batch(
            ctx.n, ctx.edges, ctx.weights, num_trials,
            num_sweeps=int(budget), num_temps=num_temps,
            t_min=t_min, t_max=t_max, swap_interval=swap_interval,
            icm_interval=icm_interval, seeds=seeds,
        )
        return float(ctx.score(signs).mean())

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return dict(num_temps=bp["num_temps"], t_min=bp["t_min"], t_max=bp["t_max"],
                swap_interval=bp["swap_interval"],
                icm_interval=bp["icm_interval"]), study.best_value


def tune_ga(ctx, budget, num_trials, n_trials, seed=0):
    """GA(memetic): 集団サイズ・TS 反復・摂動閾値 cr・tenure 係数・品質重み β を調整。

    budget は世代数(max_generations)。各 trial は num_trials バッチで評価。
    """
    seeds = np.arange(num_trials, dtype=np.int64)

    def objective(tr):
        pop_size = tr.suggest_int("pop_size", 6, 20)
        ts_iters = tr.suggest_int("ts_iters", 5000, 60000, log=True)
        cr = tr.suggest_int("cr", 500, 8000, log=True)
        alpha_tenure = tr.suggest_int("alpha_tenure", 5, 40)
        beta_quality = tr.suggest_float("beta_quality", 0.3, 0.9)
        cuts, _ = simulate_ga_batch(
            ctx.n, ctx.edges, ctx.weights, num_trials,
            pop_size=pop_size, max_generations=int(budget), ts_iters=ts_iters,
            cr=cr, alpha_tenure=alpha_tenure, beta_quality=beta_quality,
            seeds=seeds,
        )
        return float(np.mean(cuts))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    return dict(pop_size=bp["pop_size"], ts_iters=bp["ts_iters"], cr=bp["cr"],
                alpha_tenure=bp["alpha_tenure"],
                beta_quality=bp["beta_quality"]), study.best_value


# algo -> (tune 関数, 既定の調整時 budget)
_TUNERS = {
    "CIM": (tune_cim, 2000),
    "CAC": (tune_cac, 15000),
    "SA":  (tune_sa, 1_000_000),
    "SB":  (tune_sb, 2500),
    "PT":  (tune_pt, 600),
    "GA":  (tune_ga, 20),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--algo", required=True, choices=list(_TUNERS.keys()))
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--num-trials", type=int, default=8)
    args = ap.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ctx = load_context(args.dataset)
    tune_fn, default_budget = _TUNERS[args.algo]
    budget = args.budget or default_budget
    print(f"[tune] {args.dataset}/{args.algo} budget={budget} "
          f"num_trials={args.num_trials} optuna_trials={args.trials}", flush=True)

    best, val = tune_fn(ctx, budget, args.num_trials, args.trials)

    print(f"[tune] best mean weighted cut = {val:.1f} (bks={ctx.bks}, "
          f"gap_mean={ctx.bks - val:.1f})", flush=True)
    print(f"[tune] best params: {json.dumps(best, indent=1)}", flush=True)

    ov = _load_override()
    ov.setdefault(args.dataset, {})[args.algo] = best
    _save_override(ov)
    print(f"[tune] saved → {OVERRIDE_PATH}", flush=True)


if __name__ == "__main__":
    main()
