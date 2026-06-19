"""Optuna で CAC (Leleu 2021) のアルゴリズムパラメータをチューニングする。

既存の Method B (scripts/tuning/tune_cac.py, 逐次・感度ベース) に対し、本スクリプトは
TPESampler でベイズ最適化的に 7 パラメータを同時探索する。

探索対象 (7):
    p            : 分岐パラメータ (定数, 既定 1 − 400·d₁^-2.5 ≈ 0.78)
    alpha        : 目標振幅² の中心 (既定 3.0)
    rho          : 目標振幅変調の深さ (既定 1.0)
    delta        : ΔH 感度 (既定 2.6/N)
    beta0_error  : 誤差項 rate (既定 3/d₀)
    gamma_growth : β_inj 成長率 (既定 2/N)
    tau          : β_inj リセット窓 [外ループ step 単位] (既定 9N)

固定 (離散化パラメータ, GSET 既定から動かさない):
    n_x_inner=6, n_e_inner=4, dt_x=2⁻⁶, dt_e=2⁻⁴, e_max=32

探索範囲は compute_gset_parameters の既定値を中心(アンカー)に取るので、
G22 以外のインスタンスでも妥当な範囲になる。GSET 既定 config は trial 0 として
enqueue されるため、「論文既定からどれだけ改善したか」が直接読める。

目的関数 (--objective):
    mean : 1 trial = CAC を screen_trials 回走らせた mean_cut を最大化 (既定, TPE 安定)
    max  : 同 max_cut を最大化 (BKS ピーク志向, ばらつき大)

使い方 (プロジェクトルートから):
    .venv/Scripts/python.exe scripts/tuning/tune_cac_optuna.py \
        --optuna-trials 200 --screen-steps 3000 --screen-trials 20

出力 (CLAUDE.md 規約):
    results/optuna_cac_study.db                         # 永続ストレージ (再開可)
    results/<date>/cac_optuna/v{N}_<desc>/best_params.json
    results/<date>/cac_optuna/v{N}_<desc>/results.json
    results/<date>/cac_optuna/v{N}_<desc>/history.png
    results/<date>/cac_optuna/v{N}_<desc>/importance.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import optuna  # noqa: E402

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters, simulate_cac_batch  # noqa: E402

EXPERIMENT_KIND = "cac_optuna"
KNOWN_BEST = 13359           # G22 既知ベスト
PAPER_CAC_MEAN = 13278.8     # 本リポジトリ実測 (pumpbench_real, 100 trial)
PAPER_CAC_BEST = 13358       # 同上 (best of 100)

# 探索対象パラメータ名 (enqueue / suggest で共通利用)
SEARCH_PARAMS = ("p", "alpha", "rho", "delta", "beta0_error", "gamma_growth", "tau")


@dataclass(frozen=True)
class SearchSpace:
    """既定値 (アンカー) から決まる探索範囲。"""

    p: tuple[float, float]
    alpha: tuple[float, float]
    rho: tuple[float, float]
    delta: tuple[float, float]
    beta0_error: tuple[float, float]
    gamma_growth: tuple[float, float]
    tau: tuple[float, float]

    @staticmethod
    def from_defaults(base: dict) -> "SearchSpace":
        """compute_gset_parameters の既定値を中心に対数/線形レンジを張る。

        - p は分岐パラメータで必ず < 1 を保つ必要があるため絶対レンジ [0.30, 0.99]。
        - 他は既定値の定数倍 (log スケール) — 既定が必ずレンジ内に入る。
        """
        return SearchSpace(
            p=(0.30, 0.99),
            alpha=(0.3, 10.0),
            rho=(0.05, 5.0),
            delta=(base["delta"] / 20.0, base["delta"] * 20.0),
            beta0_error=(base["beta0_error"] / 10.0, base["beta0_error"] * 10.0),
            gamma_growth=(base["gamma_growth"] / 20.0, base["gamma_growth"] * 20.0),
            tau=(base["tau"] / 50.0, base["tau"] * 3.0),
        )


def suggest_params(trial: optuna.Trial, space: SearchSpace) -> dict:
    """1 trial 分のパラメータを suggest する。p のみ線形、他は log スケール。"""
    return {
        "p": trial.suggest_float("p", *space.p),
        "alpha": trial.suggest_float("alpha", *space.alpha, log=True),
        "rho": trial.suggest_float("rho", *space.rho, log=True),
        "delta": trial.suggest_float("delta", *space.delta, log=True),
        "beta0_error": trial.suggest_float("beta0_error", *space.beta0_error, log=True),
        "gamma_growth": trial.suggest_float("gamma_growth", *space.gamma_growth, log=True),
        "tau": trial.suggest_float("tau", *space.tau, log=True),
    }


def evaluate_cac(
    *,
    n: int,
    J,
    edges: list[tuple[int, int]],
    params: dict,
    fixed: dict,
    num_outer_steps: int,
    num_trials: int,
    seeds: np.ndarray,
) -> dict:
    """CAC を num_trials 並列実行し、cut 統計を返す。"""
    cuts, _ = simulate_cac_batch(
        n=n, J=J, edges=edges,
        num_outer_steps=num_outer_steps, num_trials=num_trials,
        p=params["p"], alpha=params["alpha"], rho=params["rho"],
        delta=params["delta"], beta0_error=params["beta0_error"],
        gamma_growth=params["gamma_growth"], tau=params["tau"],
        n_x_inner=fixed["n_x_inner"], n_e_inner=fixed["n_e_inner"],
        dt_x=fixed["dt_x"], dt_e=fixed["dt_e"], e_max=fixed["e_max"],
        seeds=seeds,
    )
    cuts = cuts.astype(np.int64)
    return {
        "mean": float(cuts.mean()),
        "max": int(cuts.max()),
        "min": int(cuts.min()),
        "std": float(cuts.std()),
        "num_optimal": int((cuts == KNOWN_BEST).sum()),
    }


def _next_version_dir(kind_root: Path, desc: str) -> Path:
    """results/<date>/<kind>/v{N}_<desc>/ を採番して作る (CLAUDE.md 規約)。"""
    kind_root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in kind_root.iterdir():
        if q.is_dir() and q.name.startswith("v"):
            head = q.name.split("_", 1)[0][1:]
            if head.isdigit():
                v = max(v, int(head))
    out_dir = kind_root / f"v{v + 1}_{desc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _plot_history(study: optuna.Study, best_value: float, args, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    values = np.array(
        [t.value if t.value is not None else np.nan for t in study.trials], dtype=float
    )
    running_best = np.fmax.accumulate(np.where(np.isnan(values), -np.inf, values))
    trial_idx = np.arange(1, len(values) + 1)

    metric_jp = "平均カット" if args.objective == "mean" else "最良カット"
    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    ax.scatter(trial_idx, values, s=10, color="#1f77b4", alpha=0.35,
               label=f"各試行の{metric_jp}")
    ax.plot(trial_idx, running_best, color="#d62728", linewidth=2.2,
            label=f"これまでの最良{metric_jp}")
    ax.axhline(PAPER_CAC_MEAN, color="black", linestyle=":", linewidth=1.3,
               label=f"既定CAC 平均 {PAPER_CAC_MEAN:.0f}")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
               label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xlabel("Optuna 試行番号")
    ax.set_ylabel(f"{metric_jp}（CAC {args.screen_trials} 試行）")
    ax.set_title(
        f"Optuna による CAC パラメータ最適化\n"
        f"({args.optuna_trials} 試行 × CAC {args.screen_trials} 試行 × "
        f"{args.screen_steps} step, 目的={args.objective})  最終 best={best_value:.1f}"
    )
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _plot_importance(study: optuna.Study, out_png: Path) -> bool:
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:  # 試行数不足など
        print(f"  importance plot skipped: {exc}")
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    names = list(importance.keys())
    vals = list(importance.values())
    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    ax.barh(names, vals, color="#1f77b4")
    ax.set_xlabel("パラメータ重要度（fANOVA 推定）")
    ax.set_title("各パラメータの相対重要度（CAC / G22）")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    return True


def main() -> None:
    args = _parse_args()

    print("=" * 64)
    print("CAC Optuna hyperparameter tuner (Leleu 2021, TPESampler)")
    print(f"  objective      : {args.objective}")
    print(f"  optuna trials  : {args.optuna_trials}")
    print(f"  screen budget  : {args.screen_trials} trials x {args.screen_steps} steps")
    print(f"  final budget   : {args.final_trials} trials x {args.final_steps} steps")
    print("=" * 64)

    # ---- グラフ & 既定パラメータ ----
    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / args.graph))
    print(f"Graph: {args.graph}  N={n}  K={k_edges}")
    J = build_coupling_matrix(n, edges, -1.0)

    base = compute_gset_parameters(J, n)
    d_0, d_1 = base.pop("d_0"), base.pop("d_1")
    fixed = {k: base[k] for k in ("n_x_inner", "n_e_inner", "dt_x", "dt_e", "e_max")}
    baseline_params = {k: base[k] for k in SEARCH_PARAMS}
    space = SearchSpace.from_defaults(base)
    print(f"  d_0={d_0:.2f} d_1={d_1:.2f}")
    print("  GSET defaults: " + ", ".join(f"{k}={baseline_params[k]:.4g}" for k in SEARCH_PARAMS))

    screen_seeds = np.arange(args.seed_base, args.seed_base + args.screen_trials, dtype=np.int64)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, space)
        try:
            stats = evaluate_cac(
                n=n, J=J, edges=edges, params=params, fixed=fixed,
                num_outer_steps=args.screen_steps, num_trials=args.screen_trials,
                seeds=screen_seeds,
            )
        except Exception as exc:  # シミュレーション発散など
            print(f"  [trial {trial.number}] sim error: {exc}")
            return 0.0
        trial.set_user_attr("mean_cut", stats["mean"])
        trial.set_user_attr("max_cut", stats["max"])
        trial.set_user_attr("min_cut", stats["min"])
        trial.set_user_attr("std_cut", stats["std"])
        trial.set_user_attr("num_optimal", stats["num_optimal"])
        return stats["mean"] if args.objective == "mean" else float(stats["max"])

    # ---- study ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage_url = f"sqlite:///{(ROOT / 'results' / 'optuna_cac_study.db').as_posix()}"
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name=args.study_name, storage=storage_url,
        load_if_exists=not args.fresh,
    )
    # GSET 既定を trial 0 として評価し、改善幅を可読化
    if len(study.trials) == 0:
        study.enqueue_trial(baseline_params)

    t0 = time.time()

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (trial.number + 1) / elapsed
            print(f"[{trial.number + 1:4d}/{args.optuna_trials}] "
                  f"best({args.objective})={study.best_value:.1f}  "
                  f"({rate:.2f} trial/s, elapsed {elapsed:.1f}s)")

    print(f"\nStarting Optuna ({args.optuna_trials} trials)...")
    study.optimize(objective, n_trials=args.optuna_trials, callbacks=[cb])
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed / args.optuna_trials * 1000:.0f} ms/trial)")

    best = study.best_trial
    best_params = dict(best.params)
    print("=" * 64)
    print(f"best {args.objective}_cut : {best.value:.2f}")
    print(f"  mean={best.user_attrs.get('mean_cut'):.2f}  "
          f"max={best.user_attrs.get('max_cut')}  "
          f"min={best.user_attrs.get('min_cut')}  "
          f"std={best.user_attrs.get('std_cut'):.2f}  "
          f"opt_hits={best.user_attrs.get('num_optimal')}/{args.screen_trials}")
    print("best params:")
    for k in SEARCH_PARAMS:
        print(f"  {k:13s} = {best_params[k]:.6g}   (既定 {baseline_params[k]:.6g})")
    print("=" * 64)

    # ---- full budget で baseline と tuned を再評価 ----
    print(f"\nFinal re-evaluation at full budget "
          f"({args.final_trials} trials x {args.final_steps} steps)...")
    final_seeds = np.arange(args.seed_base, args.seed_base + args.final_trials, dtype=np.int64)
    tf = time.time()
    baseline_final = evaluate_cac(
        n=n, J=J, edges=edges, params=baseline_params, fixed=fixed,
        num_outer_steps=args.final_steps, num_trials=args.final_trials, seeds=final_seeds,
    )
    tuned_final = evaluate_cac(
        n=n, J=J, edges=edges, params=best_params, fixed=fixed,
        num_outer_steps=args.final_steps, num_trials=args.final_trials, seeds=final_seeds,
    )
    print(f"  final eval done in {time.time() - tf:.1f}s")
    print(f"  baseline(GSET): mean={baseline_final['mean']:.2f} max={baseline_final['max']} "
          f"std={baseline_final['std']:.2f} opt={baseline_final['num_optimal']}/{args.final_trials}")
    print(f"  tuned(Optuna) : mean={tuned_final['mean']:.2f} max={tuned_final['max']} "
          f"std={tuned_final['std']:.2f} opt={tuned_final['num_optimal']}/{args.final_trials}")
    print(f"  Δmean={tuned_final['mean'] - baseline_final['mean']:+.2f}  "
          f"Δmax={tuned_final['max'] - baseline_final['max']:+d}  "
          f"Δopt={tuned_final['num_optimal'] - baseline_final['num_optimal']:+d}")

    # ---- 出力 (CLAUDE.md 規約) ----
    desc = f"{args.optuna_trials}trial_{args.objective}_s{args.screen_steps}"
    if args.tag:
        desc += f"_{args.tag}"
    out_dir = _next_version_dir(ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND, desc)

    results = {
        "graph": args.graph, "N": n, "K": k_edges,
        "objective": args.objective,
        "optuna_trials": args.optuna_trials,
        "screen_steps": args.screen_steps, "screen_trials": args.screen_trials,
        "final_steps": args.final_steps, "final_trials": args.final_trials,
        "d_0": d_0, "d_1": d_1,
        "search_space": {k: list(getattr(space, k)) for k in SEARCH_PARAMS},
        "baseline_params": baseline_params,
        "best_params": best_params,
        "best_screen_value": best.value,
        "best_screen_attrs": dict(best.user_attrs),
        "baseline_final": baseline_final,
        "tuned_final": tuned_final,
        "known_best": KNOWN_BEST,
        "elapsed_sec": elapsed,
    }
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "best_params.json").write_text(
        json.dumps({**best_params, **fixed}, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_history(study, best.value, args, out_dir / "history.png")
    _plot_importance(study, out_dir / "importance.png")

    print(f"\nSaved to: {out_dir}")
    print(f"  results.json / best_params.json / history.png / importance.png")
    print(f"Study DB: results/optuna_cac_study.db  (study_name={args.study_name})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAC Optuna hyperparameter tuner")
    p.add_argument("--graph", default="G22.txt", help="input/ 下のグラフファイル名")
    p.add_argument("--objective", choices=["mean", "max"], default="mean",
                   help="最適化する指標 (既定 mean: 安定, max: BKSピーク志向)")
    p.add_argument("--optuna-trials", type=int, default=200, dest="optuna_trials")
    p.add_argument("--screen-steps", type=int, default=3000, dest="screen_steps",
                   help="各 Optuna trial の CAC 外ループ step 数")
    p.add_argument("--screen-trials", type=int, default=20, dest="screen_trials",
                   help="各 Optuna trial の CAC 並列 trial 数")
    p.add_argument("--final-steps", type=int, default=50000, dest="final_steps")
    p.add_argument("--final-trials", type=int, default=100, dest="final_trials")
    p.add_argument("--seed-base", type=int, default=0, dest="seed_base")
    p.add_argument("--log-every", type=int, default=20, dest="log_every")
    p.add_argument("--study-name", default="cac_g22_tuning", dest="study_name")
    p.add_argument("--fresh", action="store_true",
                   help="既存 study をロードせず新規作成する")
    p.add_argument("--tag", default="", help="出力ディレクトリ名に付ける任意サフィックス")
    return p.parse_args()


if __name__ == "__main__":
    main()
