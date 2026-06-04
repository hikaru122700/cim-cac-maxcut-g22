"""CIM-PT 比較 3 手法の Optuna パラメータチューニング (G22)。

CIM+PT が以前の比較で最下位だったのは「パラメータ未調整」が原因かもしれない、
という仮説を検証するため、対照も含む 3 手法すべてを Optuna で独立に最適化する。

各手法とも **5 変数のみ** を探索する:
  共通 4 変数  : L, gamma, loss_dB, abs_coupling   (物理/結合)
  手法固有 1 変数:
    ramp   (ランプ CIM, baseline) : dP_per_round   … ランプ速度
    noswap (CIM-3 固定 / swap 無) : pump_spread Δ  … pump_mults=[1-Δ, 1, 1+Δ]
    cimpt  (CIM + PT)            : pump_spread Δ  … β は swap 無 run から自動較正 (κ_target=1)

固定値 (論文の装置パラメータ): kappa, bandwidth, photon_energy。
PT 系の swap_interval=10, kappa_target=1.0, ポンプ中段=1.0×P_th も固定。

目的関数: G22 で CIM を N_CIM_TRIALS 回まわした best_cut の平均 (maximize)。
各手法 N_OPTUNA_TRIALS=300 試行。出力は results 規約に従い
  results/<date>/cim_pt_optuna/v{N}_<desc>/<method>/  に保存する。

実行 (プロジェクトルートから):
  uv run python scripts/tuning/tune_cim_pt_optuna.py
  uv run python scripts/tuning/tune_cim_pt_optuna.py --methods ramp        # 一手法だけ
  uv run python scripts/tuning/tune_cim_pt_optuna.py --n-optuna 20 --num-rounds 300 --n-cim 8 --tag smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch

# --- 日付始まりのファイル名なので importlib でロード ---
_PT_PATH = ROOT / "modules" / "2026-05-29_CIM_PT.py"
_spec = importlib.util.spec_from_file_location("cim_pt_mod", _PT_PATH)
cim_pt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cim_pt)
simulate_cim_pt_batch = cim_pt.simulate_cim_pt_batch
compute_threshold_pump = cim_pt.compute_threshold_pump
calibrate_betas = cim_pt.calibrate_betas

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

EXPERIMENT_KIND = "cim_pt_optuna"
KNOWN_BEST = 13359
# 以前の (未/簡易調整) 比較での各手法 平均カット (2026-06-02 v2, 6/8 発表値) を参照線に
PREV_MEAN = {"ramp": 13276.0, "noswap": 13193.0, "cimpt": 13156.0}
METHOD_LABEL = {"ramp": "ランプCIM", "noswap": "CIM-3固定(swap無)", "cimpt": "CIM+PT"}

# 固定パラメータ (論文の装置値)
FIXED_KAPPA = 130.0
FIXED_BANDWIDTH = 1.0e9
FIXED_PHOTON_ENERGY = 1.28e-19
# PT 系の固定運用パラメータ
SWAP_INTERVAL = 10
KAPPA_TARGET = 1.0
SAMPLE_INTERVAL = 25  # cimpt の β 較正用に軌跡を記録する間隔
N_CALIB_TRIALS = 8    # cimpt の β 較正 run の CIM 試行数(目的評価より粗くてよい)

# 共通 4 変数 + 手法固有 1 変数の warm start (論文ベース)
WARM_COMMON = {"L": 0.05, "gamma": 42.09, "loss_dB": 11.0, "abs_coupling": 0.03}
WARM_EXTRA = {"ramp": {"dP_per_round": 5.0e-5},
              "noswap": {"pump_spread": 0.5},
              "cimpt": {"pump_spread": 0.5}}


def suggest_common(trial: optuna.Trial) -> dict:
    """3 手法共通の 4 物理/結合変数を提案する。"""
    L = trial.suggest_float("L", 0.01, 0.20, log=True)
    gamma = trial.suggest_float("gamma", 5.0, 200.0, log=True)
    loss_dB = trial.suggest_float("loss_dB", 3.0, 25.0)
    abs_coupling = trial.suggest_float("abs_coupling", 1e-3, 0.2, log=True)
    return {"L": L, "gamma": gamma, "loss_dB": loss_dB, "abs_coupling": abs_coupling,
            "eta": 10.0 ** (-loss_dB / 10.0), "coupling": -abs_coupling}


def _record(trial: optuna.Trial, cuts: np.ndarray) -> float:
    trial.set_user_attr("std_cut", float(np.std(cuts)))
    trial.set_user_attr("max_cut", int(np.max(cuts)))
    trial.set_user_attr("min_cut", int(np.min(cuts)))
    return float(np.mean(cuts))


def make_objective(method: str, N: int, EDGES, NUM_ROUNDS: int, SEEDS: np.ndarray):
    phys = dict(kappa=FIXED_KAPPA, bandwidth=FIXED_BANDWIDTH,
                photon_energy=FIXED_PHOTON_ENERGY)

    def objective(trial: optuna.Trial) -> float:
        c = suggest_common(trial)
        J = build_coupling_matrix(N, EDGES, c["coupling"])
        try:
            if method == "ramp":
                dP = trial.suggest_float("dP_per_round", 1e-6, 5e-4, log=True)
                cuts, _ = simulate_cim_batch(
                    n=N, J=J, edges=EDGES, num_rounds=NUM_ROUNDS, num_trials=N,
                    L=c["L"], gamma=c["gamma"], eta=c["eta"], dP_per_round=dP,
                    seeds=SEEDS, **phys)
                return _record(trial, cuts)

            # --- PT 系 (noswap / cimpt) は pump_spread Δ を持つ ---
            d = trial.suggest_float("pump_spread", 0.1, 0.9)
            p_th = compute_threshold_pump(FIXED_KAPPA, c["L"], c["eta"])
            pump_levels = np.sort(np.array([(1.0 - d) * p_th, p_th, (1.0 + d) * p_th]))
            pt_kwargs = dict(
                n=N, J=J, edges=EDGES, num_rounds=NUM_ROUNDS, num_trials=N,
                L=c["L"], gamma=c["gamma"], eta=c["eta"], seeds=SEEDS,
                pump_levels=pump_levels, swap_interval=SWAP_INTERVAL, **phys)

            if method == "noswap":
                res = simulate_cim_pt_batch(do_swap=False,
                                            sample_interval=NUM_ROUNDS, **pt_kwargs)
                return _record(trial, res["best_cuts"])

            # cimpt: swap 無 run で β を較正 → swap 有を評価。
            # β 較正は定常カット差(平均)だけ要るので、較正 run は CIM 試行数を
            # 減らして高速化する(最適値そのものは不要なので統計が粗くても可)。
            calib_kwargs = dict(pt_kwargs)
            n_calib = min(pt_kwargs["num_trials"], N_CALIB_TRIALS)
            calib_kwargs["num_trials"] = n_calib
            calib_kwargs["seeds"] = pt_kwargs["seeds"][:n_calib]
            res0 = simulate_cim_pt_batch(do_swap=False,
                                         sample_interval=SAMPLE_INTERVAL, **calib_kwargs)
            sr = res0["sample_rounds"]
            tail = max(1, sr.size // 3)
            cut_tail = res0["traj_cut"][:, -tail:, :].mean(axis=(0, 1))
            betas = calibrate_betas(cut_tail, kappa_target=KAPPA_TARGET)
            res = simulate_cim_pt_batch(do_swap=True, betas=betas,
                                        sample_interval=SAMPLE_INTERVAL, **pt_kwargs)
            trial.set_user_attr("swap_rate", [float(x) for x in res["swap_rate"]])
            trial.set_user_attr("betas", [float(x) for x in betas])
            return _record(trial, res["best_cuts"])
        except Exception as exc:  # noqa: BLE001
            print(f"  [trial {trial.number}] sim error: {exc}")
            return 0.0

    return objective


def run_study(method: str, N: int, EDGES, NUM_ROUNDS: int, N_CIM: int,
              N_OPTUNA: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    SEEDS = np.arange(0, N_CIM, dtype=np.int64)
    objective = make_objective(method, N, EDGES, NUM_ROUNDS, SEEDS)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(
        direction="maximize", sampler=sampler,
        study_name=f"{method}_g22_nr{NUM_ROUNDS}",
        storage=f"sqlite:///{(out_dir / 'optuna.db').as_posix()}",
        load_if_exists=True)
    study.enqueue_trial({**WARM_COMMON, **WARM_EXTRA[method]})

    t0 = time.time()

    def cb(st: optuna.Study, tr: optuna.trial.FrozenTrial) -> None:
        if (tr.number + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  [{method}] {tr.number + 1:4d}/{N_OPTUNA}  "
                  f"best mean_cut={st.best_value:.2f}  "
                  f"({(tr.number + 1) / el:.2f} trial/s, {el:.0f}s)")

    print(f"\n=== [{method}] {METHOD_LABEL[method]} : Optuna {N_OPTUNA} trials "
          f"(CIM {N_CIM} trial × rounds {NUM_ROUNDS}) ===")
    study.optimize(objective, n_trials=N_OPTUNA, callbacks=[cb])
    elapsed = time.time() - t0

    best = study.best_trial
    print(f"  done {elapsed:.1f}s  best mean_cut={best.value:.3f}  "
          f"(std={best.user_attrs.get('std_cut'):.2f}, "
          f"max={best.user_attrs.get('max_cut')})  prev={PREV_MEAN[method]:.0f}")
    print(f"  best params: " +
          ", ".join(f"{k}={v:.5g}" for k, v in best.params.items()))

    result = {
        "method": method, "label": METHOD_LABEL[method],
        "best_value_mean_cut": best.value,
        "best_params": best.params,
        "fixed_params": {"kappa": FIXED_KAPPA, "bandwidth": FIXED_BANDWIDTH,
                         "photon_energy": FIXED_PHOTON_ENERGY,
                         "swap_interval": SWAP_INTERVAL,
                         "kappa_target": KAPPA_TARGET},
        "best_std": best.user_attrs.get("std_cut"),
        "best_max": best.user_attrs.get("max_cut"),
        "best_min": best.user_attrs.get("min_cut"),
        "best_swap_rate": best.user_attrs.get("swap_rate"),
        "best_betas": best.user_attrs.get("betas"),
        "prev_mean_untuned": PREV_MEAN[method],
        "known_best": KNOWN_BEST,
        "n_optuna_trials": N_OPTUNA, "n_cim_trials": N_CIM,
        "num_rounds": NUM_ROUNDS, "elapsed_sec": elapsed,
    }
    with open(out_dir / "best_params.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # --- 探索履歴 ---
    vals = np.array([t.value if t.value is not None else 0.0 for t in study.trials])
    run_best = np.maximum.accumulate(vals)
    idx = np.arange(1, len(vals) + 1)
    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    ax.scatter(idx, vals, s=8, color="#1f77b4", alpha=0.35, label="各試行 mean_cut")
    ax.plot(idx, run_best, color="#d62728", linewidth=2.2, label="これまでの最良")
    ax.axhline(PREV_MEAN[method], color="black", linestyle=":", linewidth=1.4,
               label=f"調整前 平均 {PREV_MEAN[method]:.0f}")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
               label=f"既知最良 {KNOWN_BEST}")
    ax.set_xlabel("Optuna 試行番号")
    ax.set_ylabel(f"mean best_cut (CIM {N_CIM} 試行平均)")
    ax.set_title(f"{METHOD_LABEL[method]} — Optuna 5変数 {N_OPTUNA}試行\n"
                 f"最終 best={best.value:.1f} (調整前 {PREV_MEAN[method]:.0f})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out_dir / "history.png")
    plt.close(fig)

    # --- 重要度 ---
    try:
        imp = optuna.importance.get_param_importances(study)
        fig2, ax2 = plt.subplots(figsize=(9, 5), dpi=130)
        ax2.barh(list(imp.keys()), list(imp.values()), color="#1f77b4")
        ax2.set_xlabel("パラメータ重要度 (fANOVA 推定)")
        ax2.set_title(f"{METHOD_LABEL[method]}: 5変数の相対重要度")
        ax2.invert_yaxis()
        ax2.grid(axis="x", alpha=0.3)
        ax2.tick_params(direction="in", which="both", top=True, right=True)
        fig2.tight_layout()
        fig2.savefig(out_dir / "importance.png")
        plt.close(fig2)
    except Exception as exc:  # noqa: BLE001
        print(f"  importance skipped: {exc}")

    return result


def next_version(kind_root: Path) -> int:
    max_v = 0
    if kind_root.exists():
        for p in kind_root.iterdir():
            if p.is_dir() and p.name.startswith("v"):
                head = p.name.split("_", 1)[0]
                if head[1:].isdigit():
                    max_v = max(max_v, int(head[1:]))
    return max_v + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="CIM-PT 3手法の Optuna チューニング")
    ap.add_argument("--graph", default="input/G22.txt")
    ap.add_argument("--methods", nargs="+", default=["ramp", "noswap", "cimpt"],
                    choices=["ramp", "noswap", "cimpt"])
    ap.add_argument("--n-optuna", type=int, default=300)
    ap.add_argument("--n-cim", type=int, default=20)
    ap.add_argument("--num-rounds", type=int, default=1500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print(f"Loading {args.graph} ...")
    N, K_EDGES, _, EDGES = load_graph(args.graph)
    print(f"  N={N} K={K_EDGES}")

    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    v = next_version(kind_root)
    desc = [Path(args.graph).stem.lower(), f"{args.n_optuna}trials", "5var"]
    if args.tag:
        desc.append(args.tag)
    run_dir = kind_root / f"v{v}_{'_'.join(desc)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output] {run_dir}")

    all_results = {}
    t_all = time.time()
    for method in args.methods:
        all_results[method] = run_study(
            method, N, EDGES, args.num_rounds, args.n_cim, args.n_optuna,
            run_dir / method)

    # --- 統合サマリ ---
    summary = {
        "graph": Path(args.graph).stem, "n": N, "k_edges": K_EDGES,
        "n_optuna_trials": args.n_optuna, "n_cim_trials": args.n_cim,
        "num_rounds": args.num_rounds, "known_best": KNOWN_BEST,
        "elapsed_sec_total": time.time() - t_all,
        "methods": {m: {"best_mean_cut": r["best_value_mean_cut"],
                        "prev_mean": r["prev_mean_untuned"],
                        "gain": r["best_value_mean_cut"] - r["prev_mean_untuned"],
                        "best_params": r["best_params"]}
                    for m, r in all_results.items()},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 72)
    print(f"{'method':<22}{'調整前平均':>12}{'調整後best':>12}{'改善':>10}")
    print("-" * 72)
    for m in args.methods:
        r = all_results[m]
        print(f"{METHOD_LABEL[m]:<22}{r['prev_mean_untuned']:>12.0f}"
              f"{r['best_value_mean_cut']:>12.1f}"
              f"{r['best_value_mean_cut'] - r['prev_mean_untuned']:>+10.1f}")
    print("=" * 72)
    print(f"[saved] {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
