"""CIM の num_rounds スイープを Optuna 5 パラ縮約版で個別チューニング + 振幅可視化。

設計
----
* 探索パラ: L, gamma, loss_dB, dP_per_round, abs_coupling (5 個)
* 固定パラ: kappa=130, bandwidth=1e9, photon_energy=1.28e-19 (論文値)
* 各 Optuna trial では `simulate_cim_batch` を `num_trials=N_CIM_TRIALS_PER_OPTUNA`
  で呼び出し、N 個のシードを Numba prange で並列に消化する(=シードレベル並列化)
* 4 条件 num_rounds ∈ {30, 300, 1500, 10000} を独立にチューニング
* 各条件のベストパラで `simulate_cim_with_trajectory` を走らせ、
  - 振幅 c_i(round) の軌跡(N=2000 本の薄い線 + 平均±std 帯)
  - cut(round) 曲線
  を保存

使い方:
    python scripts/tuning/tune_cim_optuna_rounds_sweep.py
    python scripts/tuning/tune_cim_optuna_rounds_sweep.py --n-optuna-trials 500
    python scripts/tuning/tune_cim_optuna_rounds_sweep.py --rounds 30 300 1500 10000

出力 (新規約: `results/<today>/cim_optuna_rounds_sweep/v{N}_<desc>/`):
    summary.json
    history.png         各条件の Optuna 探索履歴 (2x2)
    amplitudes.png      各条件の振幅軌跡 (2x2)
    cut_curves.png      各条件の cut vs round
    best_vs_rounds.png  最終 mean_cut vs num_rounds
    trajectories.npz    軌跡生データ

<desc> は CLI 引数から自動生成 (例: 4cond_300trial)。--tag で追加サフィックス指定可。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

from modules.CIM import (
    build_coupling_matrix,
    load_graph,
    simulate_cim_batch,
    simulate_cim_with_trajectory,
)


EXPERIMENT_KIND = "cim_optuna_rounds_sweep"


# ---- 共通定数(reduced 版と整合) ----
KNOWN_BEST: int = 13359
FIXED_KAPPA: float = 130.0
FIXED_BANDWIDTH: float = 1.0e9
FIXED_PHOTON_ENERGY: float = 1.28e-19

PAPER_WARM_START: dict[str, float] = {
    "L": 0.05,
    "gamma": 42.09,
    "loss_dB": 11.0,
    "dP_per_round": 5.0e-5,
    "abs_coupling": 0.03,
}


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def apply_ticks_inward(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)


def get_kind_root() -> Path:
    """results/<today>/<EXPERIMENT_KIND>/ を返す(必要なら作成)。"""
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    """kind_root 配下の v{N}_* サブディレクトリを見て次の N を返す。"""
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def build_description(args) -> str:
    """CLI 引数から run 内容を表す簡潔な説明文字列を生成する。"""
    parts = [f"{len(args.rounds)}cond", f"{args.n_optuna_trials}trial"]
    if args.n_cim_trials != 20:
        parts.append(f"cim{args.n_cim_trials}seed")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def make_objective(n, edges, num_rounds, seeds):
    """num_rounds 別の objective を生成。CIM batch は seeds 並列で実行される。"""
    def objective(trial: optuna.Trial) -> float:
        L = trial.suggest_float("L", 0.01, 0.20, log=True)
        gamma = trial.suggest_float("gamma", 5.0, 200.0, log=True)
        loss_dB = trial.suggest_float("loss_dB", 3.0, 25.0)
        dP_per_round = trial.suggest_float("dP_per_round", 1e-6, 5e-4, log=True)
        coupling = -trial.suggest_float("abs_coupling", 1e-3, 0.2, log=True)

        eta = 10.0 ** (-loss_dB / 10.0)
        J = build_coupling_matrix(n, edges, coupling)

        try:
            best_cuts, _ = simulate_cim_batch(
                n=n,
                J=J,
                edges=edges,
                num_rounds=num_rounds,
                num_trials=seeds.shape[0],
                kappa=FIXED_KAPPA,
                L=L,
                gamma=gamma,
                eta=eta,
                bandwidth=FIXED_BANDWIDTH,
                photon_energy=FIXED_PHOTON_ENERGY,
                dP_per_round=dP_per_round,
                seeds=seeds,
            )
        except Exception as exc:
            print(f"  [trial {trial.number}] sim error: {exc}")
            return 0.0

        mean_cut = float(np.mean(best_cuts))
        trial.set_user_attr("std_cut", float(np.std(best_cuts)))
        trial.set_user_attr("max_cut", int(np.max(best_cuts)))
        trial.set_user_attr("min_cut", int(np.min(best_cuts)))
        return mean_cut

    return objective


def tune_one_rounds(n, edges, num_rounds, seeds, n_optuna_trials):
    """1 条件分のチューニングを実行し、(best_params, study, elapsed) を返す。"""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.enqueue_trial(PAPER_WARM_START)

    t0 = time.time()
    log_every = max(1, n_optuna_trials // 10)

    def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % log_every == 0:
            now = time.time()
            elapsed = now - t0
            print(
                f"    [{trial.number + 1:>4d}/{n_optuna_trials}] "
                f"best mean_cut={study.best_value:.2f}  "
                f"({(trial.number + 1) / elapsed:.2f} trial/s, "
                f"elapsed {elapsed:.1f}s)"
            )

    objective = make_objective(n, edges, num_rounds, seeds)
    study.optimize(objective, n_trials=n_optuna_trials, callbacks=[cb])
    elapsed = time.time() - t0

    best = study.best_trial
    return best, study, elapsed


def plot_history(studies: dict, out_path: Path, n_optuna_trials: int) -> None:
    """4 条件の Optuna 探索履歴を 2x2 で表示。"""
    rounds_list = sorted(studies.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    axes_flat = axes.flatten()
    for ax, nr in zip(axes_flat, rounds_list):
        study = studies[nr]["study"]
        values = np.array(
            [t.value if t.value is not None else 0.0 for t in study.trials]
        )
        running_best = np.maximum.accumulate(values)
        x = np.arange(1, len(values) + 1)
        ax.scatter(x, values, s=8, color="#1f77b4", alpha=0.35, label="各試行 mean_cut")
        ax.plot(x, running_best, color="#d62728", linewidth=2.0, label="これまでの最良")
        ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.0,
                   label=f"既知 {KNOWN_BEST}")
        ax.set_title(
            f"num_rounds={nr}  最終 best = {study.best_value:.2f}",
            fontsize=11,
        )
        ax.set_xlabel("Optuna 試行番号")
        ax.set_ylabel("mean best_cut")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        apply_ticks_inward(ax)
    fig.suptitle(
        f"CIM パラメータ探索履歴 (5 パラ縮約版, 各 {n_optuna_trials} trial)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_amplitudes(traj_data: dict, out_path: Path, num_show: int = 200) -> None:
    """4 条件の振幅軌跡 c_i(round) を 2x2 で表示。"""
    rounds_list = sorted(traj_data.keys())
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    axes_flat = axes.flatten()
    rng = np.random.default_rng(0)
    for ax, nr in zip(axes_flat, rounds_list):
        d = traj_data[nr]
        c_hist = d["c_history"]   # (S, N)
        sample_rounds = d["sample_rounds"]  # (S,)
        n_spins = c_hist.shape[1]
        sel = rng.choice(n_spins, size=min(num_show, n_spins), replace=False)
        for i in sel:
            ax.plot(sample_rounds + 1, c_hist[:, i], color="black",
                    alpha=0.05, linewidth=0.5)
        mean_abs = np.mean(np.abs(c_hist), axis=1)
        std_c = np.std(c_hist, axis=1)
        ax.plot(sample_rounds + 1, mean_abs, color="#d62728", linewidth=2.0)
        ax.plot(sample_rounds + 1, -mean_abs, color="#d62728", linewidth=2.0)
        ax.fill_between(sample_rounds + 1, -std_c, std_c, color="#1f77b4",
                        alpha=0.15)
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.set_title(
            f"num_rounds={nr}  best_cut={d['best_cut']:.0f}",
            fontsize=11,
        )
        ax.set_xlabel("round step")
        ax.set_ylabel("In-phase amplitude (arb.)")
        ax.grid(alpha=0.3)
        apply_ticks_inward(ax)
    fig.suptitle(
        f"チューニング後ベストパラでの CIM 振幅推移(全 {num_show} スピンを薄線で表示)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_cut_curves(traj_data: dict, out_path: Path) -> None:
    """4 条件の cut vs round を 1 つの図にまとめる(対数 x 軸)。"""
    rounds_list = sorted(traj_data.keys())
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)
    for nr, color in zip(rounds_list, colors):
        d = traj_data[nr]
        ax.plot(d["sample_rounds"] + 1, d["cut_history"], color=color,
                linewidth=1.7, label=f"num_rounds={nr} (best={d['best_cut']:.0f})")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.2,
               label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xscale("log")
    ax.set_xlabel("ラウンド k (log scale)")
    ax.set_ylabel("cut")
    ax.set_title("チューニング後ベストパラでの cut(k) 推移")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    apply_ticks_inward(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_best_vs_rounds(summary_per_rounds: dict, out_path: Path) -> None:
    rounds = sorted(summary_per_rounds.keys())
    means = [summary_per_rounds[r]["best_mean_cut"] for r in rounds]
    stds = [summary_per_rounds[r]["best_std"] for r in rounds]
    bests = [summary_per_rounds[r]["best_max"] for r in rounds]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=130)
    ax.errorbar(rounds, means, yerr=stds, fmt="o-", color="#1f77b4",
                linewidth=2.0, capsize=4, label="ベストパラ平均 ± std (20 seed)")
    ax.plot(rounds, bests, "s--", color="#d62728", linewidth=1.5,
            label="20 seed 中の最良")
    ax.axhline(KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.2,
               label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xscale("log")
    ax.set_xlabel("num_rounds (log scale)")
    ax.set_ylabel("mean best_cut")
    ax.set_title("num_rounds とチューニング後の解品質(5 パラ縮約)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    apply_ticks_inward(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="num_rounds スイープ Optuna チューニング")
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--rounds", type=int, nargs="+",
                        default=[30, 300, 1500, 10000],
                        help="比較する num_rounds のリスト")
    parser.add_argument("--n-optuna-trials", type=int, default=200,
                        help="各条件あたりの Optuna trial 数")
    parser.add_argument("--n-cim-trials", type=int, default=20,
                        help="各 Optuna trial 内で並列実行する CIM seed 数")
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--traj-samples", type=int, default=200,
                        help="振幅軌跡の記録サンプル数(round 軸)")
    parser.add_argument("--tag", type=str, default="",
                        help="出力ディレクトリ名末尾に付ける任意の説明タグ")
    args = parser.parse_args()

    setup_plot_style()

    print(f"Loading {args.graph}...")
    n, k_edges, _adj, edges = load_graph(args.graph)
    print(f"  N={n} K={k_edges}")
    print(
        f"  Fixed: kappa={FIXED_KAPPA:.4g}, bandwidth={FIXED_BANDWIDTH:.4g}, "
        f"photon_energy={FIXED_PHOTON_ENERGY:.4g}"
    )
    print(f"  Searching: L, gamma, loss_dB, dP_per_round, abs_coupling")
    print(f"  Conditions: num_rounds = {args.rounds}")
    print(f"  CIM seeds (parallel via prange): {args.n_cim_trials}")
    print(f"  Optuna trials per condition: {args.n_optuna_trials}")

    seeds = np.arange(
        args.seed_base, args.seed_base + args.n_cim_trials, dtype=np.int64
    )

    # ==== 各条件をチューニング ====
    studies: dict[int, dict] = {}
    summary_per_rounds: dict[int, dict] = {}

    for nr in args.rounds:
        print(f"\n[num_rounds={nr}] Optuna チューニング開始...")
        best, study, elapsed = tune_one_rounds(
            n, edges, nr, seeds, args.n_optuna_trials
        )
        studies[nr] = {"study": study, "elapsed": elapsed}
        summary_per_rounds[nr] = {
            "best_mean_cut": float(best.value),
            "best_params": best.params,
            "best_std": float(best.user_attrs.get("std_cut", 0.0)),
            "best_max": int(best.user_attrs.get("max_cut", 0)),
            "best_min": int(best.user_attrs.get("min_cut", 0)),
            "elapsed_sec": elapsed,
        }
        print(
            f"  done in {elapsed:.1f}s  "
            f"best mean={best.value:.2f}  "
            f"max={best.user_attrs.get('max_cut')}  "
            f"std={best.user_attrs.get('std_cut'):.2f}"
        )
        print(f"  best params:")
        for k, v in best.params.items():
            print(f"    {k} = {v:.6g}")

    # ==== 各条件のベストパラで振幅軌跡を取得 ====
    print("\n[trajectory] 各条件のベストパラで振幅軌跡を計算中...")
    traj_data: dict[int, dict] = {}
    for nr in args.rounds:
        params = summary_per_rounds[nr]["best_params"]
        eta = 10.0 ** (-params["loss_dB"] / 10.0)
        coupling = -params["abs_coupling"]
        J = build_coupling_matrix(n, edges, coupling)
        t0 = time.time()
        c_hist, cut_hist, sample_rounds, best_cut, _signs = simulate_cim_with_trajectory(
            n=n,
            J=J,
            edges=edges,
            num_rounds=nr,
            kappa=FIXED_KAPPA,
            L=params["L"],
            gamma=params["gamma"],
            eta=eta,
            bandwidth=FIXED_BANDWIDTH,
            photon_energy=FIXED_PHOTON_ENERGY,
            dP_per_round=params["dP_per_round"],
            seed=int(seeds[0]),
            num_samples=min(args.traj_samples, nr),
        )
        traj_data[nr] = {
            "c_history": c_hist,
            "cut_history": cut_hist,
            "sample_rounds": sample_rounds,
            "best_cut": best_cut,
        }
        print(f"  num_rounds={nr}: traj recorded ({time.time() - t0:.2f}s)  best_cut={best_cut:.0f}")

    # ==== 出力 ====
    kind_root = get_kind_root()
    v = next_version(kind_root)
    desc = build_description(args)
    out_dir = kind_root / f"v{v}_{desc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    plot_history(studies, out_dir / "history.png", args.n_optuna_trials)
    plot_amplitudes(traj_data, out_dir / "amplitudes.png")
    plot_cut_curves(traj_data, out_dir / "cut_curves.png")
    plot_best_vs_rounds(summary_per_rounds, out_dir / "best_vs_rounds.png")

    # --- 生データ (npz) ---
    npz_path = out_dir / "trajectories.npz"
    npz_payload = {}
    for nr in args.rounds:
        npz_payload[f"c_history_R{nr}"] = traj_data[nr]["c_history"]
        npz_payload[f"cut_history_R{nr}"] = traj_data[nr]["cut_history"]
        npz_payload[f"sample_rounds_R{nr}"] = traj_data[nr]["sample_rounds"]
    np.savez(npz_path, **npz_payload)
    print(f"  saved: {npz_path}")

    # --- summary JSON ---
    summary = {
        "graph": args.graph,
        "n": n,
        "k": k_edges,
        "known_best": KNOWN_BEST,
        "fixed_params": {
            "kappa": FIXED_KAPPA,
            "bandwidth": FIXED_BANDWIDTH,
            "photon_energy": FIXED_PHOTON_ENERGY,
        },
        "searched_params": ["L", "gamma", "loss_dB", "dP_per_round", "abs_coupling"],
        "n_optuna_trials": args.n_optuna_trials,
        "n_cim_trials": args.n_cim_trials,
        "rounds_results": {
            str(nr): summary_per_rounds[nr] for nr in args.rounds
        },
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved: {json_path}")

    # --- 表示用サマリ ---
    print("\n" + "=" * 78)
    print(f"{'num_rounds':>10} {'mean':>10} {'std':>8} {'max':>8} {'min':>8} {'time[s]':>10}")
    print("-" * 78)
    for nr in args.rounds:
        s = summary_per_rounds[nr]
        print(
            f"{nr:>10d} {s['best_mean_cut']:>10.2f} {s['best_std']:>8.2f} "
            f"{s['best_max']:>8d} {s['best_min']:>8d} {s['elapsed_sec']:>10.1f}"
        )
    print("=" * 78)


if __name__ == "__main__":
    main()
