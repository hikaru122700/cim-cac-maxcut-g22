"""ラウンドごとにCIMパラメータを最適化し、振幅変化を観察する。

方針:
- シミュレーションを複数のフェーズに分割
- 各フェーズで最適なパラメータ（dP_per_round, coupling等）をOptunaで探索
- フェーズごとの最適パラメータの推移を可視化
- 最適パラメータでの振幅ダイナミクスを観察

使い方:
  python scripts/tuning/tune_cim_per_round.py [total_rounds]
  例: python scripts/tuning/tune_cim_per_round.py 300
"""

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from typing import NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.sparse._sparsetools import csr_matvec

from modules.CIM import build_coupling_matrix, load_graph

plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

N_OPTUNA_TRIALS_PER_PHASE = 100
N_CIM_TRIALS = 10
KNOWN_BEST = 13359

N_PHASES = 10
ROUNDS_PER_PHASE = 300
TOTAL_ROUNDS = 3000


class PhaseParams(NamedTuple):
    kappa: float
    L: float
    gamma: float
    loss_dB: float
    bandwidth: float
    photon_energy: float
    dP_per_round: float
    coupling: float


def load_graph_data():
    n, k, adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    edges_np = np.asarray(edges, dtype=np.int64)
    return n, k, edges, edges_np


def simulate_phase(
    n: int,
    J,
    edges_np: np.ndarray,
    c_init: np.ndarray,
    start_round: int,
    end_round: int,
    params: PhaseParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, np.ndarray]:
    """1フェーズ分のCIMシミュレーションを実行。

    Returns:
        c_final: フェーズ終了時の振幅
        best_cut: フェーズ中の最良カット
        history: 振幅の履歴 (shape: (end_round - start_round, n))
    """
    eta = 10.0 ** (-params.loss_dB / 10.0)
    c = c_init.copy()
    Jc = np.zeros(n, dtype=np.float64)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * params.bandwidth * params.photon_energy)
    sqrt_eta = np.sqrt(eta)

    edge_a = edges_np[:, 0]
    edge_b = edges_np[:, 1]

    num_rounds = end_round - start_round
    history = np.zeros((num_rounds, n), dtype=np.float64)
    best_cut = 0

    for k_rel, k in enumerate(range(start_round, end_round)):
        P_p = (k + 1) * params.dP_per_round
        g0 = 2.0 * params.kappa * np.sqrt(P_p) * params.L

        Jc.fill(0.0)
        csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in
        half_g = 0.5 * g0 * (1.0 - params.gamma * I_in)
        sqrt_G_I = np.exp(half_g)
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)
        c = sqrt_G_I * coupled_in + N_I

        history[k_rel] = c
        signs = c > 0
        cut = int((signs[edge_a] != signs[edge_b]).sum())
        if cut > best_cut:
            best_cut = cut

    return c, best_cut, history


def simulate_full_with_phase_params(
    n: int,
    edges: list,
    edges_np: np.ndarray,
    phase_params_list: list[PhaseParams],
    seed: int,
) -> tuple[np.ndarray, list[int], list[np.ndarray]]:
    """各フェーズに異なるパラメータを使用してシミュレーションを実行。"""
    rng = np.random.default_rng(seed)
    c = np.zeros(n, dtype=np.float64)

    all_cuts = []
    all_histories = []

    for phase_idx, params in enumerate(phase_params_list):
        J = build_coupling_matrix(n, edges, params.coupling)
        start_round = phase_idx * ROUNDS_PER_PHASE
        end_round = (phase_idx + 1) * ROUNDS_PER_PHASE

        c, best_cut, history = simulate_phase(
            n, J, edges_np, c, start_round, end_round, params, rng
        )
        all_cuts.append(best_cut)
        all_histories.append(history)

    return c, all_cuts, all_histories


def run_batch_for_phase(
    n: int,
    edges: list,
    edges_np: np.ndarray,
    c_inits: np.ndarray,
    start_round: int,
    params: PhaseParams,
    num_trials: int,
    seed_base: int,
) -> np.ndarray:
    """複数トライアルでフェーズを実行し、平均カットを返す。"""
    J = build_coupling_matrix(n, edges, params.coupling)
    end_round = start_round + ROUNDS_PER_PHASE
    cuts = np.zeros(num_trials, dtype=np.float64)

    for trial_idx in range(num_trials):
        rng = np.random.default_rng(seed_base + trial_idx)
        c, best_cut, _ = simulate_phase(
            n, J, edges_np, c_inits[trial_idx], start_round, end_round, params, rng
        )
        cuts[trial_idx] = best_cut

    return cuts


def optimize_phase(
    phase_idx: int,
    n: int,
    edges: list,
    edges_np: np.ndarray,
    c_inits: np.ndarray,
    baseline_params: PhaseParams,
) -> tuple[PhaseParams, float]:
    """指定フェーズのパラメータをOptunaで最適化。"""
    start_round = phase_idx * ROUNDS_PER_PHASE

    def objective(trial: optuna.Trial) -> float:
        kappa = trial.suggest_float("kappa", 50.0, 250.0, log=True)
        L = trial.suggest_float("L", 0.02, 0.15, log=True)
        gamma = trial.suggest_float("gamma", 10.0, 150.0, log=True)
        loss_dB = trial.suggest_float("loss_dB", 5.0, 20.0)
        bandwidth = trial.suggest_float("bandwidth", 2e8, 3e9, log=True)
        photon_energy = trial.suggest_float("photon_energy", 0.8e-19, 3e-19, log=True)
        dP_per_round = trial.suggest_float("dP_per_round", 5e-6, 2e-4, log=True)
        coupling = -trial.suggest_float("abs_coupling", 5e-3, 0.1, log=True)

        params = PhaseParams(
            kappa=kappa, L=L, gamma=gamma, loss_dB=loss_dB,
            bandwidth=bandwidth, photon_energy=photon_energy,
            dP_per_round=dP_per_round, coupling=coupling
        )

        try:
            cuts = run_batch_for_phase(
                n, edges, edges_np, c_inits, start_round, params, N_CIM_TRIALS, 0
            )
            return float(np.mean(cuts))
        except Exception:
            return 0.0

    sampler = optuna.samplers.TPESampler(seed=phase_idx)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.enqueue_trial({
        "kappa": baseline_params.kappa,
        "L": baseline_params.L,
        "gamma": baseline_params.gamma,
        "loss_dB": baseline_params.loss_dB,
        "bandwidth": baseline_params.bandwidth,
        "photon_energy": baseline_params.photon_energy,
        "dP_per_round": baseline_params.dP_per_round,
        "abs_coupling": -baseline_params.coupling,
    })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS_PER_PHASE, show_progress_bar=False)

    best = study.best_params
    best_params = PhaseParams(
        kappa=best["kappa"],
        L=best["L"],
        gamma=best["gamma"],
        loss_dB=best["loss_dB"],
        bandwidth=best["bandwidth"],
        photon_energy=best["photon_energy"],
        dP_per_round=best["dP_per_round"],
        coupling=-best["abs_coupling"],
    )

    return best_params, study.best_value


def generate_initial_states(n: int, num_phases: int, baseline_params: PhaseParams, num_trials: int):
    """各フェーズ開始時の初期状態を生成（ベースラインパラメータで事前実行）。"""
    print("各フェーズの初期状態を生成中...")

    n_graph, _, edges, edges_np = load_graph_data()
    J = build_coupling_matrix(n_graph, edges, baseline_params.coupling)

    phase_inits = []

    for phase_idx in range(num_phases):
        c_inits = np.zeros((num_trials, n), dtype=np.float64)

        for trial_idx in range(num_trials):
            rng = np.random.default_rng(trial_idx)
            c = np.zeros(n, dtype=np.float64)

            end_round = phase_idx * ROUNDS_PER_PHASE
            eta = 10.0 ** (-baseline_params.loss_dB / 10.0)
            Jc = np.zeros(n, dtype=np.float64)
            noise_const = np.sqrt((2.0 - eta) * 0.25 * baseline_params.bandwidth * baseline_params.photon_energy)
            sqrt_eta = np.sqrt(eta)

            for k in range(end_round):
                P_p = (k + 1) * baseline_params.dP_per_round
                g0 = 2.0 * baseline_params.kappa * np.sqrt(P_p) * baseline_params.L

                Jc.fill(0.0)
                csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
                coupled_in = sqrt_eta * c + Jc
                I_in = coupled_in * coupled_in
                half_g = 0.5 * g0 * (1.0 - baseline_params.gamma * I_in)
                sqrt_G_I = np.exp(half_g)
                N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)
                c = sqrt_G_I * coupled_in + N_I

            c_inits[trial_idx] = c

        phase_inits.append(c_inits)
        print(f"  フェーズ {phase_idx + 1}/{num_phases} の初期状態生成完了")

    return phase_inits


def next_version(out_dir: Path, prefix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in out_dir.iterdir():
        if p.name.startswith("v") and prefix in p.name:
            try:
                v = int(p.name.split("_")[0][1:])
                existing.append(v)
            except ValueError:
                pass
    return (max(existing) if existing else 0) + 1


def plot_optimal_params_evolution(
    phase_params: list[PhaseParams],
    baseline_params: PhaseParams,
    out_dir: Path,
):
    """フェーズごとの最適パラメータの推移をプロット。"""
    phases = np.arange(1, len(phase_params) + 1)

    param_names = ["kappa", "L", "gamma", "loss_dB", "bandwidth", "photon_energy", "dP_per_round", "coupling"]
    param_labels = ["κ (W^-0.5 m^-1)", "L (m)", "γ (W^-1)", "損失 (dB)",
                    "帯域 (Hz)", "光子エネルギー (J)", "dP/round (W)", "結合強度"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, (name, label) in enumerate(zip(param_names, param_labels)):
        ax = axes[i]
        values = [getattr(p, name) for p in phase_params]
        baseline_val = getattr(baseline_params, name)

        ax.plot(phases, values, "o-", color="C0", lw=2, markersize=8, label="最適化後")
        ax.axhline(baseline_val, color="gray", ls="--", lw=1.5, label=f"ベースライン: {baseline_val:.3g}")

        if name in ["kappa", "L", "gamma", "bandwidth", "photon_energy", "dP_per_round"]:
            ax.set_yscale("log")

        ax.set_xlabel("フェーズ番号")
        ax.set_ylabel(label)
        ax.set_title(f"{label} の推移")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"ラウンドごとのパラメータ最適化結果\n"
        f"({N_PHASES}フェーズ × {ROUNDS_PER_PHASE}ラウンド, "
        f"各フェーズ{N_OPTUNA_TRIALS_PER_PHASE}試行)",
        fontsize=14
    )
    fig.tight_layout()

    v = next_version(out_dir, "per_round_params")
    out_path = out_dir / f"v{v}_per_round_params.png"
    fig.savefig(out_path, dpi=130)
    print(f"保存: {out_path}")
    return out_path


def plot_amplitude_dynamics(
    all_histories_optimized: list[np.ndarray],
    all_histories_baseline: list[np.ndarray],
    phase_cuts_optimized: list[int],
    phase_cuts_baseline: list[int],
    out_dir: Path,
):
    """最適化パラメータとベースラインでの振幅ダイナミクスを比較。"""
    history_opt = np.concatenate(all_histories_optimized, axis=0)
    history_base = np.concatenate(all_histories_baseline, axis=0)

    rounds = np.arange(history_opt.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    mean_abs_opt = np.mean(np.abs(history_opt), axis=1)
    mean_abs_base = np.mean(np.abs(history_base), axis=1)
    ax.plot(rounds, mean_abs_opt, lw=1.5, label="最適化パラメータ", color="C0")
    ax.plot(rounds, mean_abs_base, lw=1.5, label="ベースライン", color="C1", alpha=0.7)
    for i in range(N_PHASES):
        ax.axvline(i * ROUNDS_PER_PHASE, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("|c| の平均")
    ax.set_yscale("log")
    ax.set_title("振幅の平均値の推移")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[0, 1]
    std_opt = np.std(history_opt, axis=1)
    std_base = np.std(history_base, axis=1)
    ax.plot(rounds, std_opt, lw=1.5, label="最適化パラメータ", color="C0")
    ax.plot(rounds, std_base, lw=1.5, label="ベースライン", color="C1", alpha=0.7)
    for i in range(N_PHASES):
        ax.axvline(i * ROUNDS_PER_PHASE, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("c の標準偏差")
    ax.set_yscale("log")
    ax.set_title("振幅の標準偏差の推移")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 0]
    sample_idx = np.random.default_rng(0).choice(history_opt.shape[1], size=20, replace=False)
    for idx in sample_idx:
        ax.plot(rounds, history_opt[:, idx], lw=0.5, alpha=0.6)
    for i in range(N_PHASES):
        ax.axvline(i * ROUNDS_PER_PHASE, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i")
    ax.set_title("サンプル頂点の振幅推移（最適化パラメータ）")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 1]
    phases = np.arange(1, N_PHASES + 1)
    width = 0.35
    ax.bar(phases - width/2, phase_cuts_optimized, width, label="最適化パラメータ", color="C0")
    ax.bar(phases + width/2, phase_cuts_baseline, width, label="ベースライン", color="C1", alpha=0.7)
    ax.axhline(KNOWN_BEST, color="k", ls="--", lw=1, label=f"既知最良値 {KNOWN_BEST}")
    ax.set_xlabel("フェーズ番号")
    ax.set_ylabel("フェーズ中の最良カット")
    ax.set_title("各フェーズでの最良カット数")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"振幅ダイナミクスの比較: 最適化パラメータ vs ベースライン\n"
        f"(全{TOTAL_ROUNDS}ラウンド, {N_PHASES}フェーズ)",
        fontsize=14
    )
    fig.tight_layout()

    v = next_version(out_dir, "amplitude_dynamics")
    out_path = out_dir / f"v{v}_amplitude_dynamics.png"
    fig.savefig(out_path, dpi=130)
    print(f"保存: {out_path}")
    return out_path


def run_optimization(total_rounds: int, out_dir: Path):
    """指定ラウンド数でパラメータ最適化を実行。"""
    n_phases = 10
    rounds_per_phase = total_rounds // n_phases

    print("=" * 60)
    print(f"ラウンドごとのCIMパラメータ最適化 (総ラウンド数: {total_rounds})")
    print(f"  フェーズ数: {n_phases}")
    print(f"  各フェーズのラウンド数: {rounds_per_phase}")
    print(f"  各フェーズのOptuna試行数: {N_OPTUNA_TRIALS_PER_PHASE}")
    print(f"  1試行あたりのCIM実行数: {N_CIM_TRIALS}")
    print("=" * 60)

    n, k, edges, edges_np = load_graph_data()
    print(f"グラフ: N={n}, K={k}")

    baseline_params = PhaseParams(
        kappa=130.0,
        L=0.05,
        gamma=42.09,
        loss_dB=11.0,
        bandwidth=1.0e9,
        photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
        coupling=-0.03,
    )

    global ROUNDS_PER_PHASE, N_PHASES, TOTAL_ROUNDS
    ROUNDS_PER_PHASE = rounds_per_phase
    N_PHASES = n_phases
    TOTAL_ROUNDS = total_rounds

    phase_inits = generate_initial_states(n, n_phases, baseline_params, N_CIM_TRIALS)

    print("\n各フェーズのパラメータを最適化中...")
    t0 = time.time()
    optimized_params: list[PhaseParams] = []
    phase_best_values: list[float] = []

    for phase_idx in range(n_phases):
        print(f"\nフェーズ {phase_idx + 1}/{n_phases} を最適化中...")
        best_params, best_value = optimize_phase(
            phase_idx, n, edges, edges_np, phase_inits[phase_idx], baseline_params
        )
        optimized_params.append(best_params)
        phase_best_values.append(best_value)
        print(f"  最良カット: {best_value:.1f}")
        print(f"  dP_per_round: {best_params.dP_per_round:.3e}")
        print(f"  coupling: {best_params.coupling:.4f}")

    elapsed = time.time() - t0
    print(f"\n最適化完了: {elapsed:.1f}秒")

    print("\n最適化パラメータでシミュレーション実行中...")
    c_final_opt, cuts_opt, histories_opt = simulate_full_with_phase_params(
        n, edges, edges_np, optimized_params, seed=42
    )

    print("ベースラインパラメータでシミュレーション実行中...")
    baseline_list = [baseline_params] * n_phases
    c_final_base, cuts_base, histories_base = simulate_full_with_phase_params(
        n, edges, edges_np, baseline_list, seed=42
    )

    print(f"\n結果サマリー ({total_rounds}ラウンド):")
    print(f"  最適化パラメータでの最良カット: {max(cuts_opt)}")
    print(f"  ベースラインでの最良カット: {max(cuts_base)}")
    print(f"  既知最良値: {KNOWN_BEST}")

    subdir = out_dir / f"rounds_{total_rounds}"
    subdir.mkdir(parents=True, exist_ok=True)

    plot_optimal_params_evolution(optimized_params, baseline_params, subdir)
    plot_amplitude_dynamics(histories_opt, histories_base, cuts_opt, cuts_base, subdir)

    results = {
        "total_rounds": total_rounds,
        "n_phases": n_phases,
        "rounds_per_phase": rounds_per_phase,
        "n_optuna_trials_per_phase": N_OPTUNA_TRIALS_PER_PHASE,
        "n_cim_trials": N_CIM_TRIALS,
        "elapsed_sec": elapsed,
        "baseline_params": baseline_params._asdict(),
        "phase_params": [p._asdict() for p in optimized_params],
        "phase_best_values": phase_best_values,
        "cuts_optimized": cuts_opt,
        "cuts_baseline": cuts_base,
        "known_best": KNOWN_BEST,
    }

    v = next_version(subdir, "per_round_optimization")
    out_json = subdir / f"v{v}_per_round_optimization.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"保存: {out_json}")

    return results


def main():
    round_configs = [30, 300, 3000, 30000]

    if len(sys.argv) > 1:
        try:
            round_configs = [int(sys.argv[1])]
        except ValueError:
            pass

    out_dir = ROOT / "results" / date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for total_rounds in round_configs:
        print("\n" + "=" * 70)
        print(f"実行: {total_rounds} ラウンド")
        print("=" * 70)
        results = run_optimization(total_rounds, out_dir)
        all_results[total_rounds] = results

    print("\n" + "=" * 70)
    print("全ラウンド数での結果サマリー")
    print("=" * 70)
    for total_rounds, res in all_results.items():
        print(f"  {total_rounds:>5} ラウンド: "
              f"最適化={max(res['cuts_optimized'])}, "
              f"ベースライン={max(res['cuts_baseline'])}, "
              f"時間={res['elapsed_sec']:.1f}秒")

    summary_path = out_dir / "per_round_summary.json"
    summary_data = {
        str(k): {
            "max_cut_optimized": max(v["cuts_optimized"]),
            "max_cut_baseline": max(v["cuts_baseline"]),
            "elapsed_sec": v["elapsed_sec"],
        }
        for k, v in all_results.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nサマリー保存: {summary_path}")


if __name__ == "__main__":
    main()
