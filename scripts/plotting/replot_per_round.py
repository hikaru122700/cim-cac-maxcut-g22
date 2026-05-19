"""tune_cim_per_round.py の結果JSONから日本語フォントで再プロットする。

Linuxで日本語フォントがない問題を回避するため、Noto Sans CJK JP を使用。
振幅履歴は保存されていないため、JSONのパラメータで再シミュレーションする。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
from collections import namedtuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

from scripts.tuning.tune_cim_per_round import (
    PhaseParams, simulate_phase, load_graph_data,
)
from modules.CIM import build_coupling_matrix

KNOWN_BEST = 13359


def simulate_for_history(n, edges, edges_np, phase_params_list, rounds_per_phase, seed):
    rng = np.random.default_rng(seed)
    c = np.zeros(n, dtype=np.float64)
    all_cuts = []
    all_histories = []
    for phase_idx, params in enumerate(phase_params_list):
        J = build_coupling_matrix(n, edges, params.coupling)
        start = phase_idx * rounds_per_phase
        end = (phase_idx + 1) * rounds_per_phase
        c, best_cut, history = simulate_phase(
            n, J, edges_np, c, start, end, params, rng
        )
        all_cuts.append(best_cut)
        all_histories.append(history)
    return c, all_cuts, all_histories


def next_version(out_dir: Path, suffix: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    for p in out_dir.iterdir():
        if p.name.startswith("v") and suffix in p.name:
            try:
                v = int(p.name.split("_")[0][1:])
                existing.append(v)
            except ValueError:
                pass
    return (max(existing) if existing else 0) + 1


def plot_params(phase_params, baseline_params, n_phases, n_optuna, out_dir, rounds_per_phase, total_rounds):
    phases = np.arange(1, n_phases + 1)
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
        f"({n_phases}フェーズ × {rounds_per_phase}ラウンド = {total_rounds}ラウンド, "
        f"各フェーズ{n_optuna}試行)",
        fontsize=14
    )
    fig.tight_layout()
    v = next_version(out_dir, "per_round_params_jp")
    out_path = out_dir / f"v{v}_per_round_params_jp.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"保存: {out_path}")


def plot_dynamics(histories_opt, histories_base, cuts_opt, cuts_base,
                  n_phases, rounds_per_phase, total_rounds, out_dir,
                  final_opt_mean, final_opt_std, final_base_mean, final_base_std):
    history_opt = np.concatenate(histories_opt, axis=0)
    history_base = np.concatenate(histories_base, axis=0)
    rounds = np.arange(history_opt.shape[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    mean_abs_opt = np.mean(np.abs(history_opt), axis=1)
    mean_abs_base = np.mean(np.abs(history_base), axis=1)
    ax.plot(rounds, mean_abs_opt, lw=1.5, label="最適化パラメータ", color="C0")
    ax.plot(rounds, mean_abs_base, lw=1.5, label="ベースライン", color="C1", alpha=0.7)
    for i in range(n_phases):
        ax.axvline(i * rounds_per_phase, color="gray", ls=":", lw=0.5, alpha=0.5)
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
    for i in range(n_phases):
        ax.axvline(i * rounds_per_phase, color="gray", ls=":", lw=0.5, alpha=0.5)
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
    for i in range(n_phases):
        ax.axvline(i * rounds_per_phase, color="gray", ls=":", lw=0.5, alpha=0.5)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i")
    ax.set_title("サンプル頂点の振幅推移（最適化パラメータ）")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 1]
    phases = np.arange(1, n_phases + 1)
    width = 0.35
    ax.bar(phases - width/2, cuts_opt, width, label="最適化パラメータ", color="C0")
    ax.bar(phases + width/2, cuts_base, width, label="ベースライン", color="C1", alpha=0.7)
    ax.axhline(KNOWN_BEST, color="k", ls="--", lw=1, label=f"既知最良値 {KNOWN_BEST}")
    ax.set_xlabel("フェーズ番号")
    ax.set_ylabel("フェーズ中の最良カット (seed=42)")
    ax.set_title("各フェーズでの最良カット数")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"振幅ダイナミクス比較 ({total_rounds}ラウンド, {n_phases}フェーズ)\n"
        f"held-out 100シード: 最適化={final_opt_mean:.0f}±{final_opt_std:.0f}, "
        f"ベースライン={final_base_mean:.0f}±{final_base_std:.0f}",
        fontsize=14
    )
    fig.tight_layout()
    v = next_version(out_dir, "amplitude_dynamics_jp")
    out_path = out_dir / f"v{v}_amplitude_dynamics_jp.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"保存: {out_path}")


def replot_from_json(json_path: Path):
    print(f"\n読み込み: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    total_rounds = data["total_rounds"]
    n_phases = data["n_phases"]
    rounds_per_phase = data["rounds_per_phase"]
    n_optuna = data["n_optuna_trials_per_phase"]

    baseline_params = PhaseParams(**data["baseline_params"])
    phase_params = [PhaseParams(**p) for p in data["phase_params"]]

    n, k, edges, edges_np = load_graph_data()

    print(f"  {total_rounds}ラウンド × {n_phases}フェーズ、再シミュレーション中...")
    import scripts.tuning.tune_cim_per_round as tune_module
    tune_module.ROUNDS_PER_PHASE = rounds_per_phase
    tune_module.N_PHASES = n_phases
    tune_module.TOTAL_ROUNDS = total_rounds

    _, cuts_opt, histories_opt = simulate_for_history(
        n, edges, edges_np, phase_params, rounds_per_phase, seed=42
    )
    baseline_list = [baseline_params] * n_phases
    _, cuts_base, histories_base = simulate_for_history(
        n, edges, edges_np, baseline_list, rounds_per_phase, seed=42
    )

    out_dir = json_path.parent

    plot_params(phase_params, baseline_params, n_phases, n_optuna,
                out_dir, rounds_per_phase, total_rounds)

    final_opt_mean = data.get("final_optimized_mean", float(np.max(cuts_opt)))
    final_opt_std = data.get("final_optimized_std", 0.0)
    final_base_mean = data.get("final_baseline_mean", float(np.max(cuts_base)))
    final_base_std = data.get("final_baseline_std", 0.0)

    plot_dynamics(histories_opt, histories_base, cuts_opt, cuts_base,
                  n_phases, rounds_per_phase, total_rounds, out_dir,
                  final_opt_mean, final_opt_std, final_base_mean, final_base_std)


def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        results_dir = ROOT / "results" / "2026-05-19"
        targets = []
        for sub in sorted(results_dir.glob("rounds_*")):
            jsons = sorted(sub.glob("v*_per_round_optimization.json"))
            if jsons:
                targets.append(jsons[-1])

    for t in targets:
        replot_from_json(t)


if __name__ == "__main__":
    main()
