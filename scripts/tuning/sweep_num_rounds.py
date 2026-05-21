"""num_rounds = [30, 300, 3000, 10000] のスイープで Optuna 最適化を行い、
振幅軌跡と最適化前後の cut 分布を 2 枚の図にまとめる。

設計:
- 各 num_rounds に対して独立な study を sqlite に作成
  (study_name = "cim_g22_sweep_nr<rounds>", load_if_exists=True で再開可)
- TPE で 5 パラ探索(物理パラ kappa/BW/photon_energy は論文値固定)
- 計算予算は rounds に反比例して trial 数を調節(短いほど多く探索)
- 各 rounds で held-out seeds で paper vs optuna best を 100 試行ずつ評価
- 振幅軌跡は単発 CIM(numpy 実装)で mean|c|/round を 1 トレースだけ記録

出力 (results/<today>/ 配下):
- sweep_nr<rounds>_best_params.json
- sweep_amplitude.png           各 rounds の振幅軌跡を 4 パネル比較
- sweep_cut_distribution.png    paper vs optuna の best_cut ヒスト 4 パネル
- sweep_summary.json            全 rounds の集約結果
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy.sparse._sparsetools import csr_matvec

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# スイープ設定
# ============================================================
ROUNDS_LIST: list[int] = [30, 300, 3000, 10000]

# 計算予算: rounds × n_trials ≈ 一定 で揃える。
# (短い rounds は cheap なので試行を多く、長い rounds は少なく)
# 環境変数 BUDGET=light/medium/heavy で切替可。
BUDGET_PRESETS: dict[str, dict[int, int]] = {
    "light": {30: 300, 300: 150, 3000: 60, 10000: 20},
    "medium": {30: 1000, 300: 500, 3000: 200, 10000: 60},
    "heavy": {30: 3000, 300: 1500, 3000: 600, 10000: 200},
}
BUDGET_NAME: str = os.environ.get("BUDGET", "medium")
OPTUNA_TRIALS_BY_ROUNDS: dict[int, int] = BUDGET_PRESETS[BUDGET_NAME]

N_CIM_TRIALS_PER_OPTUNA: int = 20
N_HELDOUT_TRIALS: int = 100
SEED_OPTUNA_BASE: int = 0
SEED_HELDOUT_START: int = 100
SEED_TRAJECTORY: int = 42

# ============================================================
# 探索空間 & 固定パラメータ
# ============================================================
# 物理セットアップ(論文値)は固定し、アルゴリズム側の 5 パラを探索する。
FIXED_KAPPA: float = 130.0
FIXED_BANDWIDTH: float = 1.0e9
FIXED_PHOTON_ENERGY: float = 1.28e-19

# dP_per_round の探索範囲は、短い rounds でも P_p_max が物理的に意味のある
# 値まで届くよう広めに取る (paper 値 5e-5 は内側に含まれる)。
DP_MIN: float = 1e-7
DP_MAX: float = 1e-2

# 論文値 (8 パラ揃え) — held-out / trajectory 比較で使用
PAPER_PARAMS: dict[str, float] = dict(
    kappa=130.0, L=0.05, gamma=42.09, loss_dB=11.0,
    bandwidth=1.0e9, photon_energy=1.28e-19,
    dP_per_round=0.05e-3, coupling=-0.03,
)
# Optuna への warm start (5 パラのみ; abs_coupling は正値で渡す)
PAPER_WARM_START: dict[str, float] = dict(
    L=0.05, gamma=42.09, loss_dB=11.0,
    dP_per_round=5.0e-5, abs_coupling=0.03,
)

KNOWN_BEST: int = 13359
PAPER_FIG8_MEAN: int = 13275

OUT_DIR: Path = Path(
    os.environ.get("OUT_DIR", f"results/{date.today().isoformat()}")
)


# ============================================================
# Optuna objective ファクトリ
# ============================================================
def make_objective(n: int, edges, num_rounds: int, seeds: np.ndarray):
    def objective(trial: optuna.Trial) -> float:
        L = trial.suggest_float("L", 0.01, 0.20, log=True)
        gamma = trial.suggest_float("gamma", 5.0, 200.0, log=True)
        loss_dB = trial.suggest_float("loss_dB", 3.0, 25.0)
        dP_per_round = trial.suggest_float("dP_per_round", DP_MIN, DP_MAX, log=True)
        coupling = -trial.suggest_float("abs_coupling", 1e-3, 0.2, log=True)

        eta = 10.0 ** (-loss_dB / 10.0)
        J = build_coupling_matrix(n, edges, coupling)
        try:
            best_cuts, _ = simulate_cim_batch(
                n=n, J=J, edges=edges,
                num_rounds=num_rounds,
                num_trials=len(seeds),
                kappa=FIXED_KAPPA, L=L, gamma=gamma, eta=eta,
                bandwidth=FIXED_BANDWIDTH, photon_energy=FIXED_PHOTON_ENERGY,
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


# ============================================================
# numpy 単発 CIM (振幅軌跡用)
# ============================================================
def cim_trajectory_meanabs(
    n: int, J, edges, params: dict, num_rounds: int, seed: int
) -> np.ndarray:
    """単一 trial を回し、mean|c_i| の round 推移を返す。

    JIT 版と同じ式 (論文 Eq.3a / 14 / 6) を numpy で実装。
    可視化用なので速度より明瞭さ優先。
    """
    rng = np.random.default_rng(seed)
    eta = 10.0 ** (-params["loss_dB"] / 10.0)
    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt(
        (2.0 - eta) * 0.25 * params["bandwidth"] * params["photon_energy"]
    )

    c = np.zeros(n, dtype=np.float64)
    Jc = np.zeros(n, dtype=np.float64)
    traj = np.zeros(num_rounds, dtype=np.float64)

    J_data = J.data
    J_indices = J.indices
    J_indptr = J.indptr

    for k in range(num_rounds):
        P_p = (k + 1) * params["dP_per_round"]
        g0 = 2.0 * params["kappa"] * np.sqrt(P_p) * params["L"]
        Jc.fill(0.0)
        csr_matvec(n, n, J_indptr, J_indices, J_data, c, Jc)
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in
        half_g = 0.5 * g0 * (1.0 - params["gamma"] * I_in)
        sqrt_G_I = np.exp(half_g)
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)
        c = sqrt_G_I * coupled_in + N_I
        traj[k] = float(np.mean(np.abs(c)))
    return traj


# ============================================================
# helper: best_params (5 パラ) → 8 パラ全部 dict 化
# ============================================================
def expand_params(best_params: dict) -> dict:
    """Optuna の 5 パラ best を、固定パラと合体して 8 パラの dict にする。"""
    out = dict(
        kappa=FIXED_KAPPA,
        bandwidth=FIXED_BANDWIDTH,
        photon_energy=FIXED_PHOTON_ENERGY,
        L=best_params["L"],
        gamma=best_params["gamma"],
        loss_dB=best_params["loss_dB"],
        dP_per_round=best_params["dP_per_round"],
        coupling=-best_params["abs_coupling"],
    )
    return out


# ============================================================
# メイン
# ============================================================
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {OUT_DIR}")
    print(f"Budget preset: {BUDGET_NAME} → {OPTUNA_TRIALS_BY_ROUNDS}")

    print("Loading G22...")
    n, k_edges, _, edges = load_graph("input/G22.txt")
    print(f"  N={n}, K={k_edges}")

    # ---- JIT warmup (1 回だけ全 rounds リストの最小値で軽く回す) ----
    print("JIT warmup...")
    seeds_warm = np.arange(0, 4, dtype=np.int64)
    J_warm = build_coupling_matrix(n, edges, -0.03)
    _ = simulate_cim_batch(
        n=n, J=J_warm, edges=edges, num_rounds=10, num_trials=4,
        kappa=FIXED_KAPPA, L=0.05, gamma=42.09, eta=0.0794,
        bandwidth=FIXED_BANDWIDTH, photon_energy=FIXED_PHOTON_ENERGY,
        dP_per_round=5e-5, seeds=seeds_warm,
    )

    seeds_optuna = np.arange(
        SEED_OPTUNA_BASE, SEED_OPTUNA_BASE + N_CIM_TRIALS_PER_OPTUNA,
        dtype=np.int64,
    )
    seeds_heldout = np.arange(
        SEED_HELDOUT_START, SEED_HELDOUT_START + N_HELDOUT_TRIALS,
        dtype=np.int64,
    )

    # ---- スイープ ----
    storage_url = f"sqlite:///{OUT_DIR}/sweep_optuna.db"
    summary: dict[int, dict] = {}

    for num_rounds in ROUNDS_LIST:
        n_trials = OPTUNA_TRIALS_BY_ROUNDS[num_rounds]
        print()
        print("=" * 70)
        print(f"=== num_rounds = {num_rounds}  /  Optuna trials = {n_trials} ===")
        print("=" * 70)

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=0)
        study = optuna.create_study(
            direction="maximize", sampler=sampler,
            study_name=f"cim_g22_sweep_nr{num_rounds}",
            storage=storage_url, load_if_exists=True,
        )
        if len(study.trials) == 0:
            study.enqueue_trial(PAPER_WARM_START)

        objective = make_objective(n, edges, num_rounds, seeds_optuna)

        log_every = max(10, n_trials // 10)
        t0 = time.time()

        def cb(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            if (trial.number + 1) % log_every == 0 or trial.number + 1 == n_trials:
                elapsed = time.time() - t0
                rate = (trial.number + 1) / max(elapsed, 1e-9)
                print(
                    f"  [{trial.number + 1:5d}/{n_trials}] "
                    f"best mean_cut = {study.best_value:.2f}  "
                    f"({rate:.2f} trial/s, elapsed {elapsed:.1f}s)"
                )

        if len(study.trials) >= n_trials:
            print(f"  -> already completed ({len(study.trials)} trials), skipping optimize")
        else:
            remaining = n_trials - len(study.trials)
            study.optimize(objective, n_trials=remaining, callbacks=[cb])
        elapsed = time.time() - t0
        print(f"  Optuna done in {elapsed:.1f}s  best mean_cut = {study.best_value:.3f}")

        best_params_5 = dict(study.best_params)
        best_params_8 = expand_params(best_params_5)

        # ---- held-out 評価 (paper vs optuna best) ----
        print(f"  held-out eval (seeds {SEED_HELDOUT_START}..{SEED_HELDOUT_START + N_HELDOUT_TRIALS - 1})...")
        t1 = time.time()
        paper_cuts, _ = simulate_cim_batch(
            n=n, J=build_coupling_matrix(n, edges, PAPER_PARAMS["coupling"]),
            edges=edges, num_rounds=num_rounds,
            num_trials=N_HELDOUT_TRIALS,
            kappa=PAPER_PARAMS["kappa"], L=PAPER_PARAMS["L"],
            gamma=PAPER_PARAMS["gamma"],
            eta=10.0 ** (-PAPER_PARAMS["loss_dB"] / 10.0),
            bandwidth=PAPER_PARAMS["bandwidth"],
            photon_energy=PAPER_PARAMS["photon_energy"],
            dP_per_round=PAPER_PARAMS["dP_per_round"],
            seeds=seeds_heldout,
        )
        optuna_cuts, _ = simulate_cim_batch(
            n=n, J=build_coupling_matrix(n, edges, best_params_8["coupling"]),
            edges=edges, num_rounds=num_rounds,
            num_trials=N_HELDOUT_TRIALS,
            kappa=best_params_8["kappa"], L=best_params_8["L"],
            gamma=best_params_8["gamma"],
            eta=10.0 ** (-best_params_8["loss_dB"] / 10.0),
            bandwidth=best_params_8["bandwidth"],
            photon_energy=best_params_8["photon_energy"],
            dP_per_round=best_params_8["dP_per_round"],
            seeds=seeds_heldout,
        )
        t_held = time.time() - t1
        diff = float(optuna_cuts.mean() - paper_cuts.mean())
        print(
            f"    paper : mean={paper_cuts.mean():.2f} std={paper_cuts.std():.2f} "
            f"best={int(paper_cuts.max())} worst={int(paper_cuts.min())}"
        )
        print(
            f"    optuna: mean={optuna_cuts.mean():.2f} std={optuna_cuts.std():.2f} "
            f"best={int(optuna_cuts.max())} worst={int(optuna_cuts.min())}"
        )
        print(f"    diff (optuna - paper) = {diff:+.2f}   (held-out {t_held:.1f}s)")

        # ---- 振幅軌跡 1 トレース (paper / optuna 各 1) ----
        print(f"  trajectory (single trial, seed={SEED_TRAJECTORY})...")
        t2 = time.time()
        J_paper = build_coupling_matrix(n, edges, PAPER_PARAMS["coupling"])
        J_optuna = build_coupling_matrix(n, edges, best_params_8["coupling"])
        traj_paper = cim_trajectory_meanabs(
            n, J_paper, edges, PAPER_PARAMS, num_rounds, SEED_TRAJECTORY
        )
        traj_optuna = cim_trajectory_meanabs(
            n, J_optuna, edges, best_params_8, num_rounds, SEED_TRAJECTORY
        )
        print(f"    trajectory done in {time.time() - t2:.1f}s")

        # ---- 個別の best_params JSON ----
        result_entry = {
            "num_rounds": num_rounds,
            "n_optuna_trials": n_trials,
            "n_cim_trials_per_optuna": N_CIM_TRIALS_PER_OPTUNA,
            "elapsed_sec_optuna": elapsed,
            "best_value_mean_cut": float(study.best_value),
            "best_params": best_params_5,
            "fixed_params": {
                "kappa": FIXED_KAPPA,
                "bandwidth": FIXED_BANDWIDTH,
                "photon_energy": FIXED_PHOTON_ENERGY,
            },
            "heldout": {
                "n_trials": N_HELDOUT_TRIALS,
                "seed_start": SEED_HELDOUT_START,
                "paper": {
                    "mean": float(paper_cuts.mean()),
                    "std": float(paper_cuts.std()),
                    "best": int(paper_cuts.max()),
                    "worst": int(paper_cuts.min()),
                    "cuts": paper_cuts.astype(int).tolist(),
                },
                "optuna": {
                    "mean": float(optuna_cuts.mean()),
                    "std": float(optuna_cuts.std()),
                    "best": int(optuna_cuts.max()),
                    "worst": int(optuna_cuts.min()),
                    "cuts": optuna_cuts.astype(int).tolist(),
                },
                "diff_mean": diff,
            },
        }
        with open(OUT_DIR / f"sweep_nr{num_rounds}_best_params.json", "w", encoding="utf-8") as f:
            json.dump(result_entry, f, indent=2, ensure_ascii=False)
        print(f"    saved: {OUT_DIR / f'sweep_nr{num_rounds}_best_params.json'}")

        summary[num_rounds] = {
            "result": result_entry,
            "traj_paper": traj_paper,
            "traj_optuna": traj_optuna,
            "paper_cuts": paper_cuts,
            "optuna_cuts": optuna_cuts,
        }

    # ============================================================
    # 図 1: 振幅軌跡 (mean|c_i| vs round) 4 パネル
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    for ax, num_rounds in zip(axes.flat, ROUNDS_LIST):
        s = summary[num_rounds]
        x = np.arange(1, num_rounds + 1)
        ax.plot(x, s["traj_paper"], color="#1f77b4", linewidth=1.4,
                label="論文値パラメータ")
        ax.plot(x, s["traj_optuna"], color="#d62728", linewidth=1.4,
                label="Optuna 最適パラメータ")
        ax.set_xlabel("round")
        ax.set_ylabel("mean |c_i| (全スピン平均振幅)")
        ax.set_title(
            f"num_rounds = {num_rounds}  "
            f"(Optuna trials = {OPTUNA_TRIALS_BY_ROUNDS[num_rounds]})"
        )
        # 振幅は 1e-6 〜 1e+2 のオーダで広いので log y
        ax.set_yscale("log")
        if num_rounds >= 1000:
            ax.set_xscale("log")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        "CIM 振幅 mean|c_i| の round 推移: num_rounds スイープ\n"
        f"(seed={SEED_TRAJECTORY}, G22, 論文値 vs Optuna 最適)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png_amp = OUT_DIR / "sweep_amplitude.png"
    i = 1
    while out_png_amp.exists():
        i += 1
        out_png_amp = OUT_DIR / f"v{i}_sweep_amplitude.png"
    fig.savefig(out_png_amp)
    print(f"\nSaved: {out_png_amp}")

    # ============================================================
    # 図 2: held-out cut 分布 (paper vs optuna) 4 パネル
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=130)
    for ax, num_rounds in zip(axes.flat, ROUNDS_LIST):
        s = summary[num_rounds]
        p = s["paper_cuts"]
        o = s["optuna_cuts"]
        all_cuts = np.concatenate([p, o])
        x_min = int(all_cuts.min()) - 5
        x_max = int(all_cuts.max()) + 5
        if x_max - x_min < 20:
            x_min -= 10
            x_max += 10
        bins = np.linspace(x_min, x_max, 30)

        ax.hist(p, bins=bins, color="#1f77b4", alpha=0.6,
                edgecolor="black", linewidth=0.4,
                label=f"論文値 (mean={p.mean():.1f}, best={int(p.max())})")
        ax.hist(o, bins=bins, color="#d62728", alpha=0.6,
                edgecolor="black", linewidth=0.4,
                label=f"Optuna 最適 (mean={o.mean():.1f}, best={int(o.max())})")
        ax.axvline(p.mean(), color="#1f77b4", linestyle=":", linewidth=1.4)
        ax.axvline(o.mean(), color="#d62728", linestyle=":", linewidth=1.4)
        if x_min <= KNOWN_BEST <= x_max:
            ax.axvline(KNOWN_BEST, color="goldenrod", linestyle="--",
                       linewidth=1.2, label=f"既知最良 {KNOWN_BEST}")
        diff = o.mean() - p.mean()
        ax.set_xlabel("best_cut")
        ax.set_ylabel("頻度")
        ax.set_title(
            f"num_rounds = {num_rounds}   差 = {diff:+.1f} "
            f"(held-out {N_HELDOUT_TRIALS} trials)"
        )
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        "Optuna 最適化前後の best_cut 分布: num_rounds スイープ\n"
        f"(G22, held-out seeds {SEED_HELDOUT_START}..{SEED_HELDOUT_START + N_HELDOUT_TRIALS - 1})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png_dist = OUT_DIR / "sweep_cut_distribution.png"
    i = 1
    while out_png_dist.exists():
        i += 1
        out_png_dist = OUT_DIR / f"v{i}_sweep_cut_distribution.png"
    fig.savefig(out_png_dist)
    print(f"Saved: {out_png_dist}")

    # ============================================================
    # 集約 summary JSON
    # ============================================================
    summary_json = {
        "budget": BUDGET_NAME,
        "rounds_list": ROUNDS_LIST,
        "n_optuna_trials": OPTUNA_TRIALS_BY_ROUNDS,
        "n_cim_trials_per_optuna": N_CIM_TRIALS_PER_OPTUNA,
        "n_heldout_trials": N_HELDOUT_TRIALS,
        "fixed_params": {
            "kappa": FIXED_KAPPA,
            "bandwidth": FIXED_BANDWIDTH,
            "photon_energy": FIXED_PHOTON_ENERGY,
        },
        "per_rounds": {
            str(r): {
                "best_mean_cut": summary[r]["result"]["best_value_mean_cut"],
                "best_params": summary[r]["result"]["best_params"],
                "heldout_paper_mean": summary[r]["result"]["heldout"]["paper"]["mean"],
                "heldout_optuna_mean": summary[r]["result"]["heldout"]["optuna"]["mean"],
                "heldout_diff": summary[r]["result"]["heldout"]["diff_mean"],
                "elapsed_sec_optuna": summary[r]["result"]["elapsed_sec_optuna"],
            }
            for r in ROUNDS_LIST
        },
    }
    with open(OUT_DIR / "sweep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)
    print(f"Saved: {OUT_DIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
