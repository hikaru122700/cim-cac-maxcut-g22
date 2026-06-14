"""ARC 生成論文 PumpBench の 8 条件を G22 で再検証し実測データを収集する。

背景:
  AutoResearchClaw が生成した論文 "PumpBench: Learned CIM Pump Schedules Fall
  Short of a Tuned Sigmoid" は、合成 Erdos-Renyi 符号付きグラフ上で 8 つのポンプ
  スケジュール/ソルバ条件を比較し、「開ループのポンプ形状は小レバー / 閉ループ
  フィードバック(CAC)が大レバー」と結論した。原実装(deliverables/code/)は
  実 G-Set 読込機能を持つが、CPU 予算の都合で **n=2000(=G22)をスキップ** していた
  (data.load_heldout: meta["n"] <= 1000)。

本スクリプト:
  - 学習(fit)は論文どおり **合成 factorial グラフ** 上で行う(転移学習の設計を保持)。
  - 評価(forward only)を **実 G22**(input/G22.txt, 既知ベスト 13359)で行う。
  - 各条件の best/mean cut・最適性ギャップ・実時間を収集し JSON/CSV/図/に保存。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/runners/run_arc_pumpbench_g22.py
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

# ---- ARC 実験コードを import path に載せる -------------------------------------
ROOT = Path(__file__).resolve().parents[2]
ARC_CODE = ROOT / "AutoResearchClaw" / "artifacts" / "pump_power_opt" / "deliverables" / "code"
sys.path.insert(0, str(ARC_CODE))

import experiment_config as config  # noqa: E402  (main.py と同じ本番設定)
import data  # noqa: E402
import evaluate as ev  # noqa: E402
from methods import (  # noqa: E402
    LinearRampPump, OptunaScalarSigmoidPump, CACErrorFeedbackSolver,
    DifferentiableFreeformPump, MonotonicConstrainedPump,
    GradientFreeCMAESFreeformPump, KibbleZurekAnalyticPump,
    SpectralConditionedPumpNet,
)

EXPERIMENT_KIND = "arc_pumpbench_g22"
KNOWN_BEST_G22 = 13359

CONDITION_ORDER = [
    ("linear_ramp_baseline", LinearRampPump),
    ("kibble_zurek_analytic_pump", KibbleZurekAnalyticPump),
    ("cac_chaotic_amplitude_control", CACErrorFeedbackSolver),
    ("optuna_scalar_sigmoid_ramp", OptunaScalarSigmoidPump),
    ("differentiable_freeform_pump", DifferentiableFreeformPump),
    ("monotonic_constrained_pump", MonotonicConstrainedPump),
    ("gradient_free_cmaes_freeform_pump", GradientFreeCMAESFreeformPump),
    ("spectral_conditioned_pump_net", SpectralConditionedPumpNet),
]

# 図/表示用の短い日本語ラベル
DISP = {
    "linear_ramp_baseline": "線形ランプ",
    "kibble_zurek_analytic_pump": "KZ解析ランプ",
    "cac_chaotic_amplitude_control": "CAC(閉ループ)",
    "optuna_scalar_sigmoid_ramp": "シグモイド(調整)",
    "differentiable_freeform_pump": "自由形状FD",
    "monotonic_constrained_pump": "単調自由形状",
    "gradient_free_cmaes_freeform_pump": "自由形状CMA-ES",
    "spectral_conditioned_pump_net": "スペクトル条件付MLP",
}


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                v = max(v, int(head[1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_g22() -> "data.Graph":
    path = ROOT / "input" / "G22.txt"
    g = data._read_gset_file(str(path), "G22", KNOWN_BEST_G22)
    return g


def main() -> None:
    t_start = time.time()
    cfgs = config.get_all()
    exp = cfgs.exp

    print("=" * 70)
    print("ARC PumpBench を G22 で再検証(学習=合成 / 評価=実 G22)")
    print(f"  T_steps={cfgs.cim.T_steps} dt={cfgs.cim.dt} K={cfgs.cim.K_trajectories} "
          f"xi={cfgs.cim.xi} sigma={cfgs.cim.sigma}")
    print(f"  opt_steps_freeform={cfgs.opt.opt_steps_freeform} "
          f"optuna_n_trials={cfgs.opt.optuna_n_trials} cmaes_evals={cfgs.opt.cmaes_evals}")
    print(f"  S_train_seeds={exp.S_train_seeds} S_test_seeds={exp.S_test_seeds}")
    print("=" * 70)

    # ----- データ -----
    train_graphs = data.build_synthetic_factorial(cfgs.data, base_seed=exp.S_train_seeds[0])
    g22 = load_g22()
    print(f"DATA: 合成学習グラフ {len(train_graphs)} 個 / 評価=G22 "
          f"(N={g22.n_nodes}, E={g22.edges.shape[0]}, mean_deg={g22.mean_degree:.1f}, "
          f"coupling_scale={g22.coupling_scale:.4f}, best={KNOWN_BEST_G22})")

    # ----- 各条件: fit(合成)→ evaluate(G22) -----
    records = []  # 1 行 = 1 条件
    raw_per_seed = {}
    for name, Cls in CONDITION_ORDER:
        print(f"\n=== 条件 {name} ===")
        cond = Cls(cfgs)
        cond.deadline = None  # 学習を打ち切らず完走させる(opt_steps は有限)
        t_fit0 = time.time()
        try:
            cond.fit(train_graphs, np.random.default_rng(exp.S_train_seeds[0]))
        except Exception as e:  # noqa: BLE001
            print(f"  FIT_WARNING: {e!r}")
        t_fit = time.time() - t_fit0

        best_cuts, mean_cuts, succ = [], [], []
        t_eval0 = time.time()
        per_seed = {}
        for seed in exp.S_test_seeds:
            r = cond.run_on_graph(g22, seed)
            best_cuts.append(r["best_cut"])
            mean_cuts.append(r["mean_cut"])
            succ.append(bool(r["success"]))
            per_seed[seed] = {"best_cut": r["best_cut"], "mean_cut": r["mean_cut"],
                              "success": bool(r["success"])}
            print(f"  seed={seed} best_cut={r['best_cut']:.1f} "
                  f"mean_cut={r['mean_cut']:.1f} success={r['success']}")
        t_eval = time.time() - t_eval0
        raw_per_seed[name] = per_seed

        best_arr = np.array(best_cuts, dtype=float)
        mean_arr = np.array(mean_cuts, dtype=float)
        rec = {
            "condition": name,
            "best_cut_mean": float(best_arr.mean()),
            "best_cut_max": float(best_arr.max()),
            "best_cut_std": float(best_arr.std()),
            "mean_cut_mean": float(mean_arr.mean()),
            "success_rate": float(np.mean(succ)),
            "fit_seconds": t_fit,
            "eval_seconds": t_eval,
        }
        records.append(rec)
        print(f"  -> best_cut: mean={rec['best_cut_mean']:.1f} max={rec['best_cut_max']:.1f} "
              f"std={rec['best_cut_std']:.1f} | fit={t_fit:.1f}s eval={t_eval:.1f}s")

    # ----- 最適性ギャップ(プールド + 既知ベスト) -----
    pooled_best = max(r["best_cut_max"] for r in records)
    C_best = max(pooled_best, float(KNOWN_BEST_G22))  # 論文の C_best 定義
    for r in records:
        r["pooled_best"] = pooled_best
        r["C_best_used"] = C_best
        # best-of-K 最適性ギャップ(seed 平均の best_cut を使用, [0,1] clamp)
        gap_known = (KNOWN_BEST_G22 - min(r["best_cut_mean"], KNOWN_BEST_G22)) / KNOWN_BEST_G22
        gap_pooled = (pooled_best - min(r["best_cut_mean"], pooled_best)) / pooled_best
        r["gap_vs_known"] = float(gap_known)
        r["gap_vs_pooled"] = float(gap_pooled)
        r["pct_of_best"] = float(r["best_cut_max"] / KNOWN_BEST_G22)

    elapsed = time.time() - t_start

    # ----- 出力ディレクトリ -----
    desc = f"8cond_{len(exp.S_test_seeds)}seed_T{cfgs.cim.T_steps}"
    out_dir = get_out_dir(desc)

    # ----- JSON -----
    results = {
        "graph": "G22",
        "known_best": KNOWN_BEST_G22,
        "pooled_best": pooled_best,
        "C_best_used": C_best,
        "hyperparameters": {
            "T_steps": cfgs.cim.T_steps, "dt": cfgs.cim.dt,
            "K_trajectories": cfgs.cim.K_trajectories, "xi": cfgs.cim.xi,
            "sigma": cfgs.cim.sigma, "p0": cfgs.cim.p0, "pf": cfgs.cim.pf,
            "p_max": cfgs.cim.p_max,
            "opt_steps_freeform": cfgs.opt.opt_steps_freeform,
            "optuna_n_trials": cfgs.opt.optuna_n_trials,
            "cmaes_evals": cfgs.opt.cmaes_evals,
            "cac_beta": cfgs.cac.beta, "cac_a": cfgs.cac.a,
            "S_train_seeds": exp.S_train_seeds, "S_test_seeds": exp.S_test_seeds,
        },
        "records": records,
        "raw_per_seed": raw_per_seed,
        "elapsed_seconds": elapsed,
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=float),
                                          encoding="utf-8")

    # ----- CSV(ランキング順) -----
    ranked = sorted(records, key=lambda r: r["gap_vs_known"])
    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "condition", "best_cut_mean", "best_cut_max", "best_cut_std",
                    "mean_cut_mean", "gap_vs_known", "gap_vs_pooled", "pct_of_best",
                    "fit_s", "eval_s"])
        for i, r in enumerate(ranked, 1):
            w.writerow([i, r["condition"], f"{r['best_cut_mean']:.1f}",
                        f"{r['best_cut_max']:.0f}", f"{r['best_cut_std']:.1f}",
                        f"{r['mean_cut_mean']:.1f}", f"{r['gap_vs_known']:.4f}",
                        f"{r['gap_vs_pooled']:.4f}", f"{r['pct_of_best']:.4f}",
                        f"{r['fit_seconds']:.1f}", f"{r['eval_seconds']:.1f}"])

    # ----- 図 -----
    try:
        make_figure(records, out_dir)
    except Exception as e:  # noqa: BLE001
        print(f"FIGURE_WARNING: {e!r}")

    # ----- サマリ表示 -----
    print("\n" + "=" * 70)
    print(f"G22 再検証 完了({elapsed:.1f}s)  既知ベスト={KNOWN_BEST_G22} "
          f"プールド最良={pooled_best:.0f}")
    print("-" * 70)
    print(f"{'順':>2} {'条件':<26} {'best(平均)':>10} {'best(最大)':>10} "
          f"{'gap既知':>8} {'%既知':>7}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>2} {DISP.get(r['condition'], r['condition']):<26} "
              f"{r['best_cut_mean']:>10.1f} {r['best_cut_max']:>10.0f} "
              f"{r['gap_vs_known']:>8.4f} {r['pct_of_best']*100:>6.2f}%")
    print("=" * 70)
    print(f"保存先: {out_dir}")


def make_figure(records, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    ranked = sorted(records, key=lambda r: r["gap_vs_known"])
    labels = [DISP.get(r["condition"], r["condition"]) for r in ranked]
    gaps = [r["gap_vs_known"] for r in ranked]
    bests = [r["best_cut_mean"] for r in ranked]
    errs = [r["best_cut_std"] for r in ranked]
    colors = ["#c0392b" if r["condition"] == "cac_chaotic_amplitude_control"
              else "#2c5f8a" for r in ranked]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.bar(range(len(labels)), gaps, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("最適性ギャップ(既知ベスト基準, 小さいほど良い)")
    ax.set_title("G22 best-of-K 最適性ギャップ")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(range(len(labels)), bests, yerr=errs, color=colors, capsize=3)
    ax.axhline(KNOWN_BEST_G22, color="#e67e22", ls="--", lw=1.5,
               label=f"既知ベスト {KNOWN_BEST_G22}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("best-of-K カット値(平均 ± 標準偏差)")
    ax.set_title("G22 到達カット値")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.suptitle("ARC PumpBench の G22 再検証(学習=合成, 評価=実 G22)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "comparison.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
