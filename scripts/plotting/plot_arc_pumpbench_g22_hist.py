"""ARC PumpBench を G22 で再実行し、8 条件のカット値分布ヒストグラムを描く。

参照スタイル: results/2026-06-08/cim_pt_v3/.../hist.png
 (各手法のカット値分布 + 平均(黒点線) + 既知ベスト(赤破線))。

本図は 8 条件 × 5 seed × K=20 軌道 = 条件あたり 100 試行のカット値を集めて描く。
G22 では条件間で分布位置が大きく離れる(10445〜12801)ため、各パネルは
分布形が見えるよう自前 bin で解像し、xlim を右端 13359 まで延ばして既知ベスト線と
ギャップを可視化する。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_arc_pumpbench_g22_hist.py
"""
from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARC_CODE = ROOT / "AutoResearchClaw" / "artifacts" / "pump_power_opt" / "deliverables" / "code"
sys.path.insert(0, str(ARC_CODE))

import experiment_config as config  # noqa: E402
import data  # noqa: E402
from methods import (  # noqa: E402
    LinearRampPump, OptunaScalarSigmoidPump, CACErrorFeedbackSolver,
    DifferentiableFreeformPump, MonotonicConstrainedPump,
    GradientFreeCMAESFreeformPump, KibbleZurekAnalyticPump,
    SpectralConditionedPumpNet,
)

EXPERIMENT_KIND = "arc_pumpbench_g22"
KNOWN_BEST_G22 = 13359
N_SEEDS_HIST = 5  # 5 seed × K=20 軌道 = 条件あたり 100 試行

CONDITIONS = [
    ("linear_ramp_baseline", LinearRampPump, "線形ランプ", "#7f8c8d"),
    ("kibble_zurek_analytic_pump", KibbleZurekAnalyticPump, "KZ解析ランプ", "#8e44ad"),
    ("cac_chaotic_amplitude_control", CACErrorFeedbackSolver, "CAC(閉ループ)", "#c0392b"),
    ("optuna_scalar_sigmoid_ramp", OptunaScalarSigmoidPump, "シグモイド(調整)", "#2c5f8a"),
    ("differentiable_freeform_pump", DifferentiableFreeformPump, "自由形状FD", "#16a085"),
    ("monotonic_constrained_pump", MonotonicConstrainedPump, "単調自由形状", "#27ae60"),
    ("gradient_free_cmaes_freeform_pump", GradientFreeCMAESFreeformPump, "自由形状CMA-ES", "#d35400"),
    ("spectral_conditioned_pump_net", SpectralConditionedPumpNet, "スペクトル条件付MLP", "#95a5a6"),
]


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


def main() -> None:
    t0 = time.time()
    cfgs = config.get_all()
    train_graphs = data.build_synthetic_factorial(cfgs.data, base_seed=cfgs.exp.S_train_seeds[0])
    g22 = data._read_gset_file(str(ROOT / "input" / "G22.txt"), "G22", KNOWN_BEST_G22)
    print(f"DATA: 合成学習 {len(train_graphs)} / 評価=G22 N={g22.n_nodes} "
          f"E={g22.edges.shape[0]} (各条件 {N_SEEDS_HIST}seed×K={cfgs.cim.K_trajectories}軌道)")

    # 条件ごとに fit(合成)→ G22 で全軌道カットをプール
    pooled = {}   # name -> np.array (N_SEEDS_HIST * K,)
    meta = {}     # name -> (label, color)
    for name, Cls, label, color in CONDITIONS:
        cond = Cls(cfgs)
        cond.deadline = None
        t_fit = time.time()
        try:
            cond.fit(train_graphs, np.random.default_rng(cfgs.exp.S_train_seeds[0]))
        except Exception as e:  # noqa: BLE001
            print(f"  FIT_WARNING {name}: {e!r}")
        cuts = []
        for seed in range(N_SEEDS_HIST):
            r = cond.run_on_graph(g22, seed)
            cuts.append(np.asarray(r["per_traj"], dtype=float))
        arr = np.concatenate(cuts)
        pooled[name] = arr
        meta[name] = (label, color)
        print(f"  {label:<20} n={arr.size} 平均={arr.mean():.1f} 最良={arr.max():.0f} "
              f"std={arr.std():.1f} (fit+eval {time.time()-t_fit:.1f}s)")

    # 性能順(平均カット降順)に並べる
    order = sorted(pooled, key=lambda k: -pooled[k].mean())

    out_dir = get_out_dir(f"hist_{N_SEEDS_HIST}seed_T{cfgs.cim.T_steps}")
    np.savez(out_dir / "pooled_cuts.npz", **pooled)
    make_hist(pooled, meta, order, out_dir)
    print(f"\n保存先: {out_dir}  ({time.time()-t0:.1f}s)")


def make_hist(pooled, meta, order, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 13

    ncol = 4
    nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow))
    axes = axes.ravel()

    x_right = KNOWN_BEST_G22 + 60
    for ax, name in zip(axes, order):
        c = pooled[name]
        label, color = meta[name]
        m = max(8.0, c.std() * 0.6)
        bins = np.linspace(c.min() - m, c.max() + m, 26)  # 分布形を解像
        ax.hist(c, bins=bins, color=color, alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axvline(c.mean(), color="black", linestyle=":", linewidth=1.3,
                   label=f"平均 {c.mean():.0f}")
        ax.axvline(KNOWN_BEST_G22, color="red", linestyle="--", linewidth=1.3,
                   label=f"既知ベスト {KNOWN_BEST_G22}")
        ax.set_title(f"{label}\n平均:{c.mean():.0f} 最良:{c.max():.0f} "
                     f"(既知比 {c.max()/KNOWN_BEST_G22*100:.1f}%)", fontsize=10)
        ax.set_xlabel("カット値", fontsize=LABEL_FS)
        ax.set_ylabel("頻度", fontsize=LABEL_FS)
        ax.set_xlim(c.min() - 4 * m, x_right)  # 右端=既知ベストでギャップを可視化
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, loc="upper left")

    for ax in axes[len(order):]:
        ax.axis("off")

    fig.suptitle(
        f"ARC PumpBench 8 条件のカット値分布 — G22 "
        f"(学習=合成 / 評価=実 G22, 各 {N_SEEDS_HIST*20} 試行=5seed×20軌道)",
        fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'hist.png'}")


if __name__ == "__main__":
    main()
