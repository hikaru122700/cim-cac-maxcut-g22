"""warm_ga_q3.py — 質問3「GA の初期解を CIM 解で代用すると改善するか」検証。

cold-GA(乱数初期集団)と CIM-seeded GA(各 run の集団 1 個体目に CIM 解を注入)を
同一の世代予算・同一 seed で比較し、(実時間, 最良重み付きカット)を記録する。
データセット: G22(GA が単体で BKS 到達する疎)と K2000(密。物理ソルバ救済が顕著)。

出力: <paper>/qa_v2/warm_ga_result_v1.json, <paper>/qa_v2/warm_ga_cold_vs_warm_v1.png
使い方: python scripts/benchmarks/warm_ga_q3.py
"""
from __future__ import annotations

import os
os.environ.setdefault("NUMBA_NUM_THREADS", "4")  # スレッド過剰購読対策(規約)

import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "benchmarks"))
os.chdir(ROOT)  # input/ や results/ への相対パス解決のため

import algo_registry as reg
from modules.GA import simulate_ga_batch

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

OUT = ROOT / "results/2026-06-19/maxcut_algo_combo/v1_full/paper/qa_v2"
OUT.mkdir(parents=True, exist_ok=True)

NUM_TRIALS = 16
GA_BUDGETS = [3, 8, 20, 50]   # 世代数(anytime グリッドと同じ刻み)
CIM_SEED_BUDGET = 600         # 探索器としての短い CIM(ウォームスタート用 seed)
DATASETS = ["G22", "K2000"]


def run_ga_sweep(ctx, init_signs, seeds):
    """各予算で独立に GA を回し (time, best_weighted_cut) を返す。"""
    out = []
    for b in GA_BUDGETS:
        p = reg.PARAMS[ctx.name]["GA"]
        t0 = time.perf_counter()
        _, signs = simulate_ga_batch(
            ctx.n, ctx.edges, ctx.weights, NUM_TRIALS,
            pop_size=p["pop_size"], max_generations=int(b),
            ts_iters=p["ts_iters"], cr=p["cr"], alpha_tenure=p["alpha_tenure"],
            beta_quality=p["beta_quality"], init_signs=init_signs, seeds=seeds,
        )
        dt = time.perf_counter() - t0
        cut = int(ctx.score(signs).max())  # 統一重み付き採点
        out.append({"budget": b, "time": dt, "cut_max": cut,
                    "gap": ctx.bks - cut})
        print(f"    budget={b:4d}  t={dt:6.2f}s  cut={cut}  gap={ctx.bks - cut}")
    return out


def main():
    # JIT ウォームアップ(コンパイル時間を計測から除外)
    print("[warmup] JIT compile ...")
    ctx0 = reg.load_context("G22")
    seeds_w = np.arange(2, dtype=np.int64)
    simulate_ga_batch(ctx0.n, ctx0.edges, ctx0.weights, 2,
                      pop_size=4, max_generations=2, ts_iters=2000, seeds=seeds_w)
    cw, sw = reg.run_cim(ctx0, 150, 2, seeds_w)

    results = {"meta": {"num_trials": NUM_TRIALS, "ga_budgets": GA_BUDGETS,
                        "cim_seed_budget": CIM_SEED_BUDGET}, "data": {}}
    seeds = np.arange(NUM_TRIALS, dtype=np.int64)

    for ds in DATASETS:
        print(f"\n==== {ds} ====")
        ctx = reg.load_context(ds)

        # --- CIM 探索器で seed を作る ---
        t0 = time.perf_counter()
        _, cim_signs = reg.run_cim(ctx, CIM_SEED_BUDGET, NUM_TRIALS, seeds)
        cim_time = time.perf_counter() - t0
        cim_cuts = ctx.score(cim_signs)
        cim_best = int(cim_cuts.max())
        print(f"  CIM seed: budget={CIM_SEED_BUDGET} t={cim_time:.2f}s "
              f"best={cim_best} gap={ctx.bks - cim_best}")

        print("  [cold GA]")
        cold = run_ga_sweep(ctx, None, seeds)
        print("  [warm GA (CIM-seeded)]")
        warm = run_ga_sweep(ctx, cim_signs, seeds)

        results["data"][ds] = {
            "bks": ctx.bks,
            "cim_seed": {"budget": CIM_SEED_BUDGET, "time": cim_time,
                         "cut_max": cim_best, "gap": ctx.bks - cim_best},
            "cold": cold,
            "warm": warm,
        }

    (OUT / "warm_ga_result_v1.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 図: gap 対 実時間(cold vs warm) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ds in zip(axes, DATASETS):
        d = results["data"][ds]
        for key, color, lab in [("cold", "#888888", "cold GA(乱数初期)"),
                                ("warm", "#d81b9e", "CIM→GA(CIM 解で初期化)")]:
            xs = [r["time"] for r in d[key]]
            ys = [max(r["gap"], 0.5) for r in d[key]]
            ax.plot(xs, ys, "-o", color=color, lw=1.8, ms=6, label=lab)
        # CIM seed 単体の位置
        cs = d["cim_seed"]
        ax.scatter([cs["time"]], [max(cs["gap"], 0.5)], marker="*", s=180,
                   color="#e74c3c", zorder=5, label=f"CIM seed 単体(budget={cs['budget']})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{ds}  (BKS={d['bks']})", fontsize=12)
        ax.set_xlabel("実時間 [秒]  (best-of-16, 4スレッド)")
        ax.set_ylabel("BKS との絶対 gap (小さいほど良い)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8.5)
    fig.suptitle("質問3: GA の初期集団を CIM 解で代用した場合の gap-時間推移", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "warm_ga_cold_vs_warm_v1.png", dpi=130)
    plt.close(fig)
    print("\n[OK] wrote", OUT / "warm_ga_result_v1.json")
    print("[OK] wrote", OUT / "warm_ga_cold_vs_warm_v1.png")


if __name__ == "__main__":
    main()
