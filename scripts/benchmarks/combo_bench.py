"""combo_bench.py — ハイブリッド(ウォームスタート)組合せの検証。

物理ソルバ(CIM/CAC)で高速に良い初期スピン配置を作り、それを古典局所探索
(Tabu Search=GA の局所探索 / warm-start SA)の初期値にして磨く。同一の
「精錬予算」で、乱数初期からの cold-start と比較し、ウォームスタートが
(a) より良いスコア / (b) より短時間で同等、をもたらすか検証する。

検証する組合せ:
  探索器 explorer ∈ {CIM, CAC}
  精錬器 refiner  ∈ {TS(memetic 局所探索), SA(warm-start)}
  → CIM→TS, CAC→TS, CIM→SA, CAC→SA
ベースライン:
  - refiner を乱数初期から(cold-start TS / cold-start SA)
  - explorer 単体(精錬なし)

各 refiner 予算 b_ref で:
  hybrid 総時間 = t_explore(固定) + t_refine(b_ref)
  cold   総時間 =                    t_refine(b_ref)
全カットは ctx.score() の統一重み付きカットで採点。

実行:
  python scripts/benchmarks/combo_bench.py --dataset G22 --num-trials 16
出力: results/<date>/combo_hybrid/v{N}_<desc>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import (
    DATASETS, load_context, run_cim, run_cac, PARAMS,
)
from modules.GA import tabu_refine_batch
from modules.SA import simulate_sa_warm, simulate_sa_batch

EXPERIMENT_KIND = "combo_hybrid"
MAX_REFINE_SEC = 120.0   # 1 精錬バッチがこれを超えたら以後の大予算をスキップ

# explorer の固定予算(短く回して良い初期解を得る)
EXPLORER_BUDGET = {"CIM": 800, "CAC": 4000}
# refiner の予算グリッド
REFINER_GRID = {
    "TS": [2000, 8000, 30000, 100000, 300000],     # ts_iters
    "SA": [100000, 300000, 1000000, 3000000, 10000000],  # num_iters
}


def get_out_dir(desc):
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(p.name.split("_", 1)[0][1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def refine_TS(ctx, init_signs, budget, num_trials, seeds):
    cuts, signs = tabu_refine_batch(
        ctx.n, ctx.edges, ctx.weights, init_signs.astype(np.int8),
        ts_iters=int(budget), seeds=seeds,
    )
    return signs


def refine_SA_warm(ctx, init_signs, budget, num_trials, seeds):
    cuts, signs = simulate_sa_warm(
        ctx.n, ctx.edges, ctx.weights, init_signs, int(budget),
        t_start=0.5, t_end=0.001, seeds=seeds,
    )
    return signs


def cold_TS(ctx, budget, num_trials, seeds):
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 2, (num_trials, ctx.n)).astype(np.int8)
    _, signs = tabu_refine_batch(
        ctx.n, ctx.edges, ctx.weights, rand, ts_iters=int(budget), seeds=seeds)
    return signs


def cold_SA(ctx, budget, num_trials, seeds):
    _, signs = simulate_sa_batch(
        ctx.n, ctx.edges, ctx.weights, int(budget), num_trials,
        t_start=2.0, t_end=0.001, seeds=seeds)
    return signs


REFINERS = {
    "TS": {"warm": refine_TS, "cold": cold_TS},
    "SA": {"warm": refine_SA_warm, "cold": cold_SA},
}
EXPLORERS = {"CIM": run_cim, "CAC": run_cac}


def time_score(fn, *a):
    t0 = time.perf_counter()
    signs = fn(*a)
    dt = time.perf_counter() - t0
    return signs, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22", choices=list(DATASETS.keys()))
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--explorers", nargs="+", default=["CIM", "CAC"])
    ap.add_argument("--refiners", nargs="+", default=["TS", "SA"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ds = args.dataset
    nt = args.num_trials
    seeds = np.arange(nt, dtype=np.int64)
    desc = f"{ds}_nt{nt}" + (f"_{args.tag}" if args.tag else "")
    out_dir = get_out_dir(desc)

    def log(m):
        print(m, flush=True)
        with open(out_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(m + "\n")

    ctx = load_context(ds)
    bks = ctx.bks
    log(f"=== combo_hybrid {ds} n={ctx.n} bks={bks} num_trials={nt} ===")

    # --- explorer を固定予算で 1 回(初期解を得る)。warmup 込みで2回目を計測 ---
    explorer_out = {}
    for ex in args.explorers:
        b = EXPLORER_BUDGET[ex]
        EXPLORERS[ex](ctx, b, nt, seeds)  # warmup(JIT)
        # 計測
        t0 = time.perf_counter()
        _c, signs = EXPLORERS[ex](ctx, b, nt, seeds)
        t_exp = time.perf_counter() - t0
        sc = ctx.score(signs)
        explorer_out[ex] = {"signs": signs, "t": t_exp,
                            "cut_max": float(sc.max()), "cut_mean": float(sc.mean())}
        log(f"[explorer] {ex} b={b} t={t_exp:.2f}s cut_max={int(sc.max())} "
            f"mean={sc.mean():.1f} gap={bks - sc.max():.1f}")

    results = {"meta": {"dataset": ds, "num_trials": nt,
                        "explorer_budget": EXPLORER_BUDGET,
                        "refiner_grid": REFINER_GRID, "bks": bks},
               "explorers": {k: {kk: vv for kk, vv in v.items() if kk != "signs"}
                             for k, v in explorer_out.items()},
               "runs": {}}

    # --- 各 refiner: cold(乱数) と warm(各 explorer) の anytime ---
    for rf in args.refiners:
        grid = REFINER_GRID[rf]
        log(f"\n[refiner] {rf} grid={grid}")
        # warmup refiner JIT
        REFINERS[rf]["cold"](ctx, grid[0], nt, seeds)
        REFINERS[rf]["warm"](ctx, explorer_out[args.explorers[0]]["signs"],
                             grid[0], nt, seeds)

        # cold-start
        cold_pts = []
        for b in grid:
            signs, dt = time_score(REFINERS[rf]["cold"], ctx, b, nt, seeds)
            sc = ctx.score(signs)
            cold_pts.append({"budget": b, "time": dt, "total_time": dt,
                             "cut_max": float(sc.max()), "cut_mean": float(sc.mean())})
            log(f"  {rf}-cold      b={b:>9} t={dt:6.2f}s max={int(sc.max()):6d} "
                f"mean={sc.mean():9.1f} gap={bks - sc.max():.1f}")
            if dt > MAX_REFINE_SEC:
                log(f"  {rf}-cold batch>{MAX_REFINE_SEC}s → 以後スキップ")
                break
        results["runs"][f"{rf}_cold"] = cold_pts

        # warm-start from each explorer
        for ex in args.explorers:
            warm_pts = []
            ex_signs = explorer_out[ex]["signs"]
            t_exp = explorer_out[ex]["t"]
            for b in grid:
                signs, dt = time_score(REFINERS[rf]["warm"], ctx, ex_signs, b, nt, seeds)
                sc = ctx.score(signs)
                warm_pts.append({"budget": b, "time": dt,
                                 "total_time": dt + t_exp,
                                 "cut_max": float(sc.max()),
                                 "cut_mean": float(sc.mean())})
                log(f"  {ex}→{rf}     b={b:>9} t={dt:6.2f}s(+{t_exp:.2f}) "
                    f"max={int(sc.max()):6d} mean={sc.mean():9.1f} gap={bks - sc.max():.1f}")
                if dt > MAX_REFINE_SEC:
                    log(f"  {ex}→{rf} batch>{MAX_REFINE_SEC}s → 以後スキップ")
                    break
            results["runs"][f"{ex}_{rf}"] = warm_pts

        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)

    # --- プロット(refiner ごとに warm vs cold) ---
    plt = setup_style()
    colors = {"cold": "#888888", "CIM": "#e74c3c", "CAC": "#e67e22"}
    for rf in args.refiners:
        fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=140)
        cold = results["runs"][f"{rf}_cold"]
        ax.plot([p["total_time"] for p in cold], [p["cut_max"] for p in cold],
                "-o", color=colors["cold"], lw=2, ms=5, label=f"{rf} 乱数初期(cold)")
        for ex in args.explorers:
            pts = results["runs"][f"{ex}_{rf}"]
            ax.plot([p["total_time"] for p in pts], [p["cut_max"] for p in pts],
                    "-o", color=colors[ex], lw=2, ms=5, label=f"{ex}→{rf} (warm)")
            # explorer 単体点
            ax.plot([explorer_out[ex]["t"]], [explorer_out[ex]["cut_max"]],
                    "*", color=colors[ex], ms=14, label=f"{ex} 単体")
        ax.axhline(bks, color="k", ls=":", lw=1.3, label=f"BKS={bks}")
        ax.set_xscale("log")
        ax.set_xlabel("総実時間 [秒]（explorer + refiner, log）")
        ax.set_ylabel("最大カット値")
        ax.set_title(f"{ds}: {rf} のウォームスタート vs 乱数初期")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_{rf}_warm_vs_cold.png", bbox_inches="tight")
        plt.close(fig)

    log(f"\nsaved → {out_dir}")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
