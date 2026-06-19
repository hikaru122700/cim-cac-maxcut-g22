"""anytime_bench.py — 単体アルゴリズムの「実時間 × 最大カット」anytime 曲線。

各アルゴリズム(CIM/CAC/SA/SB/PT-ICM/GA)を、その主要計算量ノブの幾何グリッドで
走らせ、各点で (バッチ実時間, カット統計) を記録する。x=実時間(log)、
y=最大カット(バッチ最良) と平均カットを重ね描き、BKS 線も引く。

予算は適応的に打ち切る: あるバッチが MAX_BATCH_SEC を超えたら、その algo の
より大きな予算はスキップ(G55/G70/K2000 の暴走防止)。

実行:
  python scripts/benchmarks/anytime_bench.py --datasets G22 --num-trials 16
  python scripts/benchmarks/anytime_bench.py --datasets G22 K2000 G55 G70 --num-trials 16 --tag full

出力: results/<date>/anytime_single/v{N}_<desc>/
  <DATASET>_anytime.png, <DATASET>_gap.png, results.json, raw.npz, combined.png
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
    DATASETS, ALGOS, BUDGET_GRIDS, load_context,
)

EXPERIMENT_KIND = "anytime_single"
MAX_BATCH_SEC_DEFAULT = 45.0   # 1 バッチがこれを超えたら以後の大予算をスキップ


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    mx = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            h = p.name.split("_", 1)[0]
            if h[1:].isdigit():
                mx = max(mx, int(h[1:]))
    return mx + 1


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def run_algo_sweep(ctx, algo_key, num_trials, max_batch_sec, log):
    """1 algo を予算グリッドでスイープ。各点 dict のリストを返す。"""
    run = ALGOS[algo_key]["run"]
    seeds = np.arange(num_trials, dtype=np.int64)
    grid = BUDGET_GRIDS[algo_key]

    # --- JIT ウォームアップ(最小予算で 1 回、計測しない) ---
    try:
        run(ctx, grid[0], num_trials, seeds)
    except Exception as e:  # noqa: BLE001
        log(f"    [WARN] {algo_key} warmup failed: {e}")
        return []

    points = []
    for b in grid:
        t0 = time.perf_counter()
        try:
            _cuts, signs = run(ctx, b, num_trials, seeds)
        except Exception as e:  # noqa: BLE001
            log(f"    [WARN] {algo_key} b={b} failed: {type(e).__name__}: {e} → skip rest")
            break
        dt = time.perf_counter() - t0
        # 全手法を統一の重み付きカットで再採点(CAC の非重みカウント等を補正)
        cuts = np.asarray(ctx.score(signs), dtype=float)
        pt = {
            "budget": float(b),
            "time": dt,
            "cut_mean": float(cuts.mean()),
            "cut_max": float(cuts.max()),
            "cut_min": float(cuts.min()),
            "cut_std": float(cuts.std()),
            "cut_p25": float(np.percentile(cuts, 25)),
            "cut_p75": float(np.percentile(cuts, 75)),
        }
        points.append(pt)
        gap = ctx.bks - pt["cut_max"]
        log(f"    {algo_key:4s} b={b:>9} t={dt:6.2f}s  mean={pt['cut_mean']:9.1f}  "
            f"max={int(pt['cut_max']):6d}  gap={gap:6.1f}")
        if dt > max_batch_sec:
            log(f"    {algo_key:4s} batch>{max_batch_sec}s → 以後の大予算スキップ")
            break
    return points


def plot_dataset(plt, ds, data, out_dir):
    """1 データセットの anytime 曲線(絶対カット)と gap(%) 図。"""
    bks = DATASETS[ds]["bks"]

    # --- 絶対カット ---
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=140)
    for algo_key, pts in data.items():
        if not pts:
            continue
        t = [p["time"] for p in pts]
        cmax = [p["cut_max"] for p in pts]
        cmean = [p["cut_mean"] for p in pts]
        c = ALGOS[algo_key]["color"]
        lab = ALGOS[algo_key]["label"]
        ax.plot(t, cmax, "-o", color=c, lw=2.0, ms=5, label=f"{lab} (最良)")
        ax.plot(t, cmean, "--", color=c, lw=1.0, alpha=0.6)
    ax.axhline(bks, color="k", ls=":", lw=1.4, label=f"BKS={bks}")
    ax.set_xscale("log")
    ax.set_xlabel("実時間 [秒]（バッチ, log）")
    ax.set_ylabel("カット値")
    ax.set_title(f"{ds}: 単体アルゴリズムの実時間 × 最大カット（実線=最良, 破線=平均）")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{ds}_anytime.png", bbox_inches="tight")
    plt.close(fig)

    # --- BKS への gap(%) ---
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=140)
    for algo_key, pts in data.items():
        if not pts:
            continue
        t = [p["time"] for p in pts]
        gap = [100.0 * (bks - p["cut_max"]) / bks for p in pts]
        c = ALGOS[algo_key]["color"]
        ax.plot(t, gap, "-o", color=c, lw=2.0, ms=5, label=ALGOS[algo_key]["label"])
    ax.axhline(0.0, color="k", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_xlabel("実時間 [秒]（バッチ, log）")
    ax.set_ylabel("BKS への gap [%]（最良解, 小さいほど良い）")
    ax.set_title(f"{ds}: BKS への到達 gap の時間推移")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / f"{ds}_gap.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["G22"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS.keys()),
                    choices=list(ALGOS.keys()))
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--max-batch-sec", type=float, default=MAX_BATCH_SEC_DEFAULT)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    kind_root = get_kind_root()
    v = next_version(kind_root)
    desc_parts = ["_".join(args.datasets), f"nt{args.num_trials}"]
    if args.tag:
        desc_parts.append(args.tag)
    out_dir = kind_root / f"v{v}_{'_'.join(desc_parts)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)
        with open(out_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== anytime_single v{v} ===")
    log(f"datasets={args.datasets} algos={args.algos} num_trials={args.num_trials}")

    all_results = {}
    for ds in args.datasets:
        log(f"\n[{ds}] loading...")
        ctx = load_context(ds)
        log(f"[{ds}] n={ctx.n} edges={len(ctx.edges)} bks={ctx.bks}")
        ds_data = {}
        for algo_key in args.algos:
            log(f"  --- {algo_key} ---")
            pts = run_algo_sweep(ctx, algo_key, args.num_trials,
                                 args.max_batch_sec, log)
            ds_data[algo_key] = pts
        all_results[ds] = ds_data
        # 逐次保存(長時間 run の途中確認用)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"meta": vars(args), "results": all_results}, f,
                      ensure_ascii=False, indent=1)

    # --- プロット ---
    plt = setup_style()
    for ds in args.datasets:
        plot_dataset(plt, ds, all_results[ds], out_dir)

    # --- 統合グリッド ---
    nds = len(args.datasets)
    if nds > 1:
        ncol = 2
        nrow = (nds + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 5.2 * nrow), dpi=130)
        axes = np.atleast_1d(axes).ravel()
        for i, ds in enumerate(args.datasets):
            ax = axes[i]
            bks = DATASETS[ds]["bks"]
            for algo_key, pts in all_results[ds].items():
                if not pts:
                    continue
                t = [p["time"] for p in pts]
                gap = [100.0 * (bks - p["cut_max"]) / bks for p in pts]
                ax.plot(t, gap, "-o", color=ALGOS[algo_key]["color"],
                        lw=1.8, ms=4, label=ALGOS[algo_key]["label"])
            ax.axhline(0.0, color="k", ls=":", lw=1.0)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_title(f"{ds} (BKS={bks})")
            ax.set_xlabel("実時間 [秒]")
            ax.set_ylabel("BKS gap [%]")
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(alpha=0.25, which="both")
            if i == 0:
                ax.legend(fontsize=7, ncol=2)
        for j in range(nds, len(axes)):
            axes[j].axis("off")
        fig.suptitle("単体アルゴリズム anytime 比較（BKS gap の時間推移）", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "combined.png", bbox_inches="tight")
        plt.close(fig)

    log(f"\nsaved → {out_dir}")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
