"""reproduce_ga_memetic.py — MAX-CUT 向け GA(メメティック)先行研究の再現実走。

再現対象(先行研究):
  - Q. Wu, J.-K. Hao, "A Memetic Approach for the Max-Cut Problem," PPSN XII,
    LNCS 7492:297-306, 2012.            ← MACUT(グルーピング交叉 + 摂動付き TS)
  - Q. Wu, J.-K. Hao, "Memetic search for the max-bisection problem,"
    Computers & Operations Research 40(1):166-179, 2013.   ← DisQual プール更新
  - Q. Wu, Y. Wang, Z. Lü, "A tabu search based hybrid evolutionary algorithm for
    the max-cut problem," Applied Soft Computing 34:827-837, 2015.  ← TSHEA

これらは G-set ベンチで MAX-CUT のメメティック法を確立した代表的研究で、G22 の
最良カットとして 13359 を報告している(Benlic & Hao の Breakout Local Search も同値)。

評価プロトコル(先行研究にならう):
  - 1 グラフにつき num_trials 個の独立 run(既定 20)
  - 各 run の最終 best カットから best / mean / worst / 標準偏差を集計
  - BKS 到達率(== BKS の run 割合)と平均 gap、平均実時間を報告
  - 収束履歴(世代 × best カット)を記録 → 収束曲線
  - 最終 best カットの分布 → ヒストグラム

本実装は modules/GA.py(同 3 研究を制約なし MAX-CUT 向けに具現化したもの)を呼ぶ。

実行:
  python scripts/benchmarks/reproduce_ga_memetic.py --dataset G22 --num-trials 20
出力: results/<YYYY-MM-DD>/ga_memetic_reproduce/v{N}_<desc>/
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

from modules.GA import load_graph, simulate_ga_batch

EXPERIMENT_KIND = "ga_memetic_reproduce"

# 先行研究が報告する G-set の最良既知カット(BKS)。
BKS = {
    "G22": 13359, "G1": 11624, "G14": 3064, "G15": 3050,
    "G23": 13344, "G32": 1410, "G55": 10299, "G70": 9591,
}

# Wu & Hao / TSHEA 系の標準設定(modules/GA.py の既定に対応)。
DEFAULT_GA = dict(
    pop_size=10, ts_iters=20000, cr=3000, alpha_tenure=15, beta_quality=0.6,
)


def get_out_dir(desc: str) -> Path:
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
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


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def build_description(args) -> str:
    parts = [args.dataset, f"nt{args.num_trials}", f"gen{args.generations}"]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="G22", choices=list(BKS.keys()))
    ap.add_argument("--num-trials", type=int, default=20,
                    help="独立 run 数(先行研究は 20 が標準)")
    ap.add_argument("--generations", type=int, default=50,
                    help="メメティック世代数")
    ap.add_argument("--ts-iters", type=int, default=DEFAULT_GA["ts_iters"])
    ap.add_argument("--pop-size", type=int, default=DEFAULT_GA["pop_size"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ds = args.dataset
    path = f"input/{ds}.txt"
    bks = BKS[ds]
    out_dir = get_out_dir(build_description(args))

    def log(m: str):
        print(m, flush=True)
        with open(out_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(m + "\n")

    n, edges, weights = load_graph(path, return_weights=True)
    # G-set は非重み(全辺 +1)。重み付きインスタンスのみ weights を渡す。
    if all(abs(w - 1.0) < 1e-12 for w in weights):
        weights = None
    log(f"=== {EXPERIMENT_KIND} {ds} n={n} edges={len(edges)} bks={bks} ===")
    log(f"GA(memetic): pop_size={args.pop_size} ts_iters={args.ts_iters} "
        f"cr={DEFAULT_GA['cr']} alpha_tenure={DEFAULT_GA['alpha_tenure']} "
        f"beta_quality={DEFAULT_GA['beta_quality']} generations={args.generations}")
    log(f"独立 run 数 num_trials={args.num_trials}")

    seeds = np.arange(args.num_trials, dtype=np.int64)

    # --- JIT ウォームアップ(計測から除外) ---
    log("\n[warmup] JIT コンパイル中 ...")
    simulate_ga_batch(
        n, edges, weights, num_trials=2, pop_size=args.pop_size,
        max_generations=2, ts_iters=2000, cr=DEFAULT_GA["cr"],
        alpha_tenure=DEFAULT_GA["alpha_tenure"],
        beta_quality=DEFAULT_GA["beta_quality"], seeds=np.arange(2),
    )

    # --- 本計測(num_trials 独立 run + 収束履歴) ---
    log("[run] 計測中 ...")
    t0 = time.perf_counter()
    best_cuts, best_signs, history = simulate_ga_batch(
        n, edges, weights, num_trials=args.num_trials, pop_size=args.pop_size,
        max_generations=args.generations, ts_iters=args.ts_iters,
        cr=DEFAULT_GA["cr"], alpha_tenure=DEFAULT_GA["alpha_tenure"],
        beta_quality=DEFAULT_GA["beta_quality"], seeds=seeds,
        return_history=True,
    )
    elapsed = time.perf_counter() - t0

    # --- 集計 ---
    best = int(best_cuts.max())
    mean = float(best_cuts.mean())
    worst = int(best_cuts.min())
    std = float(best_cuts.std())
    hits = int(np.sum(best_cuts >= bks))
    hit_rate = hits / args.num_trials
    gap_best = bks - best
    gap_mean = bks - mean
    per_run_sec = elapsed / args.num_trials

    log(f"\n--- 結果({args.num_trials} run, {elapsed:.1f}s 計, "
        f"{per_run_sec:.1f}s/run) ---")
    log(f"  best  cut = {best:6d}   (gap {gap_best:+d})")
    log(f"  mean  cut = {mean:9.2f} (gap {gap_mean:+.2f})")
    log(f"  worst cut = {worst:6d}")
    log(f"  std       = {std:.2f}")
    log(f"  BKS({bks}) 到達率 = {hits}/{args.num_trials} = {hit_rate:.0%}")
    log(f"  各 run best: {sorted(int(c) for c in best_cuts)}")

    results = {
        "experiment": EXPERIMENT_KIND,
        "dataset": ds, "n": n, "edges": len(edges), "bks": bks,
        "ga_params": {
            "pop_size": args.pop_size, "ts_iters": args.ts_iters,
            "cr": DEFAULT_GA["cr"], "alpha_tenure": DEFAULT_GA["alpha_tenure"],
            "beta_quality": DEFAULT_GA["beta_quality"],
            "generations": args.generations,
        },
        "num_trials": args.num_trials,
        "elapsed_sec": elapsed, "per_run_sec": per_run_sec,
        "best": best, "mean": mean, "worst": worst, "std": std,
        "hits": hits, "hit_rate": hit_rate,
        "gap_best": gap_best, "gap_mean": gap_mean,
        "per_run_best": [int(c) for c in best_cuts],
        "history_mean": [float(np.mean(h)) for h in history],
        "history_best": [float(np.max(h)) for h in history],
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    # --- 図1: 収束曲線(世代 × カット) ---
    plt = setup_style()
    gens = np.arange(history.shape[0])
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=140)
    hist_mean = history.mean(axis=1)
    hist_best = history.max(axis=1)
    hist_min = history.min(axis=1)
    ax.plot(gens, hist_best, "-o", color="#27ae60", lw=2, ms=4,
            label="全 run 最良")
    ax.plot(gens, hist_mean, "-s", color="#2980b9", lw=2, ms=4,
            label="run 平均")
    ax.fill_between(gens, hist_min, hist_best, color="#27ae60", alpha=0.12,
                    label="run 最小〜最大")
    ax.axhline(bks, color="k", ls=":", lw=1.3, label=f"BKS={bks}")
    ax.set_xlabel("世代")
    ax.set_ylabel("カット値")
    ax.set_title(f"{ds}: メメティック GA の収束(Wu & Hao 系)")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", bbox_inches="tight")
    plt.close(fig)

    # --- 図2: 最終 best カットの分布ヒストグラム ---
    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=140)
    lo = min(int(best_cuts.min()), bks) - 2
    hi = max(int(best_cuts.max()), bks) + 2
    bins = np.arange(lo, hi + 1) - 0.5
    ax.hist(best_cuts, bins=bins, color="#27ae60", alpha=0.8,
            edgecolor="white")
    ax.axvline(bks, color="k", ls=":", lw=1.5, label=f"BKS={bks}")
    ax.set_xlabel("各 run の最終 best カット値")
    ax.set_ylabel("run 数")
    ax.set_title(f"{ds}: {args.num_trials} run の到達カット分布")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both", axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "hist.png", bbox_inches="tight")
    plt.close(fig)

    log(f"\nsaved → {out_dir}")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
