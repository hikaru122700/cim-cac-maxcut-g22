"""diversity_refine.py — 各手法の出力解を「同一の局所探索」で磨いてから多様性を測る。

素の多様性は品質と交絡する(悪い解ほど散らばって見える)。そこで
diversity_bench.py が保存した解を、**全手法共通の Tabu Search**(摂動なし、
同一反復数)で局所最適まで落としてから、同じ指標を測り直す。

  - 磨いた後も解が散らばる → その手法は本当に別々の谷を見つけている
  - 磨くと 1 点に潰れる     → 見かけの多様性は「収束不足のゆらぎ」だった

使い方:
    .venv/Scripts/python.exe -u scripts/benchmarks/diversity_refine.py \
        results/<date>/solution_diversity/v{N}_... [--ts-iters 2000]

出力(同じ run ディレクトリ):
    signs_refined.npz / results_refined.json / refine.log
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, load_context
from scripts.benchmarks.diversity_metrics import summarize
from modules.GA import tabu_refine_batch

NO_PERTURB = 10 ** 9   # cr をこの値にすると摂動が発動しない(純粋な TS 降下)


def _travel_distance(S0: np.ndarray, S1: np.ndarray) -> float:
    """磨く前後で解がどれだけ動いたか(反転対称の正規化ハミング距離の平均)。"""
    n = S0.shape[1]
    h = (S0 != S1).sum(axis=1).astype(float)
    return float(np.minimum(h, n - h).mean() / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--ts-iters", type=int, default=2000)
    ap.add_argument("--alpha-tenure", type=int, default=15)
    ap.add_argument("--out-suffix", default="_refined",
                    help="出力ファイル名の接尾辞(signs<suffix>.npz など)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "results.json", encoding="utf-8") as f:
        payload = json.load(f)
    meta, results = payload["meta"], payload["results"]
    npz = np.load(run_dir / "signs.npz")

    sfx = args.out_suffix
    logf = open(run_dir / f"refine{sfx}.log", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"refine: ts_iters={args.ts_iters} (摂動なし: cr={NO_PERTURB})")

    arrays = {}
    out = {}
    for ds in meta["datasets"]:
        if ds not in results or not results[ds]:
            continue
        ctx = load_context(ds)
        out[ds] = {}
        log(f"\n=== {ds} (BKS={ctx.bks}) ===")
        for algo in meta["algos"]:
            key = f"{ds}__{algo}_signs"
            if key not in npz:
                continue
            S0 = npz[key]
            t0 = time.perf_counter()
            _cuts, signs = tabu_refine_batch(
                ctx.n, ctx.edges, ctx.weights, S0,
                ts_iters=args.ts_iters, cr=NO_PERTURB,
                alpha_tenure=args.alpha_tenure,
                seeds=np.arange(S0.shape[0], dtype=np.int64),
            )
            dt = time.perf_counter() - t0
            cuts = np.asarray(ctx.score(signs), dtype=float)
            S = (np.asarray(signs) > 0)
            S0b = (np.asarray(S0) > 0).astype(np.int8)
            travel = _travel_distance(S0b, S.astype(np.int8))
            arrays[f"{ds}__{algo}_signs"] = S
            arrays[f"{ds}__{algo}_cuts"] = cuts
            summ = summarize(S, cuts, ctx.bks,
                             near_gap_pct=meta.get("near_gap_pct", 0.5))
            base = results[ds][algo]
            summ.update({"budget": base["budget"], "time": base["time"],
                         "refine_time": float(dt),
                         "gap_pct_mean_before": base["gap_pct_mean"],
                         "mean_pairwise_before": base["mean_pairwise"],
                         "travel_dist": float(travel)})
            out[ds][algo] = summ
            log(f"  {algo:4s} 磨き {dt:6.2f}s  gap平均 {base['gap_pct_mean']:.3f}%"
                f" → {summ['gap_pct_mean']:.3f}%   平均距離 "
                f"{base['mean_pairwise']:.4f} → {summ['mean_pairwise']:.4f}  "
                f"相異なる解 {summ['n_distinct']}/{summ['num_trials']}  "
                f"移動距離 {travel:.4f}")

    np.savez_compressed(run_dir / f"signs{sfx}.npz", **arrays)
    meta2 = dict(meta)
    meta2.update({"stage": "refined", "refine_ts_iters": args.ts_iters,
                  "refine_alpha_tenure": args.alpha_tenure})
    with open(run_dir / f"results{sfx}.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta2, "results": out}, f, ensure_ascii=False, indent=1)
    log(f"\nsaved -> {run_dir}/signs_refined.npz, results_refined.json")
    logf.close()


if __name__ == "__main__":
    main()
