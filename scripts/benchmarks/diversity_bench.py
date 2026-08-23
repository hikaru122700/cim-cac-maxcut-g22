"""diversity_bench.py — 各アルゴリズムが出力する「解の多様性」を測る。

同一インスタンスを **独立シード T 本** で解かせ、返ってきた解ベクトル(分割)を
すべて保存する。時間予算は手法間で揃える: 予算グリッドを昇順に走らせ、
バッチ実時間が --target-sec を超える直前の予算を採用する(anytime 比較と同じ
「実時間を横軸に揃える」思想)。

出力: results/<date>/solution_diversity/v{N}_<desc>/
  signs.npz   : <DS>__<ALGO>_signs (T, n) bool / <DS>__<ALGO>_cuts (T,)
  results.json: 予算・実時間・カット統計・多様性指標
  run.log

実行(プロジェクトルートから):
  .venv/Scripts/python.exe -u scripts/benchmarks/diversity_bench.py \
      --datasets G22 K2000 G55 G70 --num-trials 32 --tag main
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
from scripts.benchmarks.diversity_metrics import summarize

EXPERIMENT_KIND = "solution_diversity"

# データセット別の既定ターゲット実時間 [秒/バッチ]。anytime 曲線で
# 各手法が品質プラトーに乗る付近を狙った値。
DEFAULT_TARGET_SEC = {"G22": 20.0, "G55": 30.0, "G70": 30.0, "K2000": 120.0}


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


def build_description(args) -> str:
    parts = ["_".join(args.datasets), f"nt{args.num_trials}"]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def run_matched(ctx, algo_key, num_trials, target_sec, log):
    """予算グリッドを昇順に走らせ、target_sec を超える直前の結果を返す。

    Returns (budget, elapsed, signs) or None。
    """
    run = ALGOS[algo_key]["run"]
    seeds = np.arange(num_trials, dtype=np.int64)
    grid = BUDGET_GRIDS[algo_key]

    # JIT ウォームアップ(最小予算・少数トライアル、計測しない)
    nw = min(num_trials, 4)
    try:
        run(ctx, grid[0], nw, seeds[:nw])
    except Exception as e:  # noqa: BLE001
        log(f"    [WARN] {algo_key} warmup failed: {type(e).__name__}: {e}")
        return None

    best = None
    prev_b, prev_t = None, None
    for b in grid:
        # 直前の測定から外挿して、明らかに超過するものは走らせない
        if prev_t is not None and prev_b:
            pred = prev_t * (b / prev_b)
            if pred > 4.0 * target_sec and best is not None:
                log(f"    {algo_key:4s} b={b:>9} 予測 {pred:.0f}s > 4x目標 → 打ち切り")
                break
        t0 = time.perf_counter()
        try:
            _cuts, signs = run(ctx, b, num_trials, seeds)
        except Exception as e:  # noqa: BLE001
            log(f"    [WARN] {algo_key} b={b} failed: {type(e).__name__}: {e} → 打ち切り")
            break
        dt = time.perf_counter() - t0
        cuts = np.asarray(ctx.score(signs), dtype=float)
        log(f"    {algo_key:4s} b={b:>9} t={dt:7.2f}s  mean={cuts.mean():9.1f} "
            f"max={int(cuts.max()):7d}")
        if best is None or dt <= target_sec:
            best = (b, dt, np.asarray(signs))
        prev_b, prev_t = b, dt
        if dt > target_sec:
            break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["G22"], choices=list(DATASETS))
    ap.add_argument("--algos", nargs="+", default=list(ALGOS), choices=list(ALGOS))
    ap.add_argument("--num-trials", type=int, default=32)
    ap.add_argument("--target-sec", type=float, default=None,
                    help="1 バッチの目標実時間。省略時はデータセット別既定値。")
    ap.add_argument("--near-gap-pct", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    kind_root = get_kind_root()
    v = next_version(kind_root)
    out_dir = kind_root / f"v{v}_{build_description(args)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logf = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"out_dir = {out_dir}")
    log(f"datasets={args.datasets} algos={args.algos} num_trials={args.num_trials}")

    arrays = {}
    results = {}
    for ds in args.datasets:
        target = args.target_sec or DEFAULT_TARGET_SEC.get(ds, 30.0)
        log(f"\n=== {ds} (BKS={DATASETS[ds]['bks']}, 目標 {target:.0f}s/バッチ) ===")
        ctx = load_context(ds)
        results[ds] = {}
        for algo in args.algos:
            got = run_matched(ctx, algo, args.num_trials, target, log)
            if got is None:
                log(f"  {algo}: 取得失敗 → スキップ")
                continue
            b, dt, signs = got
            cuts = np.asarray(ctx.score(signs), dtype=float)
            S = (np.asarray(signs) > 0)
            arrays[f"{ds}__{algo}_signs"] = S
            arrays[f"{ds}__{algo}_cuts"] = cuts
            summ = summarize(S, cuts, ctx.bks, near_gap_pct=args.near_gap_pct)
            summ.update({"budget": float(b), "time": float(dt)})
            results[ds][algo] = summ
            log(f"  -> {algo:4s} 採用 b={b} t={dt:.2f}s  gap平均={summ['gap_pct_mean']:.3f}% "
                f"平均距離={summ['mean_pairwise']:.4f} 相異なる解={summ['n_distinct']}/"
                f"{summ['num_trials']} 固定頂点={summ['frozen_frac']*100:.1f}% "
                f"エントロピー={summ['entropy_bits']:.3f}bit")

    np.savez_compressed(out_dir / "signs.npz", **arrays)
    meta = {
        "datasets": args.datasets, "algos": args.algos,
        "num_trials": args.num_trials,
        "target_sec": args.target_sec, "default_target_sec": DEFAULT_TARGET_SEC,
        "near_gap_pct": args.near_gap_pct, "tag": args.tag,
        "bks": {d: DATASETS[d]["bks"] for d in args.datasets},
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=1)
    log(f"\nsaved -> {out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
