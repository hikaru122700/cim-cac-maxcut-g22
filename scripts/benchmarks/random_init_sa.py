"""ステップ 4 の比較相手:CIM を使わず、ランダム初期化から SA をかけた結果と実行時間。

seed_value_pilot.py の warm SA と同じ実装(simulate_sa_warm)に一様ランダムな初期解を渡す。
CIM の種を使う方法と、同じ総時間で比べるために使う。

使い方(プロジェクトルートから):
  python scripts/benchmarks/random_init_sa.py --dataset G22 --runs 512 --sweeps 500 5000 --t-start 2.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "24")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402

EXPERIMENT_KIND = "random_init_sa"
INIT_SEED = 777
SA_SEED_OFFSET = 8_000_000


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--runs", type=int, default=512)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[500, 5000])
    ap.add_argument("--t-start", type=float, default=2.0)
    ap.add_argument("--t-end", type=float, default=0.001)
    args = ap.parse_args()
    if args.runs <= 0 or any(s <= 0 for s in args.sweeps):
        ap.error("runs and sweeps must be positive")

    ctx = load_context(args.dataset)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.dataset}_N{args.runs}_T{args.t_start:g}".replace(".", "p")
    out_dir.mkdir(parents=True)
    init = np.random.default_rng(INIT_SEED).integers(0, 2, (args.runs, ctx.n)) * 2 - 1
    seeds = SA_SEED_OFFSET + np.arange(args.runs, dtype=np.int64)
    simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init[:2], 100,
                     t_start=args.t_start, t_end=args.t_end, seeds=seeds[:2])
    arrays, times = {}, {}
    for sw in args.sweeps:
        t0 = time.perf_counter()
        _, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, sw * ctx.n,
                                    t_start=args.t_start, t_end=args.t_end, seeds=seeds)
        times[str(sw)] = time.perf_counter() - t0
        arrays[f"cuts_S{sw}"] = ctx.score(signs).astype(float)
        c = arrays[f"cuts_S{sw}"]
        print(f"{args.dataset} random-init SA {sw} sweeps: {times[str(sw)]:.1f}s "
              f"mean={c.mean():.1f} max={c.max():.0f}", flush=True)
    np.savez_compressed(out_dir / "raw.npz", **arrays)
    meta = {"dataset": args.dataset, "n": ctx.n, "bks": ctx.bks, "runs": args.runs,
            "sweeps": args.sweeps, "t_start": args.t_start, "t_end": args.t_end,
            "init_seed": INIT_SEED, "sa_seed_offset": SA_SEED_OFFSET, "sa_time_s": times,
            "threads": os.environ.get("NUMBA_NUM_THREADS")}
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
