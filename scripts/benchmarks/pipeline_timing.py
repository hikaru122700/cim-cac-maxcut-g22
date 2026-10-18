"""ステップ 4 の時間の測り直し:CIM 1 本と SA 1 本の時間を、ほかの計算がない状態で測る。

ステップ 3 の実行は一部がランダム初期化 SA の計算と重なり、同じ seed の CIM の時間が
2 つの温度の実行で最大 2.3 倍違った。選別評価ではランダム初期化に回す本数が
t_cim / t_sa で決まるので、ここで NUMBA_NUM_THREADS=24 のまま、JIT を済ませてから
  CIM  K=512 本・2400 ラウンド(ステップ 3 と同じ seed)
  SA   ステップ 3 の候補を初期解に 512 本(50・500・5000 掃引、T0 = 0.5 と 2.0)
を REPEATS 回ずつ測り、最小値を 1 本あたりに直して記録する。

使い方(プロジェクトルートから):
  NUMBA_NUM_THREADS=24 python scripts/benchmarks/pipeline_timing.py --runs <step3 dirs>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context, run_cim  # noqa: E402

EXPERIMENT_KIND = "pipeline_timing"
REPEATS = 2
SA_RUNS = 512
SWEEPS = (50, 500, 5000)
T_STARTS = (0.5, 2.0)
T_END = 0.001
SA_SEED = 9_000_000


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
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


def best_time(fn) -> float:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def measure(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["meta"]
    ctx = load_context(meta["dataset"])
    signs = np.load(run_dir / "raw.npz")["cim_signs"]
    k = len(signs)
    seeds = meta["cim_seed_start"] + np.arange(k, dtype=np.int64) if "cim_seed_start" in meta \
        else 1000 + np.arange(k, dtype=np.int64)
    rounds = meta["cim_rounds"]
    run_cim(ctx, 10, 2, seeds[:2])  # JIT
    t_cim = best_time(lambda: run_cim(ctx, rounds, k, seeds)) / k
    init = np.where(signs[:SA_RUNS] > 0, 1, -1).astype(np.int8)
    sa_seeds = SA_SEED + np.arange(len(init), dtype=np.int64)
    simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init[:2], 100, t_start=0.5, t_end=T_END, seeds=sa_seeds[:2])
    t_sa = {}
    for t0 in T_STARTS:
        for sw in SWEEPS:
            t = best_time(lambda: simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, sw * ctx.n,
                                                   t_start=t0, t_end=T_END, seeds=sa_seeds))
            t_sa[f"T{t0:g}_S{sw}"] = t / len(init)
    return {"dataset": meta["dataset"], "n": int(ctx.n), "cim_rounds": rounds, "t_cim": t_cim, "t_sa": t_sa}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    out, seen = {}, set()
    for d in args.runs:
        ds = json.loads((d / "summary.json").read_text(encoding="utf-8"))["meta"]["dataset"]
        if ds in seen:
            continue
        seen.add(ds)
        out[ds] = measure(d)
        r = out[ds]
        print(f"{ds:>4} t_cim={r['t_cim'] * 1e3:.2f}ms " + " ".join(f"{k}={v * 1e3:.2f}ms" for k, v in r["t_sa"].items()),
              flush=True)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{len(out)}problems"
    out_dir.mkdir(parents=True)
    (out_dir / "timing.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
