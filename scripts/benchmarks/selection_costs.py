"""ステップ 4 の費用の実測:選別に使う処理の時間を、方法ごとに分けて測る。

ステップ 3 の記録では、指標の時間に固有ベクトルの計算(問題ごとに 1 回)と使わない指標が混ざり、
LS の時間は測っていなかった。ここではステップ 3 と同じ候補(raw.npz の cim_signs)に対して、
  ls        貪欲 1-flip LS(候補ごと)
  local     LS 前後の利得の数え上げ(model_local が使う 4 指標、候補ごと)
  eig_once  隣接行列の下端の固有ベクトル 2 本(問題ごとに 1 回)
  eig       固有ベクトルとの内積(候補ごと)
を 1 スレッドで測り、REPEATS 回の最小値を記録する。

使い方(プロジェクトルートから):
  python scripts/benchmarks/selection_costs.py --runs <step3 dirs>
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

from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.seed_value_pilot import (  # noqa: E402
    adjacency, eig_features, flip_gains, greedy_local_opt, low_eigvecs,
)

EXPERIMENT_KIND = "selection_costs"
REPEATS = 3


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


def local_counts(A, signs_pm: np.ndarray) -> None:
    g = flip_gains(A, signs_pm)
    (np.abs(g) < 1e-9).sum(axis=1)
    ((g >= -2 - 1e-9) & (g <= 1e-9)).sum(axis=1)


def measure(dataset: str, signs_pm: np.ndarray) -> dict:
    ctx = load_context(dataset)
    A = adjacency(ctx)
    k = len(signs_pm)
    greedy_local_opt(ctx, signs_pm[:2])  # JIT
    t_ls = best_time(lambda: greedy_local_opt(ctx, signs_pm))
    ls = greedy_local_opt(ctx, signs_pm)
    t_local = best_time(lambda: (local_counts(A, signs_pm), local_counts(A, ls)))
    t_eig_once = best_time(lambda: low_eigvecs(A))
    _, vecs = low_eigvecs(A)
    t_eig = best_time(lambda: eig_features(signs_pm, vecs))
    return {"dataset": dataset, "n": int(ctx.n), "candidates": k,
            "t_ls": t_ls / k, "t_local": t_local / k, "t_eig_once": t_eig_once, "t_eig": t_eig / k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    args = ap.parse_args()
    out, seen = {}, set()
    for d in args.runs:
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))["meta"]
        ds = meta["dataset"]
        if ds in seen:
            continue
        seen.add(ds)
        s = np.load(d / "raw.npz")["cim_signs"]
        signs_pm = np.where(s > 0, 1, -1).astype(np.int8)
        out[ds] = measure(ds, signs_pm)
        r = out[ds]
        print(f"{ds:>4} n={r['n']:>5} ls={r['t_ls'] * 1e3:.3f}ms local={r['t_local'] * 1e3:.3f}ms "
              f"eig_once={r['t_eig_once']:.2f}s eig={r['t_eig'] * 1e3:.3f}ms (per candidate, 1 thread)", flush=True)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{len(out)}problems"
    out_dir.mkdir(parents=True)
    (out_dir / "costs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
