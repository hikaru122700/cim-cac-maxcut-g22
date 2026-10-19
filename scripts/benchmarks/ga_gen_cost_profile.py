"""検証 4 の時間の前提(GA の 1 世代の時間が世代によらず一定)を確かめる。

1 スレッドで GA(trial 1 個)を長く回し、100 世代ごとの CPU 時間を記録する。ランダムな初期集団と CIM の
初期集団(カット値の上位)の 2 通り。ga_cim_timing.py の t_gen(最初の 200 世代)と、全区間の傾きを比べる。

出力:results/<日付>/ga_gen_cost_profile/v{N}_<説明>/profile.json

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_gen_cost_profile.py --generations 2365 3433 5247 --datasets G22 G55 G70
"""

from __future__ import annotations

import os

os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.GA import simulate_ga_batch  # noqa: E402
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim  # noqa: E402
from scripts.benchmarks.ga_diversity_trajectory import CIM_ROUNDS  # noqa: E402

EXPERIMENT_KIND = "ga_gen_cost_profile"
STEP = 100
SEED = 980000


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{desc}"
    out.mkdir(parents=True)
    return out


def profile(ctx, gens: int, init) -> dict:
    stamps = {}

    def observe(gen, pops, fits):
        if gen % STEP == 0 or gen == gens:
            stamps[gen] = time.process_time()

    simulate_ga_batch(ctx.n, ctx.edges, ctx.weights, 1, **PARAMS[ctx.name]["GA"],
                      seeds=np.array([SEED], dtype=np.int64), max_generations=gens,
                      init_population=init, population_observer=observe)
    g = np.array(sorted(stamps))
    s = np.array([stamps[k] for k in g]) - stamps[0]
    head = s[g == 200][0] / 200 if 200 in stamps else float("nan")
    slope = float(np.polyfit(g, s, 1)[0])
    last = (s[-1] - s[g >= g[-1] // 2][0]) / (g[-1] - g[g >= g[-1] // 2][0])  # 後半の平均
    return {"gens": g.tolist(), "cpu_s": s.tolist(), "t_gen_first200": float(head),
            "t_gen_slope": slope, "t_gen_second_half": float(last)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CIM_ROUNDS), choices=list(CIM_ROUNDS))
    ap.add_argument("--generations", type=int, nargs="+", required=True)
    args = ap.parse_args()
    if len(args.generations) != len(args.datasets):
        ap.error("--generations must match --datasets")
    out = get_out_dir("_".join(args.datasets))
    res = {"unit": "process CPU seconds, NUMBA_NUM_THREADS=1, one trial", "step": STEP, "datasets": {}}
    for ds, gens in zip(args.datasets, args.generations):
        ctx = load_context(ds)
        pop = PARAMS[ds]["GA"]["pop_size"]
        cuts, signs = run_cim(ctx, CIM_ROUNDS[ds], 4 * pop, np.arange(4 * pop, dtype=np.int64) + SEED)
        top = np.argsort(-cuts, kind="stable")[:pop]
        simulate_ga_batch(ctx.n, ctx.edges, ctx.weights, 1, **PARAMS[ds]["GA"],
                          seeds=np.array([SEED], dtype=np.int64), max_generations=2)  # JIT
        res["datasets"][ds] = {"random": profile(ctx, gens, None),
                               "cim_cut": profile(ctx, gens, signs[top][None])}
        for k, v in res["datasets"][ds].items():
            print(f"{ds} {k}: first200={v['t_gen_first200']:.4f} slope={v['t_gen_slope']:.4f} "
                  f"second_half={v['t_gen_second_half']:.4f}", flush=True)
        (out / "profile.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
