"""検証 4(GA を CIM の生成時間込みで比べる)の時間を 1 スレッドの CPU 時間で測る。

ほかの計算と重なっても比が崩れにくいよう、1 スレッドで CIM 1 本と GA(初期 TS・1 世代)を交互に測り、
各量の最小値を取る。GA は ga_diversity_trajectory.py と同じパラメータ・同じ CIM のラウンド数。

出力:results/<日付>/ga_cim_timing/v{N}_<説明>/timing.json

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_cim_timing.py --datasets G22 G55 G70 --repeats 3
"""

from __future__ import annotations

import os

os.environ["NUMBA_NUM_THREADS"] = "1"  # 1 スレッドの CPU 時間で測る(import より前に固定)
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

EXPERIMENT_KIND = "ga_cim_timing"
CAL_GENERATIONS = 200  # Windows の CPU 時間は 15.6 ms 刻みなので、1 世代(約 15 ms)を多数回まとめて測る
SEED = 990000


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{desc}"
    out.mkdir(parents=True)
    return out


def time_cim(ctx, seed: int) -> float:
    t0 = time.process_time()
    run_cim(ctx, CIM_ROUNDS[ctx.name], 1, np.array([seed], dtype=np.int64))
    return time.process_time() - t0


def time_ga(ctx, seed: int, init_population: np.ndarray | None) -> tuple[float, float]:
    """(初期集団の TS の時間, 1 世代の時間) を返す。trial は 1 個。"""
    stamps = {}

    def observe(gen, pops, fits):
        stamps[gen] = time.process_time()

    simulate_ga_batch(ctx.n, ctx.edges, ctx.weights, 1, **PARAMS[ctx.name]["GA"],
                      seeds=np.array([seed], dtype=np.int64), max_generations=CAL_GENERATIONS,
                      init_population=init_population, population_observer=observe)
    return stamps[0] - stamps[-1], (stamps[CAL_GENERATIONS] - stamps[0]) / CAL_GENERATIONS


def measure(ds: str, repeats: int, log) -> dict:
    ctx = load_context(ds)
    pop = PARAMS[ds]["GA"]["pop_size"]
    # JIT のコンパイルを測定から外す
    run_cim(ctx, 10, 1, np.array([SEED], dtype=np.int64))
    _, cim_signs = run_cim(ctx, CIM_ROUNDS[ds], pop, np.arange(pop, dtype=np.int64) + SEED)
    cim_pop = cim_signs.reshape(1, pop, ctx.n)
    time_ga(ctx, SEED, None)
    rec = {"cim": [], "init_random": [], "gen_random": [], "init_cim": [], "gen_cim": []}
    for r in range(repeats):  # 交互に測る
        rec["cim"].append(time_cim(ctx, SEED + 100 + r))
        a, b = time_ga(ctx, SEED + 200 + r, None)
        rec["init_random"].append(a)
        rec["gen_random"].append(b)
        a, b = time_ga(ctx, SEED + 300 + r, cim_pop)
        rec["init_cim"].append(a)
        rec["gen_cim"].append(b)
        log(f"{ds} rep{r}: " + ", ".join(f"{k}={v[-1]:.3f}s" for k, v in rec.items()))
    return {"n": ctx.n, "pop_size": pop, "cim_rounds": CIM_ROUNDS[ds],
            "t_cim": min(rec["cim"]), "t_init_random": min(rec["init_random"]),
            "t_gen_random": min(rec["gen_random"]), "t_init_cim": min(rec["init_cim"]),
            "t_gen_cim": min(rec["gen_cim"]), "raw": rec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CIM_ROUNDS), choices=list(CIM_ROUNDS))
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    out = get_out_dir(f"{'_'.join(args.datasets)}_rep{args.repeats}")

    def log(msg):
        print(msg, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    res = {"unit": "process CPU seconds, NUMBA_NUM_THREADS=1, min over repeats",
           "cal_generations": CAL_GENERATIONS, "datasets": {}}
    for ds in args.datasets:
        res["datasets"][ds] = measure(ds, args.repeats, log)
        (out / "timing.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
