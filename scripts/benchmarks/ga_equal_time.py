"""検証 4:GA の初期集団を CIM で作る利点を、CIM の生成時間込みで比べる(実行部分)。

計画:docs/20261019/verify4_ga_plan.md(実行前に固定)。
各 trial で CIM 解を K = k_factor × 集団サイズ 個作り、4 通りの初期集団で GA を回す:
  random    GA 既定のランダム初期集団
  cim_all   CIM 解の先頭 pop 個(選別なし。CIM は pop 本だけ課す)
  cim_cut   K 個からカット値の上位 pop 個
  cim_div   K 個のカット値上位 2·pop 個から、互いに離れた pop 個を貪欲に選ぶ(ハミング距離、全反転を同一視)
GA のパラメータと CIM のラウンド数は ga_diversity_trajectory.py と同じ。GA の seed は 4 通りで共通。
時間は ga_cim_timing.py の 1 スレッド CPU 時間で数える(本スクリプトは世代ごとの最良値だけを保存し、
時間への換算は ga_equal_time_analysis.py で行う)。全条件を同じ世代数 G まで回す。G は、cim_cut が CIM
の時間(1 倍)を払った後に eval_gens 世代進む時刻まで、全条件・全倍率が届く世代数。

出力:results/<日付>/ga_equal_time/v{N}_<説明>/ に <問題>.npz と meta.json。

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_equal_time.py --timing results/2026-10-19/ga_cim_timing/v1_G22_G55_G70_rep3/timing.json
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "16")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
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

EXPERIMENT_KIND = "ga_equal_time"
CONDITIONS = ["random", "cim_all", "cim_cut", "cim_div"]
CIM_SEED_BASE = 3_000_000
MULTIPLIERS = [1.0, 0.1, 0.0]  # CIM の時間の倍率(解析と同じ)


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{desc}"
    out.mkdir(parents=True)
    return out


def pairwise_overlap(signs: np.ndarray) -> np.ndarray:
    """|s_i·s_j|/n(全反転を同一視した一致度、1 なら同じ分割)。"""
    s = np.where(signs > 0, 1.0, -1.0)
    return np.abs(s @ s.T) / s.shape[1]


def select_cut(cuts: np.ndarray, pop: int) -> np.ndarray:
    return np.argsort(-cuts, kind="stable")[:pop]


def select_div(cuts: np.ndarray, signs: np.ndarray, pop: int) -> np.ndarray:
    """カット値の上位 2·pop 個から、最良を起点に「既に選んだものとの最大一致度」が最小の候補を足していく。"""
    cand = np.argsort(-cuts, kind="stable")[:min(2 * pop, len(cuts))]
    ov = pairwise_overlap(signs[cand])
    chosen = [0]
    for _ in range(pop - 1):
        closest = ov[:, chosen].max(axis=1)
        closest[chosen] = np.inf
        chosen.append(int(np.argmin(closest)))
    return cand[chosen]


def mean_distance(signs: np.ndarray) -> float:
    """集団内の平均ペア距離(頂点数で正規化、全反転を同一視)。"""
    ov = pairwise_overlap(signs)
    d = (1 - ov) / 2
    iu = np.triu_indices(len(signs), 1)
    return float(d[iu].mean())


def generations_needed(t: dict, pop: int, k: int, eval_gens: int) -> int:
    """T_eval に全条件・全倍率(CIM の時間 c 倍)が届く世代数。最も多く要るのは c=0 の CIM 条件か、ランダム。"""
    t_eval = k * t["t_cim"] + t["t_init_cim"] + eval_gens * t["t_gen_cim"]
    need = [(t_eval - t["t_init_random"]) / t["t_gen_random"]]
    for n_cim in (pop, k):
        need += [(t_eval - c * n_cim * t["t_cim"] - t["t_init_cim"]) / t["t_gen_cim"] for c in MULTIPLIERS]
    return max(eval_gens, math.ceil(max(need) * 1.05))  # 5% の余裕


def run_dataset(ds: str, args, timing: dict, out: Path, log) -> dict:
    ctx = load_context(ds)
    gp = PARAMS[ds]["GA"]
    pop, T = gp["pop_size"], args.num_trials
    K = args.k_factor * pop
    G = generations_needed(timing[ds], pop, K, args.eval_gens)
    ga_seeds = np.arange(T, dtype=np.int64) + args.seed_offset
    cim_seeds = np.arange(T * K, dtype=np.int64) + CIM_SEED_BASE + args.seed_offset
    log(f"{ds}: pop={pop} K={K} generations={G}; CIM {T * K} runs x {CIM_ROUNDS[ds]} rounds")
    t0 = time.perf_counter()
    cuts, signs = run_cim(ctx, CIM_ROUNDS[ds], T * K, cim_seeds)
    np.testing.assert_allclose(ctx.score(signs), cuts, rtol=0, atol=1e-7)
    cuts, signs = cuts.reshape(T, K), signs.reshape(T, K, ctx.n)
    log(f"{ds}: CIM done in {time.perf_counter() - t0:.0f}s wall")

    picks = {"cim_all": np.tile(np.arange(pop), (T, 1)),
             "cim_cut": np.stack([select_cut(cuts[t], pop) for t in range(T)]),
             "cim_div": np.stack([select_div(cuts[t], signs[t], pop) for t in range(T)])}
    save = {"cim_cuts": cuts, "ga_seeds": ga_seeds, "cim_seeds": cim_seeds}
    info = {"pop_size": pop, "K": K, "generations": G, "bks": ctx.bks, "n": ctx.n,
            "cim_rounds": CIM_ROUNDS[ds], "ga_params": gp, "init": {}}
    for cond in CONDITIONS:
        init = None
        if cond != "random":
            idx = picks[cond]
            init = np.stack([signs[t, idx[t]] for t in range(T)])
            save[f"{cond}_picks"] = idx
            info["init"][cond] = {
                "mean_cut": float(np.mean([cuts[t, idx[t]].mean() for t in range(T)])),
                "best_cut": float(np.mean([cuts[t, idx[t]].max() for t in range(T)])),
                "mean_distance": float(np.mean([mean_distance(init[t]) for t in range(T)]))}
        t0 = time.perf_counter()
        best, best_signs, hist = simulate_ga_batch(
            ctx.n, ctx.edges, ctx.weights, T, **gp, seeds=ga_seeds, max_generations=G,
            init_population=init, return_history=True)
        np.testing.assert_allclose(ctx.score(best_signs), best, rtol=0, atol=1e-7)
        save[f"{cond}_history"] = hist  # (G+1, T)、0 行目は初期集団の TS 後
        log(f"{ds} {cond}: final gap mean={ctx.bks - best.mean():.2f}, "
            f"gen0 gap={ctx.bks - hist[0].mean():.2f}, {time.perf_counter() - t0:.0f}s wall")
    np.savez_compressed(out / f"{ds}.npz", **save)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", type=Path, required=True, help="ga_cim_timing.py の timing.json")
    ap.add_argument("--datasets", nargs="+", default=list(CIM_ROUNDS), choices=list(CIM_ROUNDS))
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--k-factor", type=int, default=4)
    ap.add_argument("--eval-gens", type=int, default=300)
    ap.add_argument("--seed-offset", type=int, default=20000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.num_trials < 2 or args.k_factor < 2 or args.eval_gens < 1:
        ap.error("need num_trials>=2, k_factor>=2, eval_gens>=1")
    if not args.timing.is_file():
        ap.error(f"not found: {args.timing}")
    timing = json.loads(args.timing.read_text(encoding="utf-8"))["datasets"]
    missing = [d for d in args.datasets if d not in timing]
    if missing:
        ap.error(f"no timing for {missing}")
    desc = f"{'_'.join(args.datasets)}_nt{args.num_trials}_k{args.k_factor}_eval{args.eval_gens}"
    out = get_out_dir(desc + (f"_{args.tag}" if args.tag else ""))

    def log(msg):
        print(msg, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "conditions": CONDITIONS,
            "threads": os.environ["NUMBA_NUM_THREADS"], "timing": timing, "datasets": {},
            "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), ROOT / "modules/GA.py", ROOT / "modules/CIM.py",
                                 ROOT / "scripts/benchmarks/algo_registry.py"]}}
    for ds in args.datasets:
        meta["datasets"][ds] = run_dataset(ds, args, timing, out, log)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
