"""BKS を達成した解を多数集め、結果③の距離を「最も近い BKS 解」までの距離で測り直す。

問い: 結果③(sa_start_distance v1)は、BKS に届いた解 1 つだけを基準に距離を測った。BKS 解が複数あるなら、
CIM の解が「BKS 解に近い」という観察は基準の選び方に左右されないか。

手順(プロジェクトルートから):
  1. 集める: ランダム開始の長い SA(T0 2.0 → 0.001、既定 42,104 スイープ)を 240 本ずつ回し、
     カット値が BKS に等しい解を保存する。--target 本の BKS 解が集まるか --max-runs 本で止める。
     ランダム開始なので、CIM の解の近くに偏った集め方にはならない。
  2. 結果③の 4 つの集合を再現する: sa_trajectory v3 の CIM 開始解・ランダム開始解と、
     そこから T0 2.0・5000 スイープの SA を同じ seed で回し直した最終解(最終 cut が結果①と一致することを確認)。
     元の基準解(ランダム開始 T0 2.0 で BKS に届いた 1 本)も同じ回し直しから取り出し、
     元の距離(sa_start_distance v1 の d_best)が完全に再現されることを確認する。
  3. 各解について、(a) 元の基準解 1 つとの距離、(b) 集めた BKS 解の中で最も近いものとの距離 を比べる。
     あわせて、BKS 解どうしの距離から、BKS 解がいくつの塊(谷)に分かれるかを調べる。

距離は全反転を同一視した正規化ハミング距離 min(h, n − h) / n(0 = 同じ解、0.5 = 無関係)。

  python scripts/benchmarks/bks_set_distance.py --target 100 --max-runs 9600
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "24")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS, BOOTSTRAP_SEED, T_END  # noqa: E402

EXPERIMENT_KIND = "bks_set_distance"
SOURCE_RUN = Path("results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000")
DIST_RUN = Path("results/2026-11-07/sa_start_distance/v1_G22_from_v3")
T_START = 2.0
SWEEPS_RESULT3 = 5000
RESULT3_ARMS = ("cim_T2", "random_T2")
COLLECT_SEED = 500_000_000
COLLECT_INIT_SEED = 20261111
BATCH = 240
CLUSTER_RADIUS = 0.05  # これより近い BKS 解は同じ塊(単連結)とみなす


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


def canonical(signs: np.ndarray) -> np.ndarray:
    """全反転を同一視するため、先頭スピンを +1 にそろえた ±1 の int8 を返す(新しい配列)。"""
    s = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
    return np.where(s[:, :1] > 0, s, -s).astype(np.int8)


def dist_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(len(a), len(b)) の正規化ハミング距離(全反転を同一視)。a, b は ±1。"""
    n = a.shape[1]
    overlap = a.astype(np.float32) @ b.astype(np.float32).T
    h = (n - overlap) / (2.0 * n)
    return np.minimum(h, 1.0 - h)


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def clusters(d: np.ndarray, radius: float) -> list[int]:
    """距離行列から単連結の塊を作り、塊の大きさを大きい順に返す。"""
    k = len(d)
    label = -np.ones(k, dtype=int)
    for i in range(k):
        if label[i] >= 0:
            continue
        label[i] = i
        stack = [i]
        while stack:
            j = stack.pop()
            for nb in np.where((d[j] < radius) & (label < 0))[0]:
                label[nb] = i
                stack.append(int(nb))
    return sorted(np.bincount(np.unique(label, return_inverse=True)[1]).tolist(), reverse=True)


def collect_bks(ctx, sweeps: int, target: int, max_runs: int, log: Callable[[str], None],
                ckpt: Path) -> tuple[np.ndarray, int, int]:
    """ランダム開始の長い SA で BKS 解を集める。(重複を除いた BKS 解, 回した本数, BKS 到達本数) を返す。"""
    rng = np.random.default_rng(COLLECT_INIT_SEED)
    found: list[np.ndarray] = []
    runs = 0
    t0 = time.perf_counter()
    while runs < max_runs and len(found) < target:
        r = min(BATCH, max_runs - runs)
        init = rng.choice(np.array([-1, 1], np.int8), (r, ctx.n))
        seeds = COLLECT_SEED + runs + np.arange(r, dtype=np.int64)
        cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, sweeps * ctx.n,
                                      t_start=T_START, t_end=T_END, seeds=seeds)
        cut = np.asarray(cut, dtype=np.float64)
        hit = cut == float(ctx.bks)
        if hit.any():
            sol = canonical(np.asarray(signs)[hit])
            if not np.array_equal(ctx.score(sol).astype(np.float64), cut[hit]):
                raise RuntimeError("rescored BKS cut differs")
            found.extend(sol)
        runs += r
        log(f"collect: {runs} runs, {len(found)} BKS hits ({time.perf_counter() - t0:.0f}s)")
        np.save(ckpt, np.array(found, dtype=np.int8).reshape(-1, ctx.n))
    hits = np.array(found, dtype=np.int8).reshape(-1, ctx.n)
    return np.unique(hits, axis=0), runs, len(hits)


def result3_sets(ctx, log: Callable[[str], None]) -> tuple[dict, np.ndarray, dict]:
    """結果③の 4 集合(開始解 2 つ、SA 後 2 つ)と元の基準解を再現する。"""
    z = np.load(SOURCE_RUN / "raw.npz")
    cim_init = z["top_signs"].reshape(-1, ctx.n).astype(np.int8)
    n_runs = len(cim_init)
    rnd_init = np.random.default_rng(BOOTSTRAP_SEED).choice(np.array([-1, 1], np.int8), (n_runs, ctx.n))
    if not np.array_equal(ctx.score(rnd_init).astype(np.int32), z["random_init_cuts"]):
        raise RuntimeError("random starts do not reproduce sa_trajectory")
    starts = {"cim": cim_init, "random": rnd_init}
    sets = {"random_start": rnd_init, "cim_start": cim_init}
    cuts = {}
    for arm in RESULT3_ARMS:
        start, t_start, seed_base = ARMS[arm]
        seeds = seed_base + SWEEPS_RESULT3 * 100_000 + np.arange(n_runs, dtype=np.int64)
        cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, starts[start], SWEEPS_RESULT3 * ctx.n,
                                      t_start=t_start, t_end=T_END, seeds=seeds)
        if not np.array_equal(np.asarray(cut).astype(np.float32), z[f"best_{arm}_S{SWEEPS_RESULT3}"][:, -1]):
            raise RuntimeError(f"{arm}: final cuts differ from {SOURCE_RUN}")
        sets[f"{start}_sa"] = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
        cuts[f"{start}_sa"] = np.asarray(cut, dtype=np.float64)
        log(f"result3 {arm}: verified {n_runs} runs")
    idx = np.where(cuts["random_sa"] == float(ctx.bks))[0]
    if len(idx) != 1:
        raise RuntimeError(f"expected one BKS run in random_T2_S{SWEEPS_RESULT3}, got {len(idx)}")
    ref = sets["random_sa"][idx[0]]
    old = np.load(DIST_RUN / "raw.npz")
    for key, arm in (("cim_sa", "cim_T2"), ("random_sa", "random_T2")):
        d = dist_matrix(sets[key], ref[None, :])[:, 0].astype(np.float32)
        if not np.allclose(d, old[f"d_best_{arm}_S{SWEEPS_RESULT3}"], atol=1e-6):
            raise RuntimeError(f"{key}: distances to the reference differ from {DIST_RUN}")
    log("reference solution and its distances reproduce sa_start_distance v1")
    return sets, ref, cuts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22")
    ap.add_argument("--sweeps", type=int, default=42_104, help="BKS 解を集める SA 1 本のスイープ数")
    ap.add_argument("--target", type=int, default=100, help="この本数の BKS 到達で止める")
    ap.add_argument("--max-runs", type=int, default=9600)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.dataset != "G22":
        ap.error("result 3 exists only for G22")
    if args.sweeps < 1 or args.target < 1 or args.max_runs < 1:
        ap.error("sweeps, target and max-runs must be positive")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    ctx = load_context(args.dataset)
    kind_root = get_kind_root()
    desc = f"{args.dataset}_S{args.sweeps}_target{args.target}" + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    sets, ref, cuts = result3_sets(ctx, log)
    bks_set, runs, n_hits = collect_bks(ctx, args.sweeps, args.target, args.max_runs, log,
                                        out_dir / "bks_hits_checkpoint.npy")
    if len(bks_set) == 0:
        raise RuntimeError("no BKS solution collected")
    log(f"collected {n_hits} BKS hits in {runs} runs, {len(bks_set)} distinct")

    d_bks = dist_matrix(bks_set, bks_set)
    off = d_bks[~np.eye(len(bks_set), dtype=bool)]
    sizes = clusters(d_bks, CLUSTER_RADIUS)
    d_ref_bks = dist_matrix(bks_set, ref[None, :])[:, 0]
    log(f"BKS-BKS distance: {describe(off) if len(off) else 'n/a'}; clusters (r<{CLUSTER_RADIUS}): {sizes}")
    log(f"original reference to collected BKS: min {d_ref_bks.min():.4f}, mean {d_ref_bks.mean():.4f}")

    # 元の基準解も BKS 解の集合に入れて「最も近い BKS 解」を測る
    ref_set = np.unique(np.vstack([bks_set, canonical(ref[None, :])]), axis=0)
    results, arrays = {}, {}
    for key, sol in sets.items():
        d_single = dist_matrix(sol, ref[None, :])[:, 0]
        d_all = dist_matrix(sol, ref_set)
        d_near = d_all.min(axis=1)
        results[key] = {"single_reference": describe(d_single), "nearest_bks": describe(d_near),
                        "mean_over_bks": describe(d_all.mean(axis=1))}
        arrays.update({f"d_single_{key}": d_single.astype(np.float32), f"d_nearest_{key}": d_near.astype(np.float32),
                       f"nearest_idx_{key}": d_all.argmin(axis=1).astype(np.int32)})
        log(f"{key:>13}: single {d_single.mean():.3f}  nearest-BKS {d_near.mean():.3f}  "
            f"mean-over-BKS {d_all.mean():.3f}")

    summary = {"meta": {"dataset": args.dataset, "bks": float(ctx.bks), "collect_sweeps": args.sweeps,
                        "collect_t_start": T_START, "t_end": T_END, "collect_runs": runs, "bks_hits": n_hits,
                        "bks_distinct": int(len(bks_set)), "reference_set_size": int(len(ref_set)),
                        "cluster_radius": CLUSTER_RADIUS, "source_run": str(SOURCE_RUN), "dist_run": str(DIST_RUN),
                        "result3_sweeps": SWEEPS_RESULT3, "threads": int(os.environ["NUMBA_NUM_THREADS"])},
               "bks_structure": {"pairwise": describe(off) if len(off) else None, "cluster_sizes": sizes,
                                 "original_reference_to_collected": describe(d_ref_bks)},
               "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", bks_signs=bks_set, reference=canonical(ref[None, :])[0],
                        bks_pairwise=d_bks.astype(np.float32), **arrays)
    (out_dir / "bks_hits_checkpoint.npy").unlink()
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
