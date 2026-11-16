"""SA の途中で、解のパターンが BKS 解の谷へどう近づくかを追う(G22、実験 A)。

問い(11/9 進捗報告のフィードバック): CIM の解・ランダム解から SA で最適解に至るまでに、スピンのパターンは
どう変わるのか。CIM 開始は最初から BKS 解の谷の近くにいて、そこから離れずに仕上がるのか。ランダム開始は
別の谷に落ちるのか。

手順(プロジェクトルートから):
  1. 開始解は結果①③と同じ(sa_trajectory v3 の CIM 上位 16 個 × 100 組と、同じ seed のランダム解、各 1600 個)。
  2. SA も結果①と同じ(開始温度・seed・5000 スイープ・終了温度 0.001)。乱数の消費順まで modules/SA.py と同じ
     カーネルで、対数等間隔のチェックポイントごとに今の解について次を記録する。
       - 最も近い BKS 解までの距離(bks_set_distance v3 で集めた BKS 解 88 個 + 元の基準解 1 個)
       - 開始解からの距離
       - 今のカット値、ここまでに反転したスピンの延べ数
  3. 先頭の数本は、チェックポイントごとのスピン全体も保存する(パターンの移り変わりの図用)。

距離は全反転を同一視した正規化ハミング距離 min(h, n − h) / n(0 = 同じ解、0.5 = 無関係)。

検証: 最終の最良カット値が sa_trajectory v3 と完全一致すること(同じ乱数列で動いている証拠)。

  python scripts/benchmarks/sa_path_bks_distance.py
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
from numba import njit, prange  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import _build_csr_sa  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS, BOOTSTRAP_SEED, T_END  # noqa: E402

EXPERIMENT_KIND = "sa_path_bks_distance"
TRAJ_RUN = Path("results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000")
BKS_RUN = Path("results/2026-11-11/bks_set_distance/v3_G22_S42104_target100")
SWEEPS = 5000
DEFAULT_ARMS = ("cim_T2", "random_T2", "cim_T0p5")
N_CHECKPOINTS = 80
VALLEY_RADIUS = 0.05  # BKS 解の谷の中とみなす距離(bks_set_distance の塊の判定と同じ)


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


def make_checkpoints(n: int, sweeps: int, count: int = N_CHECKPOINTS) -> np.ndarray:
    """0 と、0.01 スイープから最後までの対数等間隔の反転提案回数。最後は必ず sweeps * n。"""
    grid = np.logspace(-2, np.log10(sweeps), count)
    iters = np.unique(np.round(grid * n).astype(np.int64))
    return np.unique(np.concatenate([[0], iters, [sweeps * n]]))


@njit(cache=True)
def _nearest_and_start(x, start, refs):
    """今の解 x について (最も近い参照解までの距離, 開始解からの距離) を返す。全反転を同一視。"""
    n = x.size
    best = 1.0
    for k in range(refs.shape[0]):
        o = 0
        for i in range(n):
            o += x[i] * refs[k, i]
        d = (n - abs(o)) / (2.0 * n)
        if d < best:
            best = d
    o = 0
    for i in range(n):
        o += x[i] * start[i]
    return best, (n - abs(o)) / (2.0 * n)


@njit(cache=True, parallel=True)
def sa_path(n, num_iters, indptr, indices, weights, init, seeds, checkpoints, t_start, t_end, refs, n_snap):
    """modules/SA.py の _simulate_sa_batch_warm と同じ SA。チェックポイントで距離などを記録する。"""
    runs = init.shape[0]
    n_cp = checkpoints.size
    d_bks = np.zeros((runs, n_cp), dtype=np.float32)
    d_start = np.zeros((runs, n_cp), dtype=np.float32)
    cur_out = np.zeros((runs, n_cp), dtype=np.float32)
    flips = np.zeros((runs, n_cp), dtype=np.int64)
    snaps = np.zeros((n_snap, n_cp, n), dtype=np.int8)
    final_best = np.zeros(runs, dtype=np.float64)
    log_ratio = np.log(t_end / t_start)
    for r in prange(runs):
        np.random.seed(seeds[r])
        x = init[r].copy()
        start = init[r]
        cur = 0.0
        for i in range(n):
            for k in range(indptr[i], indptr[i + 1]):
                if indices[k] > i and x[i] != x[indices[k]]:
                    cur += weights[k]
        best = cur
        nflip = 0
        c = 0
        while c < n_cp and checkpoints[c] == 0:
            a, b = _nearest_and_start(x, start, refs)
            d_bks[r, c], d_start[r, c], cur_out[r, c], flips[r, c] = a, b, cur, nflip
            if r < n_snap:
                snaps[r, c] = x
            c += 1
        for it in range(num_iters):
            T = t_start * np.exp(log_ratio * (it / num_iters))
            v = np.random.randint(0, n)
            delta = 0.0
            for k in range(indptr[v], indptr[v + 1]):
                if x[v] == x[indices[k]]:
                    delta += weights[k]
                else:
                    delta -= weights[k]
            if delta > 0:
                x[v] = -x[v]
                cur += delta
                nflip += 1
            elif T > 0:
                if np.random.random() < np.exp(delta / T):
                    x[v] = -x[v]
                    cur += delta
                    nflip += 1
            if cur > best:
                best = cur
            while c < n_cp and checkpoints[c] == it + 1:
                a, b = _nearest_and_start(x, start, refs)
                d_bks[r, c], d_start[r, c], cur_out[r, c], flips[r, c] = a, b, cur, nflip
                if r < n_snap:
                    snaps[r, c] = x
                c += 1
        final_best[r] = best
    return d_bks, d_start, cur_out, flips, snaps, final_best


def load_starts(ctx) -> dict[str, np.ndarray]:
    """結果①③と同じ開始解(CIM 上位 16 個 × 100 組、同じ seed のランダム解)。"""
    z = np.load(TRAJ_RUN / "raw.npz")
    cim = z["top_signs"].reshape(-1, ctx.n).astype(np.int8)
    rnd = np.random.default_rng(BOOTSTRAP_SEED).choice(np.array([-1, 1], np.int8), (len(cim), ctx.n))
    if not np.array_equal(ctx.score(rnd).astype(np.int32), z["random_init_cuts"]):
        raise RuntimeError("random starts do not reproduce sa_trajectory v3")
    if not np.array_equal(ctx.score(cim).astype(np.int32), z["top_init_cuts"]):
        raise RuntimeError("CIM starts do not reproduce sa_trajectory v3")
    return {"cim": cim, "random": rnd}


def load_refs(ctx) -> np.ndarray:
    """BKS 解の集合(集めた 88 個 + 元の基準解)。カット値が BKS であることを確かめる。"""
    z = np.load(BKS_RUN / "raw.npz")
    refs = np.unique(np.vstack([z["bks_signs"], z["reference"][None, :]]), axis=0).astype(np.int8)
    if not np.all(ctx.score(refs).astype(np.float64) == float(ctx.bks)):
        raise RuntimeError("reference set contains a non-BKS solution")
    return refs


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3)}


def path_table(sweeps: np.ndarray, d_bks: np.ndarray, d_start: np.ndarray, cur: np.ndarray, bks: float) -> list:
    """代表的なスイープ数での中央値の表(RESULTS.md 用)。"""
    rows = []
    for s in (0, 1, 3, 10, 30, 100, 300, 1000, 3000, 5000):
        i = int(np.argmin(np.abs(sweeps - s)))
        rows.append({"sweep": float(sweeps[i]), "d_bks_median": float(np.median(d_bks[:, i])),
                     "d_start_median": float(np.median(d_start[:, i])),
                     "gap_pct_median": float((bks - np.median(cur[:, i])) / bks * 100),
                     "frac_in_valley": float((d_bks[:, i] < VALLEY_RADIUS).mean())})
    return rows


def run_arm(ctx, arm: str, starts: dict, refs: np.ndarray, n_snap: int, log: Callable[[str], None]) -> dict:
    start_kind, t_start, seed_base = ARMS[arm]
    init = np.ascontiguousarray(starts[start_kind])
    seeds = seed_base + SWEEPS * 100_000 + np.arange(len(init), dtype=np.int64)  # sa_trajectory v3 と同じ
    indptr, indices, weights = _build_csr_sa(ctx.n, ctx.edges, ctx.weights)
    cps = make_checkpoints(ctx.n, SWEEPS)
    t0 = time.perf_counter()
    d_bks, d_start, cur, flips, snaps, final = sa_path(ctx.n, SWEEPS * ctx.n, indptr, indices, weights, init, seeds,
                                                       cps, float(t_start), T_END, refs, n_snap)
    expect = np.load(TRAJ_RUN / "raw.npz")[f"best_{arm}_S{SWEEPS}"][:, -1]
    if not np.array_equal(final.astype(np.float32), expect):
        raise RuntimeError(f"{arm}: final best cuts differ from sa_trajectory v3")
    log(f"{arm}: verified final best cuts == sa_trajectory v3 ({time.perf_counter() - t0:.0f}s)")
    return {"checkpoints": cps, "d_bks": d_bks, "d_start": d_start, "cur": cur, "flips": flips, "snaps": snaps,
            "final": final}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS), choices=sorted(ARMS))
    ap.add_argument("--n-snap", type=int, default=8, help="スピン全体を保存する先頭の本数")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.n_snap < 0:
        ap.error("n-snap must be >= 0")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    meta = json.loads((TRAJ_RUN / "summary.json").read_text(encoding="utf-8"))["meta"]
    ctx = load_context(meta["dataset"])
    bks = float(ctx.bks)
    kind_root = get_kind_root()
    desc = f"{meta['dataset']}_S{SWEEPS}_" + "_".join(args.arms) + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    starts = load_starts(ctx)
    refs = load_refs(ctx)
    log(f"starts: {len(starts['cim'])} CIM / {len(starts['random'])} random; BKS reference set: {len(refs)}")

    results, arrays = {}, {}
    for arm in args.arms:
        out = run_arm(ctx, arm, starts, refs, args.n_snap, log)
        sweeps = out["checkpoints"] / ctx.n
        results[arm] = {
            "path": path_table(sweeps, out["d_bks"], out["d_start"], out["cur"], bks),
            "start_d_bks": describe(out["d_bks"][:, 0]), "final_d_bks": describe(out["d_bks"][:, -1]),
            "max_d_bks_along_path": describe(out["d_bks"].max(axis=1)),
            "final_frac_in_valley": float((out["d_bks"][:, -1] < VALLEY_RADIUS).mean()),
            "final_gap_pct_mean": float((bks - out["final"]).mean() / bks * 100),
            "total_flips": describe(out["flips"][:, -1].astype(np.float64)),
        }
        arrays.update({f"d_bks_{arm}": out["d_bks"], f"d_start_{arm}": out["d_start"], f"cur_{arm}": out["cur"],
                       f"flips_{arm}": out["flips"], f"snaps_{arm}": out["snaps"], f"final_{arm}": out["final"]})
        arrays["sweeps"] = sweeps.astype(np.float64)
        r = results[arm]
        log(f"{arm}: d_bks start {r['start_d_bks']['median']:.3f} -> peak {r['max_d_bks_along_path']['median']:.3f} "
            f"-> final {r['final_d_bks']['median']:.3f}; in valley at end {r['final_frac_in_valley']:.1%}")

    summary = {"meta": {"dataset": meta["dataset"], "bks": bks, "sweeps": SWEEPS, "t_end": T_END,
                        "arms": {a: {"start": ARMS[a][0], "t_start": ARMS[a][1]} for a in args.arms},
                        "runs_per_arm": int(len(starts["cim"])), "reference_set_size": int(len(refs)),
                        "valley_radius": VALLEY_RADIUS, "traj_run": str(TRAJ_RUN), "bks_run": str(BKS_RUN),
                        "n_snap": args.n_snap, "threads": int(os.environ["NUMBA_NUM_THREADS"])},
               "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", refs=refs, **arrays)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
