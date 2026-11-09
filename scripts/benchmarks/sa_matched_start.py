"""カット値をそろえた「CIM 以外の良い解」から SA を始め、CIM の解から始めた SA と比べる(sa_trajectory の続き)。

問い: CIM の解から始めた SA が良い解に着くのは「開始解のカット値が良いから」か、「CIM の解だから」か。

手順:
  1. ランダム開始の SA(T0 = 2.0, 5000 スイープ)を pool 本回し、その最良解をためる(CIM ではない良い解)。
  2. sa_trajectory の CIM 開始 1600 個それぞれに、カット値がいちばん近い SA の解を重複なしで 1 つずつ対応させる
     (カット値の良い CIM 解から順に選ぶ。良い解ほど数が少ないため)。
  3. 対応させた SA の解から、CIM 開始と同じ開始温度・同じ SA seed・同じスイープ数で SA を回す。
     違うのは開始解だけになる。
  4. CIM 開始の結果(sa_trajectory v3 の最終値、sa_start_distance v1 の距離)と比べる。
     対応する 16 本(CIM の 1 プール分)の平均を単位に、対応のある比較(Wilcoxon 符号順位検定、ブートストラップ区間)を行う。

使い方(プロジェクトルートから):
  python scripts/benchmarks/sa_matched_start.py
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS, T_END  # noqa: E402

EXPERIMENT_KIND = "sa_matched_start"
TRAJ_RUN = Path("results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000")
DIST_RUN = Path("results/2026-11-07/sa_start_distance/v1_G22_from_v3")
SWEEPS = 5000
POOL_T_START = 2.0
POOL_SEED_BASE = 110_000_000  # sa_trajectory の ARMS(60M〜90M 台)と重ならない
GROUP = 16  # CIM の 1 プールから選んだ本数(検定の単位)
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 20261108
COMPARE = {"T2": "cim_T2", "T0p5": "cim_T0p5"}  # この実験の条件 → 比べる CIM 開始の条件(seed も同じにする)


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


def match_by_cut(targets: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """targets の各要素に、カット値がいちばん近い pool の要素を重複なしで割り当て、その添字を返す。

    カット値の大きい target から順に選ぶ(良い解ほど pool に少ないので、先に確保する)。
    同じ距離なら小さい添字を選ぶ。pool が targets より少ないときはエラー。
    """
    targets = np.asarray(targets, dtype=np.float64)
    pool = np.asarray(pool, dtype=np.float64)
    if pool.size < targets.size:
        raise ValueError(f"pool ({pool.size}) must be at least as large as targets ({targets.size})")
    used = np.zeros(pool.size, dtype=bool)
    out = np.empty(targets.size, dtype=np.int64)
    for t in np.argsort(-targets, kind="stable"):
        diff = np.where(used, np.inf, np.abs(pool - targets[t]))
        j = int(np.argmin(diff))
        used[j] = True
        out[t] = j
    return out


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行ごとの正規化ハミング距離(全反転を同一視)。0 = 同じ解、0.5 = 無関係な解と同程度。"""
    h = (a != b).mean(axis=1)
    return np.minimum(h, 1.0 - h)


def gap_pct(cut: np.ndarray, bks: float) -> np.ndarray:
    return (bks - np.asarray(cut, dtype=np.float64)) / bks * 100


def paired_compare(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> dict:
    """16 本組の平均カットの差 a − b(対応あり)。区間は組単位のブートストラップ 95%。"""
    ga = a.reshape(-1, GROUP).mean(axis=1)
    gb = b.reshape(-1, GROUP).mean(axis=1)
    d = ga - gb
    boot = rng.choice(d, (BOOTSTRAP, d.size)).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_cut_diff": float(d.mean()), "ci95": [float(lo), float(hi)],
            "wilcoxon_p": float(wilcoxon(d).pvalue), "groups": int(d.size)}


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def arm_summary(start_cut, final_cut, d_start, bks) -> dict:
    g = gap_pct(final_cut, bks)
    rho = spearmanr(start_cut, final_cut)
    return {"start_gap": describe(gap_pct(start_cut, bks)), "final_gap": describe(g),
            "final_gap_ci95_halfwidth": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
            "frac_better_than_start": float((final_cut > start_cut).mean()),
            "frac_within_0p03pct": float((g <= 0.03).mean()),
            "distance_from_start": describe(d_start),
            "spearman_start_cut_vs_final_cut": {"rho": float(rho.statistic), "p": float(rho.pvalue)}}


def run_sa(ctx, init: np.ndarray, seeds: np.ndarray, t_start: float) -> tuple[np.ndarray, np.ndarray]:
    cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, SWEEPS * ctx.n,
                                  t_start=t_start, t_end=T_END, seeds=seeds)
    signs = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
    cut = np.asarray(cut, dtype=np.float64)
    if not np.array_equal(ctx.score(signs).astype(np.float64), cut):
        raise RuntimeError("rescored cut differs from SA's best cut")
    return cut, signs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=4800, help="CIM 以外の良い解をためる SA の本数")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    meta = json.loads((TRAJ_RUN / "summary.json").read_text(encoding="utf-8"))["meta"]
    z = np.load(TRAJ_RUN / "raw.npz")
    dz = np.load(DIST_RUN / "raw.npz")
    ctx = load_context(meta["dataset"])
    bks = float(ctx.bks)
    cim_init = z["top_signs"].reshape(-1, ctx.n).astype(np.int8)
    cim_cut = z["top_init_cuts"].astype(np.float64)
    n_runs = cim_cut.size
    if not np.array_equal(ctx.score(cim_init).astype(np.float64), cim_cut):
        raise RuntimeError("CIM starts do not reproduce sa_trajectory")

    kind_root = get_kind_root()
    desc = f"{meta['dataset']}_pool{args.pool}_S{SWEEPS}" + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t0 = time.perf_counter()
    rnd = np.random.default_rng(POOL_SEED_BASE).choice(np.array([-1, 1], np.int8), (args.pool, ctx.n))
    pool_cut, pool_signs = run_sa(ctx, rnd, POOL_SEED_BASE + np.arange(args.pool, dtype=np.int64), POOL_T_START)
    log(f"pool: {args.pool} SA runs from random (T0={POOL_T_START}), gap mean {gap_pct(pool_cut, bks).mean():.4f}% "
        f"({time.perf_counter() - t0:.0f}s)")

    idx = match_by_cut(cim_cut, pool_cut)
    sa_init, sa_cut = pool_signs[idx], pool_cut[idx]
    mismatch = np.abs(sa_cut - cim_cut)
    log(f"matched: |cut diff| mean {mismatch.mean():.3f}, max {mismatch.max():.0f}, exact {np.mean(mismatch == 0):.3f}, "
        f"signed mean (SA − CIM) {np.mean(sa_cut - cim_cut):+.3f}")
    d_sa_cim = distance(sa_init, cim_init)
    log(f"distance between matched SA start and CIM start: median {np.median(d_sa_cim):.3f}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    results, comparisons, arrays = {}, {}, {}
    for name, cim_arm in COMPARE.items():
        _, t_start, seed_base = ARMS[cim_arm]
        seeds = seed_base + SWEEPS * 100_000 + np.arange(n_runs, dtype=np.int64)  # CIM 開始と同じ seed
        t1 = time.perf_counter()
        cut, signs = run_sa(ctx, sa_init, seeds, t_start)
        log(f"sa-start {name} (T0={t_start}): gap mean {gap_pct(cut, bks).mean():.4f}% ({time.perf_counter() - t1:.0f}s)")
        cim_final = z[f"best_{cim_arm}_S{SWEEPS}"][:, -1].astype(np.float64)
        results[f"sa_start_{name}"] = arm_summary(sa_cut, cut, distance(sa_init, signs), bks)
        results[f"cim_start_{name}"] = arm_summary(cim_cut, cim_final, dz[f"d_start_{cim_arm}_S{SWEEPS}"], bks)
        comparisons[f"cim_minus_sa_{name}"] = paired_compare(cim_final, cut, rng)
        arrays.update({f"final_cut_sa_{name}": cut, f"final_cut_cim_{name}": cim_final,
                       f"d_start_sa_{name}": distance(sa_init, signs).astype(np.float32)})
    random_final = z[f"best_random_T2_S{SWEEPS}"][:, -1].astype(np.float64)
    results["random_start_T2"] = {"final_gap": describe(gap_pct(random_final, bks))}

    summary = {"meta": {"dataset": meta["dataset"], "bks": bks, "sweeps": SWEEPS, "runs": n_runs, "group": GROUP,
                        "pool": args.pool, "pool_t_start": POOL_T_START, "t_end": T_END,
                        "traj_run": str(TRAJ_RUN), "dist_run": str(DIST_RUN),
                        "match_abs_cut_diff": describe(mismatch), "match_exact_frac": float(np.mean(mismatch == 0)),
                        "match_signed_cut_diff_sa_minus_cim": float(np.mean(sa_cut - cim_cut)),
                        "distance_sa_start_vs_cim_start": describe(d_sa_cim)},
               "results": results, "comparisons": comparisons}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", pool_cut=pool_cut, match_idx=idx, sa_start_cut=sa_cut,
                        cim_start_cut=cim_cut, **arrays)
    for k, v in comparisons.items():
        log(f"{k}: {v['mean_cut_diff']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}] p={v['wilcoxon_p']:.2e}")
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
