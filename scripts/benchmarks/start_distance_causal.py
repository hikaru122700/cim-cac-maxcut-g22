"""SA の到達点を決めているのは「開始解と BKS 解の谷との距離」か(G22、実験 B)。

問い(11/9 進捗報告のフィードバック「感想でなく、裏付けの計算を」): 結果③では CIM の解が BKS 解の谷に近かった。
CIM 開始の SA が良いのは、本当にこの距離のためか。2 つの計算で確かめる。

B1. 結果②の読み直し(カット値をそろえた比較):
  sa_matched_start v2 の「カット値をそろえた SA の解」1600 個を同じ seed で作り直し(カット値の一致を確認)、
  BKS 解の谷までの距離を測る。CIM の解(同じカット値)より遠ければ、結果②の差は距離で説明できる候補になる。
  さらに、最後の BKS との差を「開始解の種類(CIM / SA の解)」と「開始解の距離」で回帰し、
  距離をそろえると種類の差が消えるかを見る。

B2. 距離だけを動かす介入:
  集めた BKS 解を 1 つ選び、決まった数のスピンをランダムに反転して、BKS 解から距離 d の開始解を作る。
  d を 0〜0.5 で振り、結果①と同じ SA(T0 = 2.0、5000 スイープ、終了温度 0.001)を回す。
  カット値は CIM の解よりずっと悪いが、到達点が d だけで決まり、d = 0.16(CIM の解の距離)で CIM 開始と
  同じくらいになれば、「谷との距離」が効いていると言える。

距離は全反転を同一視した正規化ハミング距離 min(h, n − h) / n。BKS 解の集合は bks_set_distance v3(89 個)。

  python scripts/benchmarks/start_distance_causal.py
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
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.bks_set_distance import dist_matrix  # noqa: E402
from scripts.benchmarks.sa_matched_start import POOL_SEED_BASE, POOL_T_START  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS, T_END  # noqa: E402

EXPERIMENT_KIND = "start_distance_causal"
TRAJ_RUN = Path("results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000")
MATCH_RUN = Path("results/2026-11-08/sa_matched_start/v2_G22_pool19200_S5000")
BKS_RUN = Path("results/2026-11-11/bks_set_distance/v3_G22_S42104_target100")
SWEEPS = 5000
T_START = 2.0
CIM_ARM = "cim_T2"  # B1 の SA は CIM 開始と同じ seed(sa_matched_start と同じ)
D_GRID = (0.0, 0.02, 0.05, 0.10, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50)
PERTURB_SEED = 20261112
PERTURB_SA_SEED_BASE = 130_000_000  # 既存の seed(60M〜110M 台)と重ならない
BINS = (0.0, 0.10, 0.13, 0.16, 0.19, 0.22, 0.25, 0.30, 0.50)


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


def gap_pct(cut: np.ndarray, bks: float) -> np.ndarray:
    return (bks - np.asarray(cut, dtype=np.float64)) / bks * 100


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def nearest(sol: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return dist_matrix(sol, refs).min(axis=1).astype(np.float64)


def run_sa(ctx, init: np.ndarray, seeds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, np.ascontiguousarray(init), SWEEPS * ctx.n,
                                  t_start=T_START, t_end=T_END, seeds=seeds)
    signs = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
    cut = np.asarray(cut, dtype=np.float64)
    if not np.array_equal(ctx.score(signs).astype(np.float64), cut):
        raise RuntimeError("rescored cut differs from SA's best cut")
    return cut, signs


def perturb(base: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """base のスピンを k 個(重複なし)反転した新しい配列を返す。"""
    out = base.copy()
    idx = rng.choice(base.size, size=k, replace=False)
    out[idx] = -out[idx]
    return out


def ols(y: np.ndarray, cols: dict[str, np.ndarray]) -> dict:
    """y = b0 + Σ b_j x_j の最小二乗。係数と通常の標準誤差を返す。"""
    names = ["intercept", *cols]
    X = np.column_stack([np.ones_like(y), *cols.values()])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {nm: {"coef": float(b), "se": float(e)} for nm, b, e in zip(names, beta, se)}


def binned(d: np.ndarray, gap: np.ndarray) -> list:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() >= 10:
            g = gap[m]
            rows.append({"lo": lo, "hi": hi, "n": int(m.sum()), "gap_mean": float(g.mean()),
                         "gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size))})
    return rows


def part_b1(ctx, refs: np.ndarray, log: Callable[[str], None]) -> tuple[dict, dict]:
    """結果②の SA の解を作り直し、距離で説明できるかを見る。"""
    bks = float(ctx.bks)
    mz = np.load(MATCH_RUN / "raw.npz")
    idx = mz["match_idx"]
    pool_size = mz["pool_cut"].size
    rnd = np.random.default_rng(POOL_SEED_BASE).choice(np.array([-1, 1], np.int8), (pool_size, ctx.n))[idx]
    if POOL_T_START != T_START:
        raise RuntimeError("pool temperature changed")
    t0 = time.perf_counter()
    sa_cut, sa_init = run_sa(ctx, rnd, POOL_SEED_BASE + idx.astype(np.int64))
    if not np.array_equal(sa_cut, mz["sa_start_cut"]):
        raise RuntimeError("matched SA starts do not reproduce sa_matched_start v2")
    log(f"B1: rebuilt {len(idx)} matched SA starts, cuts identical ({time.perf_counter() - t0:.0f}s)")
    _, _, seed_base = ARMS[CIM_ARM]
    seeds = seed_base + SWEEPS * 100_000 + np.arange(len(idx), dtype=np.int64)
    fin_cut, fin_signs = run_sa(ctx, sa_init, seeds)
    if not np.array_equal(fin_cut, mz["final_cut_sa_T2"]):
        raise RuntimeError("SA from matched SA starts does not reproduce sa_matched_start v2")
    log(f"B1: SA from matched SA starts reproduces final cuts ({time.perf_counter() - t0:.0f}s)")

    bz = np.load(BKS_RUN / "raw.npz")
    d_cim_start, d_cim_final = bz["d_nearest_cim_start"].astype(np.float64), bz["d_nearest_cim_sa"].astype(np.float64)
    cim_final = mz["final_cut_cim_T2"]
    d_sa_start, d_sa_final = nearest(sa_init, refs), nearest(fin_signs, refs)
    g_cim, g_sa = gap_pct(cim_final, bks), gap_pct(fin_cut, bks)

    d_all = np.concatenate([d_cim_start, d_sa_start])
    g_all = np.concatenate([g_cim, g_sa])
    is_cim = np.concatenate([np.ones_like(d_cim_start), np.zeros_like(d_sa_start)])
    start_gap = np.concatenate([gap_pct(mz["cim_start_cut"], bks), gap_pct(mz["sa_start_cut"], bks)])
    res = {
        "start_d_bks": {"cim": describe(d_cim_start), "sa_solution": describe(d_sa_start)},
        "final_d_bks": {"cim": describe(d_cim_final), "sa_solution": describe(d_sa_final)},
        "final_gap": {"cim": describe(g_cim), "sa_solution": describe(g_sa)},
        "start_distance_mannwhitney_p": float(mannwhitneyu(d_cim_start, d_sa_start).pvalue),
        "spearman_start_d_vs_final_gap": {
            "cim": float(spearmanr(d_cim_start, g_cim).statistic),
            "sa_solution": float(spearmanr(d_sa_start, g_sa).statistic),
            "pooled": float(spearmanr(d_all, g_all).statistic)},
        "ols_source_only": ols(g_all, {"is_cim": is_cim}),
        "ols_source_and_distance": ols(g_all, {"is_cim": is_cim, "start_d_bks": d_all}),
        "ols_source_distance_startgap": ols(g_all, {"is_cim": is_cim, "start_d_bks": d_all, "start_gap": start_gap}),
        "binned": {"cim": binned(d_cim_start, g_cim), "sa_solution": binned(d_sa_start, g_sa)},
    }
    log(f"B1: start d_bks median CIM {np.median(d_cim_start):.3f} vs SA-solution {np.median(d_sa_start):.3f}; "
        f"source effect {res['ols_source_only']['is_cim']['coef']:+.4f} -> "
        f"{res['ols_source_and_distance']['is_cim']['coef']:+.4f} after distance")
    arrays = {"b1_d_sa_start": d_sa_start, "b1_d_sa_final": d_sa_final, "b1_final_cut_sa": fin_cut,
              "b1_d_cim_start": d_cim_start, "b1_final_cut_cim": cim_final}
    return res, arrays


def part_b2(ctx, refs: np.ndarray, runs: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    """BKS 解から距離 d だけ崩した開始解で SA を回す。"""
    bks = float(ctx.bks)
    rng = np.random.default_rng(PERTURB_SEED)
    res, arrays = {}, {}
    for j, d in enumerate(D_GRID):
        k = int(round(d * ctx.n))
        init = np.stack([perturb(refs[i % len(refs)], k, rng) for i in range(runs)])
        seeds = PERTURB_SA_SEED_BASE + j * 1_000_000 + np.arange(runs, dtype=np.int64)
        t0 = time.perf_counter()
        cut, signs = run_sa(ctx, init, seeds)
        start_cut = ctx.score(init).astype(np.float64)
        d0, d1 = nearest(init, refs), nearest(signs, refs)
        g = gap_pct(cut, bks)
        res[f"{d:.2f}"] = {"flipped_spins": k, "start_gap": describe(gap_pct(start_cut, bks)),
                           "start_d_bks": describe(d0), "final_gap": describe(g),
                           "final_gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
                           "frac_reach_bks": float((cut >= bks).mean()), "frac_within_0p03pct": float((g <= 0.03).mean()),
                           "final_d_bks": describe(d1), "final_in_valley": float((d1 < 0.05).mean())}
        arrays.update({f"b2_d{j}_start_cut": start_cut, f"b2_d{j}_final_cut": cut,
                       f"b2_d{j}_d_start": d0, f"b2_d{j}_d_final": d1})
        log(f"B2 d={d:.2f} (k={k}): start gap {gap_pct(start_cut, bks).mean():.2f}% -> final {g.mean():.4f}% "
            f"± {res[f'{d:.2f}']['final_gap_ci95']:.4f}; final d_bks median {np.median(d1):.3f} "
            f"({time.perf_counter() - t0:.0f}s)")
    return res, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-per-d", type=int, default=960)
    ap.add_argument("--skip-b1", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.runs_per_d < 16:
        ap.error("runs-per-d must be >= 16")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    ctx = load_context("G22")
    bz = np.load(BKS_RUN / "raw.npz")
    refs = np.unique(np.vstack([bz["bks_signs"], bz["reference"][None, :]]), axis=0).astype(np.int8)
    if not np.all(ctx.score(refs).astype(np.float64) == float(ctx.bks)):
        raise RuntimeError("reference set contains a non-BKS solution")

    kind_root = get_kind_root()
    desc = f"G22_S{SWEEPS}_T2_runs{args.runs_per_d}" + ("_noB1" if args.skip_b1 else "") + \
        (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"BKS reference set: {len(refs)}")
    results, arrays = {}, {}
    if not args.skip_b1:
        results["b1_matched"], a1 = part_b1(ctx, refs, log)
        arrays.update(a1)
    results["b2_perturbed"], a2 = part_b2(ctx, refs, args.runs_per_d, log)
    arrays.update(a2)
    tz = np.load(TRAJ_RUN / "raw.npz")
    results["reference_points"] = {
        "cim_T2_final_gap_mean": float(gap_pct(tz[f"best_cim_T2_S{SWEEPS}"][:, -1], ctx.bks).mean()),
        "random_T2_final_gap_mean": float(gap_pct(tz[f"best_random_T2_S{SWEEPS}"][:, -1], ctx.bks).mean()),
        "cim_start_d_bks_median": float(np.median(bz["d_nearest_cim_start"])),
        "random_start_d_bks_median": float(np.median(bz["d_nearest_random_start"]))}

    summary = {"meta": {"dataset": "G22", "bks": float(ctx.bks), "sweeps": SWEEPS, "t_start": T_START, "t_end": T_END,
                        "d_grid": list(D_GRID), "runs_per_d": args.runs_per_d, "reference_set_size": int(len(refs)),
                        "perturb_seed": PERTURB_SEED, "perturb_sa_seed_base": PERTURB_SA_SEED_BASE,
                        "traj_run": str(TRAJ_RUN), "match_run": str(MATCH_RUN), "bks_run": str(BKS_RUN),
                        "threads": int(os.environ["NUMBA_NUM_THREADS"])},
               "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", d_grid=np.array(D_GRID), **arrays)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
