"""SA の探索経路を、CIM 開始とランダム開始で比べる(卒論中間報告 結果②)。

問い: 良い地点(CIM の解)から始めると、SA は本当に良い解へ早く / より良く着くのか。

手順(selection_eval の主条件に合わせる):
  1. 独立 seed の CIM を K 回走らせて 1 プールとし、cut の上位 m 個を初期解にする。これを P 組くり返す。
  2. 各初期解から SA を 1 回。ランダム開始も同じ本数。開始地点(CIM / ランダム)と開始温度(0.5 / 2.0)の
     4 条件を回し、開始地点の効果と開始温度の効果を分ける。主条件は selection_eval と同じ
     CIM 0.5 と ランダム 2.0。
  3. 累積最良 cut と現在の cut を、対数等間隔のチェックポイントで記録する。

SA の更新は modules/SA.py の _simulate_sa_batch_warm と同じ(乱数の消費順まで同一)。
起動時に simulate_sa_warm と最終値が完全一致することを確かめる(verify_matches_repo)。
プールが相関を持つため、検定は 16 本 1 組の平均を単位にする。

使い方(プロジェクトルートから):
  python scripts/benchmarks/sa_trajectory.py --dataset G22 --pools 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Callable

os.environ.setdefault("NUMBA_NUM_THREADS", "14")

import numpy as np
from numba import njit, prange
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import _build_csr_sa, simulate_sa_warm
from scripts.benchmarks.algo_registry import load_context, run_cim

EXPERIMENT_KIND = "sa_trajectory"
T_START_CIM, T_START_RANDOM, T_END = 0.5, 2.0, 0.001  # selection_eval と同じ
CIM_ROUNDS = 2400  # ステップ 3・pipeline_timing と同じ
CIM_SEED_START = 1000  # 先頭プールは seed_value_pilot の v12(seed 1000 から)と同じ解になる
REFERENCE_RUN = "results/2026-10-18/seed_value_pilot/v12_G22_K512_R16_cim2400_T0p5_seed1000_S3"
# 条件名: (開始地点, 開始温度, SA seed の基準)。主条件 2 つの seed は v1 と同じ。
ARMS = {"cim_T0p5": ("cim", T_START_CIM, 70_000_000), "random_T2": ("random", T_START_RANDOM, 80_000_000),
        "cim_T2": ("cim", T_START_RANDOM, 90_000_000), "random_T0p5": ("random", T_START_CIM, 60_000_000)}
# (比較名, A, B): A − B を組単位で比べる
COMPARISONS = [("main_cimT0p5_vs_randomT2", "cim_T0p5", "random_T2"),
               ("same_T2_cim_vs_random", "cim_T2", "random_T2"),
               ("same_T0p5_cim_vs_random", "cim_T0p5", "random_T0p5"),
               ("cim_T2_vs_T0p5", "cim_T2", "cim_T0p5")]
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 20261101
EQUAL_TIME_MC = 4000
EQUAL_TIME_RUNS = {"G22": {500: 1115, 5000: 132}}  # スライドの同じ総時間での本数


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


@njit(cache=True, parallel=True)
def sa_traj(n, num_iters, indptr, indices, weights, init, seeds, checkpoints, t_start, t_end):
    """_simulate_sa_batch_warm と同じ SA。best / current の cut をチェックポイントで記録する。"""
    runs = init.shape[0]
    best_out = np.zeros((runs, checkpoints.size), dtype=np.float32)
    cur_out = np.zeros((runs, checkpoints.size), dtype=np.float32)
    final_best = np.zeros(runs, dtype=np.float64)
    log_ratio = np.log(t_end / t_start)
    for r in prange(runs):
        np.random.seed(seeds[r])
        x = init[r].copy()
        cur = 0.0
        for i in range(n):
            for k in range(indptr[i], indptr[i + 1]):
                if indices[k] > i and x[i] != x[indices[k]]:
                    cur += weights[k]
        best = cur
        c = 0
        while c < checkpoints.size and checkpoints[c] == 0:
            best_out[r, c] = best
            cur_out[r, c] = cur
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
            elif T > 0:
                if np.random.random() < np.exp(delta / T):
                    x[v] = -x[v]
                    cur += delta
            if cur > best:
                best = cur
            while c < checkpoints.size and checkpoints[c] == it + 1:
                best_out[r, c] = best
                cur_out[r, c] = cur
                c += 1
        final_best[r] = best
    return best_out, cur_out, final_best


def make_checkpoints(n: int, sweeps: int) -> np.ndarray:
    """0 と、0.01 スイープから最後までの対数等間隔の反転提案回数。最後は必ず num_iters。"""
    grid = np.logspace(-2, np.log10(sweeps), 200)
    iters = np.unique(np.round(grid * n).astype(np.int64))
    return np.unique(np.concatenate([[0], iters, [sweeps * n]]))


def run_sa_paths(ctx, init: np.ndarray, seeds: np.ndarray, sweeps: int, t_start: float):
    indptr, indices, weights = _build_csr_sa(ctx.n, ctx.edges, ctx.weights)
    checkpoints = make_checkpoints(ctx.n, sweeps)
    best, cur, final = sa_traj(ctx.n, sweeps * ctx.n, indptr, indices, weights, np.ascontiguousarray(init), seeds,
                               checkpoints, float(t_start), T_END)
    return checkpoints, best, cur, final


def verify_matches_repo(ctx, log: Callable[[str], None]) -> None:
    """同じ初期解・seed で modules/SA.py と最終最良値が完全一致することを確かめる。"""
    rng = np.random.default_rng(1)
    init = rng.choice(np.array([-1, 1], np.int8), (32, ctx.n))
    seeds = np.arange(32, dtype=np.int64) + 123
    for t_start in (T_START_CIM, T_START_RANDOM):
        repo_cut, repo_signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, 50 * ctx.n, t_start=t_start,
                                                t_end=T_END, seeds=seeds)
        _, best, _, final = run_sa_paths(ctx, init, seeds, 50, t_start)
        if not (np.array_equal(final, repo_cut) and np.array_equal(best[:, -1], repo_cut.astype(np.float32))):
            raise RuntimeError(f"sa_traj does not match modules/SA.py at t_start={t_start}")
        if not np.array_equal(ctx.score(repo_signs).astype(np.float64), repo_cut):
            raise RuntimeError("cut recorded by SA differs from the rescored cut")
    log("verified: sa_traj final best == simulate_sa_warm for 32 runs at both start temperatures")


def collect_pools(ctx, pools: int, K: int, m: int, rounds: int, log: Callable[[str], None]):
    """独立な P プールの CIM を回し、全 cut と各プール上位 m 個の解を返す。"""
    all_cuts = np.empty((pools, K), dtype=np.int32)
    tops = np.empty((pools, m, ctx.n), dtype=np.int8)
    reference = ROOT / REFERENCE_RUN
    ref_cut = np.load(reference / "raw.npz")["init_cut"] if ctx.name == "G22" and rounds == CIM_ROUNDS else None
    t0 = time.perf_counter()
    for p in range(pools):
        seeds = CIM_SEED_START + p * K + np.arange(K, dtype=np.int64)
        _, signs = run_cim(ctx, rounds, K, seeds)
        pm = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
        cut = ctx.score(pm).astype(np.int32)
        all_cuts[p] = cut
        tops[p] = pm[np.argsort(-cut, kind="stable")[:m]]
        if ref_cut is not None and (p + 1) * K <= len(ref_cut):
            if not np.array_equal(cut, ref_cut[p * K:(p + 1) * K].astype(np.int32)):
                raise RuntimeError(f"pool {p} does not reproduce {REFERENCE_RUN}")
            log(f"verified: pool {p} reproduces the CIM cuts of the reference run")
        if p % 10 == 0 or p == pools - 1:
            log(f"CIM pool {p + 1}/{pools}: {time.perf_counter() - t0:.0f}s mean={cut.mean():.1f} top{m}={np.sort(cut)[-m:].mean():.1f}")
    return all_cuts, tops, time.perf_counter() - t0


def bootstrap_diff(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """組単位の平均差 mean(a) - mean(b) の 95% ブートストラップ区間。"""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ia = rng.integers(0, len(a), (BOOTSTRAP, len(a)))
    ib = rng.integers(0, len(b), (BOOTSTRAP, len(b)))
    lo, hi = np.percentile(a[ia].mean(axis=1) - b[ib].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def path_summary(iters: np.ndarray, n: int, best: np.ndarray, bks: float) -> dict:
    sweeps = iters / n
    med = np.median(best, axis=0)
    out = {}
    for s in (0, 1, 10, 100, 500, 1000, 2000, 5000):
        idx = int(np.searchsorted(sweeps, s))
        if idx < len(sweeps) and abs(sweeps[idx] - s) < 1e-9:
            out[str(s)] = {"median_cut": float(med[idx]), "median_gap_pct": float((bks - med[idx]) / bks * 100)}
    return out


def final_stats(final: np.ndarray, m: int, bks: float) -> dict:
    """1 条件の最終最良 cut の要約。final は (組数 * m,)。"""
    return {"n_runs": int(final.size), "mean_cut": float(final.mean()), "sd_cut": float(final.std(ddof=1)),
            "median_cut": float(np.median(final)), "max_cut": float(final.max()), "min_cut": float(final.min()),
            "mean_gap_pct": float((bks - final.mean()) / bks * 100),
            "best_of_m_mean": float(final.reshape(-1, m).max(axis=1).mean()),
            "n_reach_bks": int((final >= bks).sum()),
            "frac_gap_le_0p03pct": float(((bks - final) / bks * 100 <= 0.03).mean())}


def compare_final(a: np.ndarray, b: np.ndarray, m: int) -> dict:
    """a − b の比較。a, b は (組数 * m,) の最終最良 cut。組は独立なので検定は組の平均・最良で行う。"""
    a_g, b_g = a.reshape(-1, m), b.reshape(-1, m)
    mean_a, mean_b = a_g.mean(axis=1), b_g.mean(axis=1)
    best_a, best_b = a_g.max(axis=1), b_g.max(axis=1)
    return {"mean_diff": float(mean_a.mean() - mean_b.mean()), "mean_diff_ci95": list(bootstrap_diff(mean_a, mean_b)),
            "mannwhitney_p_group_means": float(mannwhitneyu(mean_a, mean_b, alternative="two-sided").pvalue),
            "best_of_m_diff": float(best_a.mean() - best_b.mean()),
            "best_of_m_diff_ci95": list(bootstrap_diff(best_a, best_b)),
            "mannwhitney_p_group_best": float(mannwhitneyu(best_a, best_b, alternative="two-sided").pvalue)}


def crossover_sweep(iters: np.ndarray, n: int, best_cim: np.ndarray, best_rnd: np.ndarray) -> float | None:
    """ランダム開始の中央値が CIM 開始の中央値に初めて追いつくスイープ数。追いつかなければ None。"""
    caught = np.median(best_rnd, axis=0) >= np.median(best_cim, axis=0)
    return float(iters[np.argmax(caught)] / n) if caught.any() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22")
    ap.add_argument("--pools", type=int, default=100)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--cim-rounds", type=int, default=CIM_ROUNDS)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[500, 5000])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.pools < 2 or args.m < 1 or args.m > args.pool_size:
        ap.error("need pools >= 2 and 1 <= m <= pool-size")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    kind_root = get_kind_root()
    parts = [args.dataset, f"P{args.pools}", f"K{args.pool_size}", f"m{args.m}", f"cim{args.cim_rounds}",
             "S" + "_".join(map(str, args.sweeps))] + ([args.tag] if args.tag else [])
    out_dir = kind_root / f"v{next_version(kind_root)}_{'_'.join(parts)}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    ctx = load_context(args.dataset)
    bks = float(ctx.bks)
    verify_matches_repo(ctx, log)
    all_cuts, tops, cim_time = collect_pools(ctx, args.pools, args.pool_size, args.m, args.cim_rounds, log)
    cim_init = tops.reshape(-1, ctx.n)
    n_runs = len(cim_init)
    rnd_init = np.random.default_rng(BOOTSTRAP_SEED).choice(np.array([-1, 1], np.int8), (n_runs, ctx.n))

    arrays: dict[str, np.ndarray] = {"all_init_cuts": all_cuts, "top_signs": tops,
                                     "top_init_cuts": ctx.score(cim_init).astype(np.int32),
                                     "random_init_cuts": ctx.score(rnd_init).astype(np.int32)}
    starts = {"cim": cim_init, "random": rnd_init}
    results = {}
    for s in args.sweeps:
        finals, bests, res = {}, {}, {"arms": {}, "comparisons": {}}
        for arm, (start, t_start, seed_base) in ARMS.items():
            t0 = time.perf_counter()
            seeds = seed_base + s * 100_000 + np.arange(n_runs, dtype=np.int64)
            iters, best, cur, final = run_sa_paths(ctx, starts[start], seeds, s, t_start)
            finals[arm], bests[arm] = final, best
            res["arms"][arm] = {**final_stats(final, args.m, bks), "path": path_summary(iters, ctx.n, best, bks),
                                "sa_time_s": time.perf_counter() - t0}
            arrays.update({f"best_{arm}_S{s}": best, f"cur_{arm}_S{s}": cur})
            log(f"S={s} {arm}: {res['arms'][arm]['sa_time_s']:.0f}s mean={final.mean():.2f} sd={final.std(ddof=1):.2f} "
                f"best_of_{args.m}={res['arms'][arm]['best_of_m_mean']:.2f}")
        arrays[f"sweeps_S{s}"] = iters / ctx.n
        for name, a, b in COMPARISONS:
            cmp = compare_final(finals[a], finals[b], args.m)
            cmp["crossover_sweep_b_catches_a"] = crossover_sweep(iters, ctx.n, bests[a], bests[b])
            res["comparisons"][name] = cmp
            log(f"S={s} {name}: mean diff={cmp['mean_diff']:+.2f} {np.round(cmp['mean_diff_ci95'], 2).tolist()} "
                f"p={cmp['mannwhitney_p_group_means']:.3g} | best-of-{args.m} diff={cmp['best_of_m_diff']:+.2f} "
                f"{np.round(cmp['best_of_m_diff_ci95'], 2).tolist()} p={cmp['mannwhitney_p_group_best']:.3g}")
        runs = EQUAL_TIME_RUNS.get(args.dataset, {}).get(s)
        if runs is not None and runs <= n_runs:
            rng = np.random.default_rng(BOOTSTRAP_SEED + s)
            draws = [rng.choice(finals["random_T2"], size=runs, replace=False).max() for _ in range(EQUAL_TIME_MC)]
            res["random_T2_equal_time"] = {"runs": runs, "mean_best": float(np.mean(draws))}
        results[str(s)] = res

    summary = {
        "meta": {"dataset": args.dataset, "bks": ctx.bks, "pools": args.pools, "pool_size": args.pool_size, "m": args.m,
                 "cim_rounds": args.cim_rounds, "cim_seed_start": CIM_SEED_START, "t_start_cim": T_START_CIM,
                 "t_start_random": T_START_RANDOM, "t_end": T_END, "sweeps": args.sweeps, "cim_time_s": cim_time,
                 "threads": int(os.environ["NUMBA_NUM_THREADS"]), "reference_run": REFERENCE_RUN},
        "init": {"all_cim_mean": float(all_cuts.mean()), "all_cim_sd": float(all_cuts.std(ddof=1)),
                 "top_m_mean": float(arrays["top_init_cuts"].mean()),
                 "random_mean": float(arrays["random_init_cuts"].mean())},
        "results": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", **arrays)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
