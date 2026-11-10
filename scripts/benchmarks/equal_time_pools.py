"""同じ総時間での比較(中間発表 結果④)を、結果①(sa_trajectory)と同じデータの取り方でやり直す。

結果④(selection_eval)は、CIM の候補 512 個に各 16 回の SA を事前に回した結果と、ランダム開始 512 本から
抜き出して作っていた。ここでは結果①と同じく、組ごとにすべて新しく回す。

  1 組 = CIM(2400 周回)を独立 seed で K = 256 回 → カット値の上位 m = 16 個から SA を 1 本ずつ → 16 本の最良値
         ランダム開始は、同じ総時間で回せる本数(結果④と同じ本数)の SA を新しく回す → その最良値
  これを P 組(既定 100)くり返し、組を単位に比べる(Mann–Whitney 両側・ブートストラップ区間)。

SA は modules/SA.py の simulate_sa_warm(T0 = 2.0 → 0.001、両者共通)。ランダム開始の本数は結果④の
selection_eval の result.json(runs_per_strategy.random_init)から読む。総時間の定義は結果④と同じ。

G22 は結果①の CIM の組(sa_trajectory v3 の top_signs)をそのまま使える(--cim-pools)。CIM 開始側の SA は
結果①と同じ seed で回すので、結果①の CIM 開始(T0 = 2.0)の値を再現することを確かめる。

使い方(プロジェクトルートから):
  python scripts/benchmarks/equal_time_pools.py --dataset G22 \
      --evals results/2026-10-19/selection_eval/v25_sub_T2_S500_K256_m16 results/2026-10-19/selection_eval/v29_sub_T2_S5000_K256_m16 \
      --cim-pools results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000
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

os.environ.setdefault("NUMBA_NUM_THREADS", "24")

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_trajectory import collect_pools  # noqa: E402

EXPERIMENT_KIND = "equal_time_pools"
T_START, T_END = 2.0, 0.001
CIM_ROUNDS = 2400
SA_SEED_CIM = 90_000_000  # sa_trajectory の cim_T2 と同じ(G22 で結果①を再現する)
SA_SEED_RANDOM = 300_000_000  # ランダム開始(新規)。sweeps * 100_000 + 組 * 2000 + 本 を足す
INIT_SEED_RANDOM = 20261110
MAX_RANDOM_RUNS = 2000
BOOTSTRAP, BOOTSTRAP_SEED = 10_000, 20261110
CHECKPOINT_EVERY = 10


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    heads = [p.name.split("_", 1)[0] for p in kind_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    return max([int(h[1:]) for h in heads if h[1:].isdigit()], default=0) + 1


def random_runs_from_evals(eval_dirs: list[Path], dataset: str, bks: float, sweeps: list[int], pool_size: int,
                           m: int) -> dict:
    """結果④の selection_eval から、各スイープ数でランダム開始が同じ総時間に回せる本数を読む。"""
    runs = {}
    for d in eval_dirs:
        ev = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if abs(ev["t_start"] - T_START) > 1e-9 or ev["pool"] != pool_size or ev["m"] != m:
            raise ValueError(f"{d}: settings differ (t_start={ev['t_start']}, pool={ev['pool']}, m={ev['m']})")
        if dataset not in ev["per_problem"]:
            raise ValueError(f"{d}: no {dataset}")
        pp = ev["per_problem"][dataset]
        runs[int(ev["sweeps"])] = {
            "runs": int(pp["runs_per_strategy"]["random_init"]), "total_time_s": float(pp["total_time_s"]),
            "result4_gap_pct": {k: float((bks - pp["strategies"][k]["mean_best"]) / bks * 100)
                                for k in ("cim_cut", "random_init")},
            "source": str(d)}
    missing = [s for s in sweeps if s not in runs]
    if missing:
        raise ValueError(f"no selection_eval result for sweeps {missing}")
    for s in sweeps:
        if not 1 <= runs[s]["runs"] <= MAX_RANDOM_RUNS:
            raise ValueError(f"random runs {runs[s]['runs']} at {s} sweeps outside 1..{MAX_RANDOM_RUNS}")
    return runs


def load_cim_pools(src: Path, ctx, pools: int, pool_size: int, m: int, rounds: int) -> np.ndarray:
    """結果①(sa_trajectory)の各組の上位 m 個の解を読み、設定が同じか確かめる。"""
    meta = json.loads((src / "summary.json").read_text(encoding="utf-8"))["meta"]
    want = {"dataset": ctx.name, "pool_size": pool_size, "m": m, "cim_rounds": rounds}
    got = {k: meta[k] for k in want}
    if got != want or meta["pools"] < pools:
        raise ValueError(f"{src}: settings {got} (pools={meta['pools']}) differ from {want} (pools={pools})")
    return np.load(src / "raw.npz")["top_signs"][:pools]


def run_sa(ctx, init: np.ndarray, seeds: np.ndarray, sweeps: int) -> np.ndarray:
    cuts, _ = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init, sweeps * ctx.n, t_start=T_START, t_end=T_END,
                               seeds=seeds)
    return np.asarray(cuts, dtype=np.float64)


def bootstrap_diff(a: np.ndarray, b: np.ndarray) -> list[float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ia = rng.integers(0, len(a), (BOOTSTRAP, len(a)))
    ib = rng.integers(0, len(b), (BOOTSTRAP, len(b)))
    lo, hi = np.percentile(a[ia].mean(axis=1) - b[ib].mean(axis=1), [2.5, 97.5])
    return [float(lo), float(hi)]


def arm_stats(best: np.ndarray, per_run_mean: float, bks: float) -> dict:
    gap = (bks - best) / bks * 100
    se = gap.std(ddof=1) / np.sqrt(len(gap))
    return {"mean_best_cut": float(best.mean()), "sd_best_cut": float(best.std(ddof=1)),
            "gap_pct": float(gap.mean()), "gap_pct_ci95": [float(gap.mean() - 1.96 * se), float(gap.mean() + 1.96 * se)],
            "per_run_mean_gap_pct": float((bks - per_run_mean) / bks * 100),
            "n_reach_bks": int((best >= bks).sum())}


def summarize(cim_final: np.ndarray, rnd_best: np.ndarray, rnd_mean: np.ndarray, bks: float) -> dict:
    """組ごとの最良値で比べる。差は CIM 開始 − ランダム開始(カット値、正なら CIM 開始が良い)。"""
    cim_best = cim_final.max(axis=1)
    return {"pools": int(len(cim_best)),
            "cim": arm_stats(cim_best, float(cim_final.mean()), bks),
            "random": arm_stats(rnd_best, float(rnd_mean.mean()), bks),
            "diff_best_cut": float(cim_best.mean() - rnd_best.mean()),
            "diff_best_cut_ci95": bootstrap_diff(cim_best, rnd_best),
            "mannwhitney_p": float(mannwhitneyu(cim_best, rnd_best, alternative="two-sided").pvalue)}


def save(out_dir: Path, summary: dict, arrays: dict) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", **arrays)


def run_pools(ctx, tops: np.ndarray, sweeps: list[int], rnd_runs: dict, log: Callable[[str], None],
              checkpoint: Callable[[int, dict], None]) -> dict:
    pools, m = tops.shape[0], tops.shape[1]
    arrays = {"top_signs": tops}
    for s in sweeps:
        arrays[f"cim_final_S{s}"] = np.zeros((pools, m))
        arrays[f"random_final_S{s}"] = np.zeros((pools, rnd_runs[s]["runs"]))
    init_rng = np.random.default_rng(INIT_SEED_RANDOM)
    t0 = time.perf_counter()
    for p in range(pools):
        for s in sweeps:
            # 結果①と同じ seed:組 p の 16 本は通し番号 p*m .. p*m+m-1
            seeds = SA_SEED_CIM + s * 100_000 + p * m + np.arange(m, dtype=np.int64)
            arrays[f"cim_final_S{s}"][p] = run_sa(ctx, tops[p], seeds, s)
            n_r = rnd_runs[s]["runs"]
            init = init_rng.choice(np.array([-1, 1], np.int8), (n_r, ctx.n))
            seeds = SA_SEED_RANDOM + s * 100_000 + p * MAX_RANDOM_RUNS + np.arange(n_r, dtype=np.int64)
            arrays[f"random_final_S{s}"][p] = run_sa(ctx, init, seeds, s)
        line = " ".join(f"S{s}: cim {arrays[f'cim_final_S{s}'][p].max():.0f} rnd {arrays[f'random_final_S{s}'][p].max():.0f}"
                        for s in sweeps)
        log(f"pool {p + 1}/{pools} {time.perf_counter() - t0:.0f}s {line}")
        if (p + 1) % CHECKPOINT_EVERY == 0 or p == pools - 1:
            checkpoint(p + 1, arrays)
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--evals", type=Path, nargs="+", required=True, help="結果④の selection_eval の出力ディレクトリ")
    ap.add_argument("--pools", type=int, default=100)
    ap.add_argument("--pool-size", type=int, default=256)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[500, 5000])
    ap.add_argument("--cim-pools", type=Path, default=None, help="結果①の sa_trajectory 出力(G22 で CIM を再利用)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.pools < 2 or not 1 <= args.m <= args.pool_size:
        ap.error("need pools >= 2 and 1 <= m <= pool-size")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    ctx = load_context(args.dataset)
    bks = float(ctx.bks)
    rnd_runs = random_runs_from_evals(args.evals, args.dataset, bks, args.sweeps, args.pool_size, args.m)

    kind_root = get_kind_root()
    parts = [args.dataset, f"P{args.pools}", f"K{args.pool_size}", f"m{args.m}", f"cim{CIM_ROUNDS}", "T2",
             "S" + "_".join(map(str, args.sweeps))] + ([args.tag] if args.tag else [])
    out_dir = kind_root / f"v{next_version(kind_root)}_{'_'.join(parts)}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"{args.dataset}: random-start runs per pool {{sweeps: runs}} = "
        f"{ {s: rnd_runs[s]['runs'] for s in args.sweeps} } (from selection_eval)")
    if args.cim_pools is not None:
        tops = load_cim_pools(args.cim_pools, ctx, args.pools, args.pool_size, args.m, CIM_ROUNDS)
        cim_time, cim_source = None, str(args.cim_pools)
        log(f"CIM pools loaded from {args.cim_pools}")
    else:
        _, tops, cim_time = collect_pools(ctx, args.pools, args.pool_size, args.m, CIM_ROUNDS, log)
        cim_source = "new"

    meta = {"dataset": args.dataset, "bks": bks, "pools": args.pools, "pool_size": args.pool_size, "m": args.m,
            "cim_rounds": CIM_ROUNDS, "cim_source": cim_source, "cim_time_s": cim_time, "t_start": T_START,
            "t_end": T_END, "sweeps": args.sweeps, "random_runs": rnd_runs,
            "sa_seed_cim": SA_SEED_CIM, "sa_seed_random": SA_SEED_RANDOM, "init_seed_random": INIT_SEED_RANDOM,
            "threads": int(os.environ["NUMBA_NUM_THREADS"])}

    def build_summary(done: int, arrays: dict) -> dict:
        res = {}
        for s in args.sweeps:
            rf = arrays[f"random_final_S{s}"][:done]
            res[str(s)] = summarize(arrays[f"cim_final_S{s}"][:done], rf.max(axis=1), rf.mean(axis=1), bks)
        return {"meta": {**meta, "pools_done": done}, "results": res}

    def checkpoint(done: int, arrays: dict) -> None:
        save(out_dir, build_summary(done, arrays), arrays)

    arrays = run_pools(ctx, tops, args.sweeps, rnd_runs, log, checkpoint)
    summary = build_summary(args.pools, arrays)

    if args.cim_pools is not None:  # 結果①の CIM 開始(T0 = 2.0)を再現しているか
        ref = np.load(args.cim_pools / "raw.npz")
        for s in args.sweeps:
            key = f"best_cim_T2_S{s}"
            if key in ref.files:
                same = np.array_equal(ref[key][: args.pools * args.m, -1].astype(np.float64).reshape(args.pools, args.m),
                                      arrays[f"cim_final_S{s}"])
                summary["results"][str(s)]["reproduces_result1_cim"] = bool(same)
                log(f"S={s}: CIM-start finals {'reproduce' if same else 'DO NOT reproduce'} {key}")
    save(out_dir, summary, arrays)
    for s in args.sweeps:
        r = summary["results"][str(s)]
        log(f"S={s}: CIM gap {r['cim']['gap_pct']:.4f}% vs random {r['random']['gap_pct']:.4f}% "
            f"(runs {rnd_runs[s]['runs']}) diff {r['diff_best_cut']:+.2f} cut {np.round(r['diff_best_cut_ci95'], 2).tolist()} "
            f"p={r['mannwhitney_p']:.3g} | per-run CIM {r['cim']['per_run_mean_gap_pct']:.4f}% "
            f"random {r['random']['per_run_mean_gap_pct']:.4f}%")
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
