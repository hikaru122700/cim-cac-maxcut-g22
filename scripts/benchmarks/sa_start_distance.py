"""SA の開始解と最終解(最良解)の距離を測る(sa_trajectory の続き)。

問い: CIM から始めた SA は、開始温度 2.0 でも CIM 解の近くに残るのか。開始温度 0.5 では
「初期解の近くにとどまる」のか。

sa_trajectory の run(既定は v3、スライドの図と同じ)と同じ初期解・同じ SA seed で modules/SA.py の
simulate_sa_warm を回し直し、最良解の符号を得る。最良 cut が sa_trajectory の最終値と完全一致する
ことを確かめてから、距離を計算する。距離は全反転を同一視した正規化ハミング距離 min(h, n − h) / n
(0 = 同じ解、0.5 = 無関係な解と同程度)。

  start    : 自分の開始解との距離
  other    : 別のプールの CIM 解(無関係な CIM 解)との距離。start < other なら、自分の開始解の情報が残っている
  best     : 全条件で見つかった最良解との距離(どれだけ良い谷に入ったか)

使い方(プロジェクトルートから):
  python scripts/benchmarks/sa_start_distance.py
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
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS, BOOTSTRAP_SEED, T_END  # noqa: E402

EXPERIMENT_KIND = "sa_start_distance"
DEFAULT_RUN = "results/2026-11-01/sa_trajectory/v3_G22_P100_K256_m16_cim2400_S500_5000"


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


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """行ごとの正規化ハミング距離(全反転を同一視)。a, b は (runs, n) の ±1。"""
    h = (a != b).mean(axis=1)
    return np.minimum(h, 1.0 - h)


def describe(x: np.ndarray) -> dict:
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=Path(DEFAULT_RUN), help="sa_trajectory の出力")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    meta = json.loads((args.run / "summary.json").read_text(encoding="utf-8"))["meta"]
    z = np.load(args.run / "raw.npz")
    ctx = load_context(meta["dataset"])
    bks = float(ctx.bks)
    m = meta["m"]
    cim_init = z["top_signs"].reshape(-1, ctx.n).astype(np.int8)
    n_runs = len(cim_init)
    rnd_init = np.random.default_rng(BOOTSTRAP_SEED).choice(np.array([-1, 1], np.int8), (n_runs, ctx.n))
    if not np.array_equal(ctx.score(rnd_init).astype(np.int32), z["random_init_cuts"]):
        raise RuntimeError("random starts do not reproduce sa_trajectory")
    starts = {"cim": cim_init, "random": rnd_init}
    # 別のプールの CIM 解:プールを 1 つずらして対応させる(同じプールの解とは組ませない)
    other_cim = np.roll(z["top_signs"], 1, axis=0).reshape(-1, ctx.n)

    kind_root = get_kind_root()
    desc = f"{meta['dataset']}_from_{args.run.name.split('_', 1)[0]}" + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    finals: dict[str, np.ndarray] = {}
    cuts: dict[str, np.ndarray] = {}
    for s in meta["sweeps"]:
        for arm, (start, t_start, seed_base) in ARMS.items():
            t0 = time.perf_counter()
            seeds = seed_base + s * 100_000 + np.arange(n_runs, dtype=np.int64)
            cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, starts[start], s * ctx.n,
                                          t_start=t_start, t_end=T_END, seeds=seeds)
            if not np.array_equal(cut.astype(np.float32), z[f"best_{arm}_S{s}"][:, -1]):
                raise RuntimeError(f"S={s} {arm}: final cuts differ from {args.run}")
            signs = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
            if not np.array_equal(ctx.score(signs).astype(np.float64), np.asarray(cut, dtype=np.float64)):
                raise RuntimeError(f"S={s} {arm}: rescored cut differs")
            finals[f"{arm}_S{s}"], cuts[f"{arm}_S{s}"] = signs, np.asarray(cut, dtype=np.float64)
            log(f"S={s} {arm}: verified {n_runs} runs ({time.perf_counter() - t0:.0f}s)")

    best_key = max(cuts, key=lambda k: cuts[k].max())
    best_idx = int(cuts[best_key].argmax())
    best_sol = finals[best_key][best_idx]
    best_cut = float(cuts[best_key][best_idx])
    log(f"reference best: {best_cut:.0f} (gap {(bks - best_cut) / bks * 100:.4f}%) from {best_key}")

    results = {}
    arrays = {}
    for key, signs in finals.items():
        arm = key.rsplit("_S", 1)[0]
        start = starts[ARMS[arm][0]]
        d_start = distance(start, signs)
        d_other = distance(other_cim, signs)
        d_best = distance(np.broadcast_to(best_sol, signs.shape), signs)
        gap = (bks - cuts[key]) / bks * 100
        rho, p = spearmanr(d_start, cuts[key])
        results[key] = {"start": describe(d_start), "other_cim": describe(d_other), "best": describe(d_best),
                        "frac_closer_to_own_start": float((d_start < d_other).mean()),
                        "spearman_start_vs_cut": {"rho": float(rho), "p": float(p)},
                        "mean_gap_pct": float(gap.mean())}
        arrays.update({f"d_start_{key}": d_start.astype(np.float32), f"d_other_{key}": d_other.astype(np.float32),
                       f"d_best_{key}": d_best.astype(np.float32), f"cut_{key}": cuts[key]})
        r = results[key]
        log(f"{key:>18}: start {r['start']['median']:.3f} [{r['start']['q1']:.3f}, {r['start']['q3']:.3f}] "
            f"other-CIM {r['other_cim']['median']:.3f} best {r['best']['median']:.3f} "
            f"closer-to-own {r['frac_closer_to_own_start']:.2f} rho(start,cut)={rho:+.2f} gap {gap.mean():.4f}%")
    # 参考:開始解どうしの距離(CIM 解どうし、CIM と最良解)
    base = {"cim_start_vs_other_cim_start": describe(distance(cim_init, other_cim)),
            "cim_start_vs_best": describe(distance(cim_init, np.broadcast_to(best_sol, cim_init.shape))),
            "random_start_vs_best": describe(distance(rnd_init, np.broadcast_to(best_sol, rnd_init.shape)))}
    for k, v in base.items():
        log(f"{k}: median {v['median']:.3f} [{v['q1']:.3f}, {v['q3']:.3f}]")

    summary = {"meta": {"source_run": str(args.run), "dataset": meta["dataset"], "bks": bks, "m": m, "runs": n_runs,
                        "sweeps": meta["sweeps"], "reference_best_cut": best_cut, "reference_from": best_key,
                        "threads": int(os.environ["NUMBA_NUM_THREADS"])},
               "baseline": base, "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", **arrays)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
