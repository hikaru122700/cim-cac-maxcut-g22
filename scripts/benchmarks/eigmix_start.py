"""eigmix_start.py — CIM を使わず「固有値の大きい固有ベクトルをランダムに混ぜた解」から SA を回すと、どこまで行くか(実験 G、G22)。

11/16 進捗報告会の教授の案:「固有値の大きいベクトルのセット(5〜10 個)をランダムに作って SA に入れれば、
CIM でなくてもよいかもしれない」。実験 C2(sign(u1) 1 本だけ → 0.155%)の一般化。

開始解 s = sign(Σ_{k=1..K} g_k u_k)、g_k は標準正規乱数(1 本ごとに新しく引く)。K = 1, 5, 10, 50。
重みを固有値で付けた版(w_k = exp(τ(λ_k − λ_1))、CIM の線形段で各モードが育つ割合を模したもの)も K = 50 で回す。
SA は G22 の実験 A〜D と同じ(T0 = 2.0、5000 スイープ、終了温度 0.001)。各 1600 本。

  python scripts/benchmarks/eigmix_start.py [--runs 1600]
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

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.algo_registry import PARAMS, load_context  # noqa: E402
from scripts.benchmarks.eigvec_valley import top_eigvecs  # noqa: E402
from scripts.benchmarks.sa_path_bks_distance import load_refs  # noqa: E402
from scripts.benchmarks.start_distance_causal import describe, gap_pct, nearest, run_sa  # noqa: E402

EXPERIMENT_KIND = "eigmix_start"
K_MAX = 50
ARMS = (("K1", 1, False), ("K5", 5, False), ("K10", 10, False), ("K50", 50, False), ("K50_grow", 50, True))
# --select:CIM と同じく 256 本作ってカット値上位 16 本を選ぶ(選び方の効果をそろえる)
SELECT_ARMS = (("K10_sel", 10, False), ("K50_grow_sel", 50, True))
POOL, TOP = 256, 16
MIX_SEED = 20261116
SA_SEED_BASE = 160_000_000   # 既存の seed(60M〜150M 台)と重ならない
VALLEY_EDGE = 0.3
LINEAR_ROUNDS = 236          # 実験 D の線形段の終わり(中央値)


def growth_tau(vals: np.ndarray) -> float:
    """CIM の線形段で、モード k の振幅は (√η + λ_k)^n に比例して育つ(J_cim の固有値 λ_k)。
    n = 236 ラウンド後の比 ((√η + λ_k)/(√η + λ_1))^n = exp(τ(λ_k − λ_1)) となる τ を、上位 50 本で最小二乗で決める。"""
    a = np.sqrt(PARAMS["G22"]["CIM"]["eta"])
    log_ratio = LINEAR_ROUNDS * (np.log(a + vals) - np.log(a + vals[0]))
    dl = vals - vals[0]
    return float((dl @ log_ratio) / (dl @ dl)) if np.any(dl) else 0.0


def make_starts(vecs: np.ndarray, vals: np.ndarray, k: int, grow: bool, runs: int,
                rng: np.random.Generator, tau: float) -> np.ndarray:
    g = rng.standard_normal((runs, k))
    if grow:
        g = g * np.exp(tau * (vals[:k] - vals[0]))[None, :]
    return np.where(g @ vecs[:, :k].T > 0, 1, -1).astype(np.int8)


def make_selected(ctx, vecs: np.ndarray, vals: np.ndarray, k: int, grow: bool, runs: int,
                  rng: np.random.Generator, tau: float) -> np.ndarray:
    """256 本ずつ作り、各組でカット値上位 16 本を残す(runs / 16 組)。"""
    out = []
    for _ in range(runs // TOP):
        pool = make_starts(vecs, vals, k, grow, POOL, rng, tau)
        out.append(pool[np.argsort(-ctx.score(pool), kind="stable")[:TOP]])
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1600)
    ap.add_argument("--select", action="store_true", help="256 本から上位 16 本を選ぶ条件だけを回す")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.runs < 16 or args.runs % TOP:
        ap.error("runs must be a positive multiple of 16")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")
    ctx = load_context("G22")
    refs = load_refs(ctx)
    vals, vecs = top_eigvecs(ctx.J_cim, k=K_MAX)
    tau = growth_tau(vals)
    rng = np.random.default_rng(MIX_SEED)
    res, arrays = {}, {}
    arms = SELECT_ARMS if args.select else ARMS
    offset = 10 if args.select else 0
    for j, (name, k, grow) in enumerate(arms):
        t0 = time.perf_counter()
        init = (make_selected(ctx, vecs, vals, k, grow, args.runs, rng, tau) if args.select
                else make_starts(vecs, vals, k, grow, args.runs, rng, tau))
        cut, signs = run_sa(ctx, init, SA_SEED_BASE + (j + offset) * 1_000_000 + np.arange(args.runs, dtype=np.int64))
        d0, d1 = nearest(init, refs), nearest(signs, refs)
        g = gap_pct(cut, ctx.bks)
        res[name] = {"K": k, "grow": grow, "start_gap": describe(gap_pct(ctx.score(init), ctx.bks)),
                     "start_d": describe(d0), "start_in_valley": float((d0 < VALLEY_EDGE).mean()),
                     "final_gap": describe(g), "final_gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
                     "final_d": describe(d1), "final_in_valley": float((d1 < VALLEY_EDGE).mean())}
        arrays.update({f"{name}_start_d": d0, f"{name}_final_cut": cut, f"{name}_final_d": d1})
        print(f"{name}: start d median {np.median(d0):.3f} (valley {res[name]['start_in_valley']:.1%}) -> final "
              f"{g.mean():.4f}% ± {res[name]['final_gap_ci95']:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_", 1)[0][1:]) for p in kind_root.iterdir()
            if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    out = kind_root / (f"v{max(nums, default=0) + 1}_G22_runs{args.runs}" + ("_select" if args.select else "") + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True)
    summary = {"meta": {"dataset": "G22", "runs": args.runs, "arms": [list(a) for a in arms], "select": args.select, "tau": tau,
                        "eigenvalues": vals.tolist(), "mix_seed": MIX_SEED, "sa_seed_base": SA_SEED_BASE,
                        "sa": {"t_start": 2.0, "sweeps": 5000, "t_end": 0.001}},
               "results": res}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / "raw.npz", eigvals=vals, **arrays)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
