"""eigvec_valley.py — 実験 C:BKS 解側の谷と J のトップ固有ベクトルの関係(G22)。

仮説(CIM の式から):飽和前の CIM は c ← e^{g/2}(√η I + J)c + ξ の線形写像なので、J の最大固有値の
固有ベクトル u1 の方向が一番速く育つ。だから CIM の解は sign(u1) 側(BKS 解側の谷)に出る。

C1(位置):BKS 解・CIM の解・ランダム開始 SA の解・カット値をそろえた SA の解について、
    u1, u2 との重なりを測り、谷(最も近い BKS 解までの距離 0.3 未満 / 以上)ごとに比べる。
C2(介入・対照):CIM を使わず、スペクトル緩和の解 sign(u1)(と対照の sign(u2), sign(u3))から
    同じ SA(T0 = 2.0、5000 スイープ)を回す。sign(u1) から始めて CIM 開始と同じ質になれば、
    CIM の良さは「u1 方向を選ぶこと」で説明できる。

重なり:ov_k(s) = |<s, sign(u_k)>| / n(全反転対称、0〜1)。E = ov_1 − ov_2(10/18 の事前登録と同じ定義)。

usage(プロジェクトルートから):
    NUMBA_NUM_THREADS=24 .venv/Scripts/python.exe -u scripts/benchmarks/eigvec_valley.py [--runs 960]
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
from scipy.sparse.linalg import eigsh  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.algo_registry import load_context  # noqa: E402
from scripts.benchmarks.sa_matched_start import POOL_SEED_BASE  # noqa: E402
from scripts.benchmarks.sa_path_bks_distance import load_refs, load_starts  # noqa: E402
from scripts.benchmarks.sa_trajectory import ARMS  # noqa: E402
from scripts.benchmarks.start_distance_causal import (MATCH_RUN, SWEEPS, TRAJ_RUN, describe, gap_pct,  # noqa: E402
                                                      nearest, run_sa)

EXPERIMENT_KIND = "eigvec_valley"
N_EIG = 10
VALLEY_EDGE = 0.3          # 実験 B で見えた谷の大きさ
SPECTRAL_SEED_BASE = 140_000_000  # 既存の seed(60M〜130M 台)と重ならない


def top_eigvecs(J, k: int = N_EIG) -> tuple[np.ndarray, np.ndarray]:
    """J の大きい方から k 個の固有値と固有ベクトル(列)。符号は第 1 成分が正になるようにそろえる。"""
    vals, vecs = eigsh(J.astype(np.float64), k=k, which="LA")
    order = np.argsort(-vals)
    vals, vecs = vals[order], vecs[:, order]
    vecs = vecs * np.where(vecs[0] < 0, -1.0, 1.0)
    return vals, vecs


def overlap(S: np.ndarray, v: np.ndarray) -> np.ndarray:
    """|<s, sign(v)>| / n。全反転で変わらない。S は (m, n) の ±1。"""
    S = np.where(np.asarray(S) > 0, 1.0, -1.0)
    sv = np.where(np.asarray(v) > 0, 1.0, -1.0)
    return np.abs(S @ sv) / S.shape[1]


def eig_stats(S: np.ndarray, vecs: np.ndarray, d_bks: np.ndarray) -> dict:
    """ov_1, ov_2, E を、全体と谷ごと(d < 0.3 / ≥ 0.3)にまとめる。"""
    ov1, ov2 = overlap(S, vecs[:, 0]), overlap(S, vecs[:, 1])
    e = ov1 - ov2
    near, far = d_bks < VALLEY_EDGE, d_bks >= VALLEY_EDGE
    out = {"n": int(len(S)), "frac_far": float(far.mean()),
           "ov1": describe(ov1), "ov2": describe(ov2), "E": describe(e)}
    for name, m in (("near", near), ("far", far)):
        if m.sum() >= 5:
            out[name] = {"n": int(m.sum()), "ov1_mean": float(ov1[m].mean()), "ov2_mean": float(ov2[m].mean()),
                         "E_mean": float(e[m].mean())}
    if near.sum() >= 5 and far.sum() >= 5:
        u = mannwhitneyu(e[near], e[far], alternative="two-sided")
        out["E_near_vs_far"] = {"auc": float(u.statistic / (near.sum() * far.sum())), "p": float(u.pvalue)}
    return out


def rebuild_matched_sa_starts(ctx) -> np.ndarray:
    """結果②(sa_matched_start v2)の「カット値をそろえた SA の解」を同じ seed で作り直す。"""
    mz = np.load(MATCH_RUN / "raw.npz")
    idx = mz["match_idx"]
    rnd = np.random.default_rng(POOL_SEED_BASE).choice(np.array([-1, 1], np.int8), (mz["pool_cut"].size, ctx.n))[idx]
    cut, signs = run_sa(ctx, rnd, POOL_SEED_BASE + idx.astype(np.int64))
    if not np.array_equal(cut, mz["sa_start_cut"]):
        raise RuntimeError("matched SA starts do not reproduce sa_matched_start v2")
    return signs


def finals_from(ctx, init: np.ndarray, arm: str) -> np.ndarray:
    """結果①(sa_trajectory v3)と同じ seed で SA を回し直し、最終の解を返す(最終カット値の一致を確かめる)。"""
    _, _, seed_base = ARMS[arm]
    seeds = seed_base + SWEEPS * 100_000 + np.arange(len(init), dtype=np.int64)
    cut, signs = run_sa(ctx, init, seeds)
    ref = np.load(TRAJ_RUN / "raw.npz")[f"best_{arm}_S{SWEEPS}"][:, -1]
    if not np.array_equal(cut, ref.astype(np.float64)):
        raise RuntimeError(f"{arm}: final cuts do not reproduce sa_trajectory v3")
    return signs


def part_c1(ctx, refs, vecs, log: Callable[[str], None]) -> tuple[dict, dict]:
    starts = load_starts(ctx)
    t0 = time.perf_counter()
    sets = {"bks": refs, "cim_start": starts["cim"], "random_start": starts["random"]}
    sets["cim_final"] = finals_from(ctx, starts["cim"], "cim_T2")
    sets["random_final"] = finals_from(ctx, starts["random"], "random_T2")
    log(f"C1: reproduced CIM/random SA finals ({time.perf_counter() - t0:.0f}s)")
    sets["sa_solution_start"] = rebuild_matched_sa_starts(ctx)
    log(f"C1: rebuilt matched SA-solution starts ({time.perf_counter() - t0:.0f}s)")
    res, arrays = {}, {}
    for name, S in sets.items():
        d = nearest(S, refs)
        res[name] = eig_stats(S, vecs, d)
        arrays[f"c1_{name}_ov1"] = overlap(S, vecs[:, 0])
        arrays[f"c1_{name}_ov2"] = overlap(S, vecs[:, 1])
        arrays[f"c1_{name}_d_bks"] = d
        r = res[name]
        log(f"C1 {name:18s} ov1 {r['ov1']['mean']:.3f} ov2 {r['ov2']['mean']:.3f} far {r['frac_far']:.1%}"
            + (f"  E near {r['near']['E_mean']:+.3f} far {r['far']['E_mean']:+.3f} AUC {r['E_near_vs_far']['auc']:.2f}"
               if "E_near_vs_far" in r else ""))
    return res, arrays


def part_c2(ctx, refs, vecs, runs: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    bks = float(ctx.bks)
    res, arrays = {}, {}
    for k in range(3):
        s = np.where(vecs[:, k] > 0, 1, -1).astype(np.int8)
        init = np.repeat(s[None, :], runs, axis=0)
        seeds = SPECTRAL_SEED_BASE + k * 1_000_000 + np.arange(runs, dtype=np.int64)
        t0 = time.perf_counter()
        cut, signs = run_sa(ctx, init, seeds)
        d1 = nearest(signs, refs)
        g = gap_pct(cut, bks)
        start_cut = float(ctx.score(s[None, :])[0])
        name = f"sign_u{k + 1}"
        res[name] = {"start_gap": float(gap_pct(np.array([start_cut]), bks)[0]),
                     "start_d_bks": float(nearest(s[None, :], refs)[0]),
                     "final_gap": describe(g), "final_gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
                     "final_d_bks": describe(d1), "final_frac_far": float((d1 >= VALLEY_EDGE).mean())}
        arrays[f"c2_{name}_final_cut"], arrays[f"c2_{name}_d_final"] = cut, d1
        r = res[name]
        log(f"C2 {name}: start gap {r['start_gap']:.2f}% d_bks {r['start_d_bks']:.3f} -> final {g.mean():.4f}% "
            f"± {r['final_gap_ci95']:.4f}, far {r['final_frac_far']:.1%} ({time.perf_counter() - t0:.0f}s)")
    return res, arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=960)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.runs < 16:
        ap.error("runs must be >= 16")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    ctx = load_context("G22")
    refs = load_refs(ctx)
    vals, vecs = top_eigvecs(ctx.J_cim)

    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_", 1)[0][1:]) for p in kind_root.iterdir()
            if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    out_dir = kind_root / (f"v{max(nums, default=0) + 1}_G22_S{SWEEPS}_T2_runs{args.runs}"
                           + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"top eigenvalues of J: {np.round(vals, 4).tolist()}")
    c1, a1 = part_c1(ctx, refs, vecs, log)
    c2, a2 = part_c2(ctx, refs, vecs, args.runs, log)
    tz = np.load(TRAJ_RUN / "raw.npz")
    summary = {
        "meta": {"dataset": "G22", "bks": float(ctx.bks), "sweeps": SWEEPS, "t_start": 2.0, "runs": args.runs,
                 "eigenvalues": vals.tolist(), "valley_edge": VALLEY_EDGE, "spectral_seed_base": SPECTRAL_SEED_BASE,
                 "threads": int(os.environ["NUMBA_NUM_THREADS"])},
        "results": {"c1_location": c1, "c2_spectral_start": c2,
                    "reference_points": {
                        "cim_T2_final_gap_mean": float(gap_pct(tz[f"best_cim_T2_S{SWEEPS}"][:, -1], ctx.bks).mean()),
                        "random_T2_final_gap_mean": float(gap_pct(tz[f"best_random_T2_S{SWEEPS}"][:, -1],
                                                                  ctx.bks).mean())}}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", eigvals=vals, eigvecs=vecs, **a1, **a2)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
