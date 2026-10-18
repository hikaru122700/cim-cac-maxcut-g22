"""ステップ 2:小規模問題で、CIM 解の後段 SA での価値と、厳密量・安価な指標を測る。

各問題(scripts/benchmarks/small_landscape.py の 30 問題)で:
  1. 2^(n-1) 状態を全列挙し、最適値を得る。
  2. CIM を K 回走らせ、異なる非最適解を候補とする(出現回数も記録)。
     小規模問題は 1500 ラウンドの CIM でほぼ解けてしまう(最適率 0.95 以上)ので、
     既定は増幅の途中の 512 ラウンド(最適率 0.1〜0.9、2026-10-18 の予備測定)。
  3. 候補ごとに厳密量(障壁・経路長・最適解までの距離)と安価な指標を計算する。
  4. 候補ごとに warm SA を R 回かけ、最適到達率をラベルとする(3 つの予算)。

開発用・検証用の区別は解析側(small_seed_analysis.py)で行う。ここでは全問題を同じ手順で測る。

使い方(プロジェクトルートから):
  python scripts/benchmarks/small_seed_study.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.CIM import build_coupling_matrix, simulate_cim_batch  # noqa: E402
from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.seed_value_pilot import (  # noqa: E402
    _greedy_local_opt,
    candidate_features,
    eig_features,
    low_eigvecs,
)
from scripts.benchmarks.small_landscape import (  # noqa: E402
    all_problems,
    csr,
    enumerate_cuts,
    exact_landscape,
    ids_to_states,
    state_ids,
)

EXPERIMENT_KIND = "small_seed_study"
# 9/28 の小規模 CIM 設定(modules/2026-09-28_AntiferroSeedStudy.py)と同じ
CIM_PARAMS = dict(kappa=130.0, L=0.05, gamma=42.09, eta=10 ** (-11 / 10),
                  bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3)
CIM_COUPLING = -0.03
SA_T = (0.5, 0.01)
LABEL_SEED_OFFSET = 5_000_000
TRIAL_SEED_OFFSET = 9_000_000
TRIAL_RUNS = 8
TRIAL_SWEEPS = 1


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
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


def adjacency(n, indptr, indices, data) -> csr_matrix:
    rows = np.repeat(np.arange(n), np.diff(indptr))
    return csr_matrix((data, (rows, indices)), shape=(n, n))


def greedy_ls(n, indptr, indices, data, signs_pm):
    out = signs_pm.astype(np.float64).copy()
    for k in range(len(out)):
        _greedy_local_opt(n, indptr, indices, data, out[k])
    return np.where(out > 0, 1, -1).astype(np.int8)


def unsatisfied_features(n, edges, weights, signs01):
    """満たされていない辺(w>0 で同じ側、w<0 で反対側)の数と、その連結成分。"""
    e = np.asarray(edges)
    w = np.asarray(weights)
    out = {"n_unsat": [], "unsat_components": [], "unsat_largest": []}
    for s in signs01:
        cut = s[e[:, 0]] != s[e[:, 1]]
        bad = np.where(w > 0, ~cut, cut)
        out["n_unsat"].append(bad.sum())
        if not bad.any():
            out["unsat_components"].append(0)
            out["unsat_largest"].append(0)
            continue
        a, b = e[bad, 0], e[bad, 1]
        g = csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        _, lab = connected_components(g, directed=False)
        touched = np.unique(np.concatenate([a, b]))
        counts = np.bincount(lab[touched])
        out["unsat_components"].append(int((counts > 0).sum()))
        out["unsat_largest"].append(int(counts.max()))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def run_problem(prob, args, log):
    t0 = time.perf_counter()
    n, edges, weights = prob["n"], prob["edges"], prob["weights"]
    ip, ix, dt = csr(n, edges, weights)
    cuts_all = enumerate_cuts(n, ip, ix, dt)
    opt = int(cuts_all.max())

    J = build_coupling_matrix(n, edges, CIM_COUPLING, weights=weights)
    seeds = np.arange(args.candidates, dtype=np.int64)
    _, cim_signs = simulate_cim_batch(n, J, edges, args.cim_rounds, args.candidates,
                                      **CIM_PARAMS, seeds=seeds, weights=weights)
    ids = state_ids((np.asarray(cim_signs) > 0).astype(np.int8))
    cim_cut = cuts_all[ids]
    uniq, counts = np.unique(ids, return_counts=True)
    keep = cuts_all[uniq] < opt
    cand, cand_count = uniq[keep], counts[keep]
    states01 = ids_to_states(cand, n)
    signs_pm = np.where(states01 > 0, 1, -1).astype(np.int8)
    init_cut = cuts_all[cand].astype(float)
    if len(cand) == 0:
        log(f"{prob['name']}: no non-optimal CIM candidates")
        return None

    land = exact_landscape(n, edges, weights, cand)
    A = adjacency(n, ip, ix, dt)
    ls = greedy_ls(n, ip, ix, dt, signs_pm)
    ls_cut = cuts_all[state_ids((ls > 0).astype(np.int8))].astype(float)
    feats = candidate_features(A, signs_pm)
    for k, v in candidate_features(A, ls).items():
        if k in ("n_zero_gain", "largest_zero_cluster", "n_gain_minus2_to_0"):
            feats[f"ls_{k}"] = v
    _, vecs = low_eigvecs(A)
    feats.update(eig_features(signs_pm, vecs))
    feats.update(unsatisfied_features(n, edges, weights, states01))
    feats["ls_gain"] = ls_cut - init_cut
    feats["cim_count"] = cand_count.astype(float)

    # 試し打ち:1 掃引の SA を 8 回
    tr_init = np.repeat(signs_pm, TRIAL_RUNS, axis=0)
    tr_seeds = TRIAL_SEED_OFFSET + np.arange(len(tr_init), dtype=np.int64)
    tr_cuts, _ = simulate_sa_warm(n, edges, weights, tr_init, TRIAL_SWEEPS * n,
                                  t_start=SA_T[0], t_end=SA_T[1], seeds=tr_seeds)
    tr_cuts = np.asarray(tr_cuts, dtype=float).reshape(len(cand), TRIAL_RUNS)
    feats["trial_sa_mean"] = tr_cuts.mean(axis=1)
    feats["trial_sa_success"] = (tr_cuts >= opt).mean(axis=1)

    R = args.repeats
    rep = np.repeat(signs_pm, R, axis=0)
    lab_seeds = LABEL_SEED_OFFSET + np.arange(len(rep), dtype=np.int64)
    labels = {}
    for sw in args.sweeps:
        c, s = simulate_sa_warm(n, edges, weights, rep, sw * n,
                                t_start=SA_T[0], t_end=SA_T[1], seeds=lab_seeds)
        c = np.asarray(c, dtype=float)
        chk = cuts_all[state_ids((np.asarray(s) > 0).astype(np.int8))]
        assert np.array_equal(chk, c), "SA の報告 cut が全列挙と一致しない"
        c = c.reshape(len(cand), R)
        labels[f"success_S{sw}"] = (c >= opt).mean(axis=1)
        labels[f"meancut_S{sw}"] = c.mean(axis=1)

    rec = {
        "name": prob["name"], "family": prob["family"], "split": prob["split"], "n": n,
        "m": len(edges), "opt": opt, "n_opt": land["n_opt"],
        "cim_opt_rate": float((cim_cut == opt).mean()),
        "cim_mean_gap": float((opt - cim_cut).mean()),
        "n_candidates": int(len(cand)),
        "n_unique_cim": int(len(uniq)),
        "seconds": time.perf_counter() - t0,
    }
    arrays = {
        "cand_ids": cand, "init_cut": init_cut, "ls_cut": ls_cut,
        "barrier": land["barrier"], "path_length": land["path_length"],
        "opt_distance": land["opt_distance"],
        **{f"feat_{k}": np.asarray(v, dtype=float) for k, v in feats.items()},
        **labels,
    }
    log(
        f"{prob['name']:>10} [{prob['split']}] n={n} m={len(edges)} opt={opt} n_opt={land['n_opt']} "
        f"CIM opt rate={rec['cim_opt_rate']:.2f} candidates={len(cand)} "
        f"barrier>0: {(land['barrier'] > 0).mean():.2f} "
        f"success_S10 mean={labels.get('success_S10', np.array([np.nan])).mean():.3f} "
        f"({rec['seconds']:.1f}s)"
    )
    return rec, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=256)
    ap.add_argument("--cim-rounds", type=int, default=512)
    ap.add_argument("--sweeps", type=int, nargs="+", default=[3, 10, 30])
    ap.add_argument("--only", nargs="*", default=None, help="問題名を絞る(試験用)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    kind_root = get_kind_root()
    desc = f"K{args.candidates}_R{args.repeats}_cim{args.cim_rounds}" + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)

    def log(msg):
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    problems = all_problems()
    if args.only:
        problems = [p for p in problems if p["name"] in args.only]
    records = []
    for prob in problems:
        res = run_problem(prob, args, log)
        if res is None:
            continue
        rec, arrays = res
        records.append(rec)
        np.savez_compressed(out_dir / f"{prob['name']}.npz", **arrays)
    meta = {
        "candidates": args.candidates, "repeats": args.repeats, "cim_rounds": args.cim_rounds,
        "sweeps": args.sweeps, "cim_params": CIM_PARAMS, "cim_coupling": CIM_COUPLING,
        "sa_t": SA_T, "label_seed_offset": LABEL_SEED_OFFSET, "trial": [TRIAL_RUNS, TRIAL_SWEEPS],
        "problems": {p["name"]: p for p in problems},
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "records": records}, f, ensure_ascii=False, indent=2)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
