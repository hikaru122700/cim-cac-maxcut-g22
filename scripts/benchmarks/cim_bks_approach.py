"""cim_bks_approach.py — CIM は G22 の最良解(BKS 解)にどう近づいていくか(実験 F)。

11/16 進捗報告会のアクション:「G22 の最適パターンに、CIM がどう近づくかを確認する」。
実験 D(cim_stage_valley v1)の D1(1024 本 = 256 本 × 4 プール、途中の BKS 解までの距離を記録済み)を使う。

  F1. 各プールでカット値上位 16 本(選ばれる解)と残り 240 本に分け、距離 d のラウンドごとの変化を比べる。
      (実験 D では全 1024 本の中央値しか見ておらず、選ばれる解の動きは見ていない)
  F2. 選ばれた解 1 本と、選ばれなかった解 1 本(カット値が中央の解)について、途中の振幅 c を全スピン記録し、
      各スピンが最良解と同じ向きになるラウンドを見る。

  python scripts/benchmarks/cim_bks_approach.py
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "24")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.algo_registry import load_context, run_cim  # noqa: E402
from scripts.benchmarks.bks_set_distance import dist_matrix  # noqa: E402
from scripts.benchmarks.cim_stage_valley import (CIM_SEED_START, POOL_SIZE, TOP_M, CimRunner,  # noqa: E402
                                                 VALLEY_EDGE, sample_rounds, signs_of)
from scripts.benchmarks.sa_path_bks_distance import load_refs  # noqa: E402

EXPERIMENT_KIND = "cim_bks_approach"
STAGE_RUN = ROOT / "results/2026-11-15/cim_stage_valley/v1_G22_D1P4_D2P100_sat0.01"
CHECK_ROUNDS = (0, 100, 200, 236, 300, 400, 600, 1200, 2399)


def group_path(d: np.ndarray, rounds: np.ndarray, sel: np.ndarray) -> list[dict]:
    rows = []
    for r in CHECK_ROUNDS:
        j = int(np.argmin(np.abs(rounds - r)))
        row = {"round": int(rounds[j])}
        for nm, m in (("selected", sel), ("others", ~sel)):
            row[nm] = {"d_median": float(np.median(d[m, j])), "d_q1": float(np.percentile(d[m, j], 25)),
                       "d_q3": float(np.percentile(d[m, j], 75)), "in_valley": float((d[m, j] < VALLEY_EDGE).mean())}
        rows.append(row)
    return rows


def spin_trace(ctx, runner: CimRunner, seed: int, rounds: np.ndarray, refs: np.ndarray) -> dict:
    """1 本の CIM の振幅 c を全スピン記録し、最後の解に一番近い BKS 解 b の向き(全反転をそろえる)で c·b を返す。"""
    c_hist, _, best_cut, best_signs = runner.run(int(seed), rounds)
    final = signs_of(np.where(best_signs, 1, -1))
    b = refs[int(np.argmin(dist_matrix(final[None, :], refs)[0]))].astype(np.float64)
    if final.astype(np.float64) @ b < 0:
        b = -b
    aligned = c_hist * b[None, :]
    agree = aligned > 0
    # 各スピンが「最後まで BKS 解と同じ向き」になったラウンド(最後に向きが変わった記録点の次)
    settle = np.full(ctx.n, rounds[-1], dtype=np.int64)
    for i in range(ctx.n):
        bad = np.flatnonzero(~agree[:, i])
        settle[i] = rounds[0] if bad.size == 0 else (rounds[bad[-1] + 1] if bad[-1] + 1 < len(rounds) else -1)
    return {"seed": int(seed), "cut": float(best_cut), "aligned": aligned.astype(np.float32), "settle": settle,
            "frac_agree": agree.mean(axis=1)}


def main() -> None:
    ctx = load_context("G22")
    refs = load_refs(ctx)
    z = np.load(STAGE_RUN / "raw.npz")
    rounds, d = z["d1_rounds"], z["d1_d_bks"].astype(np.float64)
    pools = d.shape[0] // POOL_SIZE
    t0 = time.perf_counter()
    cuts = np.empty(d.shape[0])
    for p in range(pools):
        seeds = CIM_SEED_START + p * POOL_SIZE + np.arange(POOL_SIZE, dtype=np.int64)
        _, signs = run_cim(ctx, 2400, POOL_SIZE, seeds)
        cuts[p * POOL_SIZE:(p + 1) * POOL_SIZE] = ctx.score(signs_of(np.where(np.asarray(signs) > 0, 1, -1)))
    sel = np.zeros(d.shape[0], bool)
    for p in range(pools):
        block = cuts[p * POOL_SIZE:(p + 1) * POOL_SIZE]
        sel[p * POOL_SIZE + np.argsort(-block, kind="stable")[:TOP_M]] = True
    final_d = d[:, -1]
    print(f"F1: cuts recomputed ({time.perf_counter() - t0:.0f}s); final d median selected "
          f"{np.median(final_d[sel]):.3f} vs others {np.median(final_d[~sel]):.3f}", flush=True)

    runner = CimRunner(ctx)
    best_i = int(np.argmax(np.where(sel, cuts, -np.inf)))
    others = np.flatnonzero(~sel)
    mid_i = int(others[np.argsort(cuts[others], kind="stable")[len(others) // 2]])
    traces = {}
    for nm, i in (("selected", best_i), ("other", mid_i)):
        seed = CIM_SEED_START + i
        traces[nm] = spin_trace(ctx, runner, seed, rounds, refs)
        if traces[nm]["cut"] != cuts[i]:
            raise RuntimeError(f"{nm}: trajectory cut {traces[nm]['cut']} != run_cim {cuts[i]}")
        print(f"F2 {nm}: seed {seed} cut {cuts[i]:.0f} final d {final_d[i]:.3f}", flush=True)

    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_", 1)[0][1:]) for p in kind_root.iterdir()
            if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    out = kind_root / f"v{max(nums, default=0) + 1}_G22_D1P{pools}"
    out.mkdir(parents=True)
    summary = {"meta": {"dataset": "G22", "source": str(STAGE_RUN.relative_to(ROOT)), "runs": int(d.shape[0]),
                        "pools": pools, "top_m": TOP_M, "valley_edge": VALLEY_EDGE,
                        "k_lin_median": float(np.median(z["d1_k_lin"]))},
               "results": {"path": group_path(d, rounds, sel),
                           "final": {nm: {"d_median": float(np.median(final_d[m])),
                                          "in_valley": float((final_d[m] < VALLEY_EDGE).mean()),
                                          "cut_mean": float(cuts[m].mean())}
                                     for nm, m in (("selected", sel), ("others", ~sel))},
                           "traces": {nm: {"seed": t["seed"], "cut": t["cut"],
                                           "unsettled_spins": int((t["settle"] < 0).sum()),
                                           "settle_median": float(np.median(t["settle"][t["settle"] >= 0]))}
                                      for nm, t in traces.items()}}}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out / "raw.npz", rounds=rounds, d=d.astype(np.float32), cuts=cuts, selected=sel,
                        k_lin=z["d1_k_lin"], **{f"{nm}_{k}": v for nm, t in traces.items()
                                                for k, v in t.items() if k not in ("seed", "cut")})
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
