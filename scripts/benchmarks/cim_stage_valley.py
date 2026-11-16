"""cim_stage_valley.py — 実験 D:CIM の線形段が谷を選び、非線形段が谷の奥へ運ぶか(G22)。

事前登録: docs/20261114/prereg_cim_stage_v1.md(予測 P1〜P3 と定義はそちらが正)。

D1(観察):CIM の途中の振幅 c を記録し、各時点の sign(c) について
    飽和度 s = γ·mean((√η c + Jc)²)、J の上位固有ベクトル u1, u2 との重なり、最も近い BKS 解までの距離 d を測る。
    線形段の終わり k_lin = s が初めて 0.01 以上になったラウンド。
D2(介入):sa_trajectory v3 の CIM 開始解と同じ 960 本を同じ seed で回し、k_lin で止めた sign(c) から
    同じ SA(T0 = 2.0、5000 スイープ)を回す。最後まで回した CIM の解からの SA(cim_T2)と比べる。

usage(プロジェクトルートから):
    NUMBA_NUM_THREADS=24 .venv/Scripts/python.exe -u scripts/benchmarks/cim_stage_valley.py [--d1-pools 4] [--d2-pools 60]
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
from scipy.stats import wilcoxon  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.CIM import _simulate_cim_trajectory  # noqa: E402
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim  # noqa: E402
from scripts.benchmarks.eigvec_valley import overlap, top_eigvecs  # noqa: E402
from scripts.benchmarks.sa_path_bks_distance import load_refs  # noqa: E402
from scripts.benchmarks.start_distance_causal import SWEEPS, TRAJ_RUN, describe, gap_pct, nearest, run_sa  # noqa: E402

EXPERIMENT_KIND = "cim_stage_valley"
CIM_ROUNDS = 2400            # sa_trajectory v3 と同じ
CIM_SEED_START = 1000        # 同上
POOL_SIZE, TOP_M = 256, 16   # 同上
SAT_LINEAR_END = 0.01        # 事前登録:飽和度がこれを超えたら線形段の終わり
EARLY_END = 600              # ここまでは 2 ラウンドおきに記録
STAGE_SEED_BASE = 150_000_000  # 既存の seed(60M〜140M 台)と重ならない
VALLEY_EDGE = 0.3


def sample_rounds(total: int = CIM_ROUNDS) -> np.ndarray:
    early = np.arange(0, min(EARLY_END, total), 2)
    late = np.arange(EARLY_END, total, 20)
    return np.unique(np.concatenate([early, late, [total - 1]])).astype(np.int64)


class CimRunner:
    """同じ seed の CIM を、途中の振幅を記録しながら 1 本ずつ回す(run_cim と同じ乱数列)。"""

    def __init__(self, ctx) -> None:
        p = PARAMS[ctx.name]["CIM"]
        e = np.asarray(ctx.edges, dtype=np.int64)
        self.ctx, self.p = ctx, p
        self.ea, self.eb = np.ascontiguousarray(e[:, 0]), np.ascontiguousarray(e[:, 1])
        self.ew = np.ones(len(e), dtype=np.float64)

    def run(self, seed: int, rounds: np.ndarray, total: int = CIM_ROUNDS):
        p, J = self.p, self.ctx.J_cim
        return _simulate_cim_trajectory(self.ctx.n, int(total), J.data, J.indices, J.indptr, self.ea, self.eb,
                                        self.ew, p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
                                        p["photon_energy"], p["dP_per_round"], np.int64(seed), rounds)

    def saturation(self, c_hist: np.ndarray) -> np.ndarray:
        cin = np.sqrt(self.p["eta"]) * c_hist + (self.ctx.J_cim @ c_hist.T).T
        return self.p["gamma"] * np.mean(cin ** 2, axis=1)


def linear_end_index(sat: np.ndarray, threshold: float = SAT_LINEAR_END) -> int:
    """飽和度が初めて threshold 以上になった記録点の番号。超えなければ最後の点。"""
    hit = np.flatnonzero(sat >= threshold)
    return int(hit[0]) if hit.size else len(sat) - 1


def signs_of(c: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(c) > 0, 1, -1).astype(np.int8)


def trace_stats(c_hist: np.ndarray, sat: np.ndarray, vecs: np.ndarray, refs: np.ndarray) -> dict[str, np.ndarray]:
    s = signs_of(c_hist)
    norm2 = np.maximum((c_hist ** 2).sum(axis=1), 1e-300)
    proj = (c_hist @ vecs[:, :3]) ** 2 / norm2[:, None]
    return {"sat": sat, "rms": np.sqrt(np.mean(c_hist ** 2, axis=1)), "ov1": overlap(s, vecs[:, 0]),
            "ov2": overlap(s, vecs[:, 1]), "d_bks": nearest(s, refs), "w1": proj[:, 0], "w2": proj[:, 1],
            "w3": proj[:, 2], "w_top10": ((c_hist @ vecs) ** 2).sum(axis=1) / norm2}


def part_d1(ctx, runner: CimRunner, vecs, refs, pools: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    rounds = sample_rounds()
    keys = ("sat", "rms", "ov1", "ov2", "d_bks", "w1", "w2", "w3", "w_top10")
    tr = {k: np.empty((pools * POOL_SIZE, len(rounds)), np.float32) for k in keys}
    k_lin, best_d, best_e = (np.empty(pools * POOL_SIZE) for _ in range(3))
    t0 = time.perf_counter()
    for p in range(pools):
        seeds = CIM_SEED_START + p * POOL_SIZE + np.arange(POOL_SIZE, dtype=np.int64)
        ref_cut, _ = run_cim(ctx, CIM_ROUNDS, POOL_SIZE, seeds)
        for j, seed in enumerate(seeds):
            c_hist, _, best_cut, best_signs = runner.run(int(seed), rounds)
            if best_cut != ref_cut[j]:
                raise RuntimeError(f"seed {seed}: trajectory kernel does not reproduce run_cim")
            i = p * POOL_SIZE + j
            st = trace_stats(c_hist, runner.saturation(c_hist), vecs, refs)
            for k in keys:
                tr[k][i] = st[k]
            k_lin[i] = rounds[linear_end_index(st["sat"])]
            bs = signs_of(np.where(best_signs, 1, -1))[None, :]
            best_d[i] = nearest(bs, refs)[0]
            best_e[i] = overlap(bs, vecs[:, 0])[0] - overlap(bs, vecs[:, 1])[0]
        log(f"D1 pool {p + 1}/{pools} ({time.perf_counter() - t0:.0f}s)")
    return summarize_d1(rounds, tr, k_lin, best_d, best_e), {**{f"d1_{k}": v for k, v in tr.items()},
                                                            "d1_rounds": rounds, "d1_k_lin": k_lin,
                                                            "d1_best_d": best_d, "d1_best_E": best_e}


def summarize_d1(rounds, tr, k_lin, best_d, best_e) -> dict:
    idx = np.searchsorted(rounds, k_lin)
    rows = np.arange(len(k_lin))
    at_lin = {k: tr[k][rows, idx].astype(np.float64) for k in tr}
    e_lin = at_lin["ov1"] - at_lin["ov2"]
    e_fin = (tr["ov1"][:, -1] - tr["ov2"][:, -1]).astype(np.float64)
    d_fin = tr["d_bks"][:, -1].astype(np.float64)
    path = []
    for r in (0, 100, 150, 200, 220, 240, 260, 280, 300, 360, 480, 600, 1200, 2399):
        j = int(np.argmin(np.abs(rounds - r)))
        e = tr["ov1"][:, j] - tr["ov2"][:, j]
        path.append({"round": int(rounds[j]), "sat_median": float(np.median(tr["sat"][:, j])),
                     "d_median": float(np.median(tr["d_bks"][:, j])), "E_median": float(np.median(e)),
                     "frac_far": float((tr["d_bks"][:, j] >= VALLEY_EDGE).mean()),
                     "w1_median": float(np.median(tr["w1"][:, j])), "w_top10_median": float(np.median(tr["w_top10"][:, j]))})
    return {"n": int(len(k_lin)), "k_lin": describe(k_lin), "path": path,
            "at_linear_end": {"d": describe(at_lin["d_bks"]), "E": describe(e_lin), "ov1": describe(at_lin["ov1"]),
                              "ov2": describe(at_lin["ov2"]), "frac_far": float((at_lin["d_bks"] >= VALLEY_EDGE).mean()),
                              "w_top10": describe(at_lin["w_top10"])},
            "final_round": {"d": describe(d_fin), "E": describe(e_fin), "frac_far": float((d_fin >= VALLEY_EDGE).mean())},
            "best_signs": {"d": describe(best_d), "E": describe(best_e), "frac_far": float((best_d >= VALLEY_EDGE).mean())},
            "d_drop_linear_to_final_median": float(np.median(at_lin["d_bks"]) - np.median(d_fin))}


def selected_seeds(ctx, pools: int, log: Callable[[str], None]) -> tuple[np.ndarray, np.ndarray]:
    """sa_trajectory v3 と同じ選び方(各プールのカット値上位 16 個)で seed と解を作り直し、一致を確かめる。"""
    top = np.load(ROOT / TRAJ_RUN / "raw.npz")["top_signs"]
    seeds_out, signs_out = [], []
    for p in range(pools):
        seeds = CIM_SEED_START + p * POOL_SIZE + np.arange(POOL_SIZE, dtype=np.int64)
        _, signs = run_cim(ctx, CIM_ROUNDS, POOL_SIZE, seeds)
        pm = signs_of(np.where(np.asarray(signs) > 0, 1, -1))
        order = np.argsort(-ctx.score(pm).astype(np.int32), kind="stable")[:TOP_M]
        if not np.array_equal(pm[order], top[p]):
            raise RuntimeError(f"pool {p}: top-{TOP_M} CIM solutions do not reproduce sa_trajectory v3")
        seeds_out.append(seeds[order])
        signs_out.append(pm[order])
    log(f"D2: reproduced the {pools * TOP_M} CIM starts of sa_trajectory v3")
    return np.concatenate(seeds_out), np.concatenate(signs_out)


def part_d2(ctx, runner: CimRunner, vecs, refs, pools: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    seeds, full = selected_seeds(ctx, pools, log)
    rounds = np.arange(0, EARLY_END, 2, dtype=np.int64)
    lin = np.empty_like(full)
    k_lin = np.empty(len(seeds))
    for i, seed in enumerate(seeds):
        c_hist, *_ = runner.run(int(seed), rounds, total=EARLY_END)
        j = linear_end_index(runner.saturation(c_hist))
        lin[i], k_lin[i] = signs_of(c_hist[j]), rounds[j]
    t0 = time.perf_counter()
    cut, final = run_sa(ctx, lin, STAGE_SEED_BASE + np.arange(len(lin), dtype=np.int64))
    g_lin = gap_pct(cut, ctx.bks)
    ref = np.load(ROOT / TRAJ_RUN / "raw.npz")[f"best_cim_T2_S{SWEEPS}"][: len(lin), -1]
    g_full = gap_pct(ref, ctx.bks)
    w = wilcoxon(g_lin, g_full, alternative="two-sided")
    d_lin, d_full, d_after = nearest(lin, refs), nearest(full, refs), nearest(final, refs)
    e_lin = overlap(lin, vecs[:, 0]) - overlap(lin, vecs[:, 1])
    res = {"n": int(len(lin)), "k_lin": describe(k_lin),
           "start_gap": {"linear_stop": describe(gap_pct(ctx.score(lin), ctx.bks)),
                         "full_cim": describe(gap_pct(ctx.score(full), ctx.bks))},
           "start_d": {"linear_stop": describe(d_lin), "full_cim": describe(d_full)},
           "start_frac_far": {"linear_stop": float((d_lin >= VALLEY_EDGE).mean()),
                              "full_cim": float((d_full >= VALLEY_EDGE).mean())},
           "start_E_linear_stop": describe(e_lin),
           "final_gap": {"linear_stop": describe(g_lin), "full_cim": describe(g_full)},
           "final_gap_ci95": {"linear_stop": float(1.96 * g_lin.std(ddof=1) / np.sqrt(g_lin.size)),
                              "full_cim": float(1.96 * g_full.std(ddof=1) / np.sqrt(g_full.size))},
           "final_frac_far_linear_stop": float((d_after >= VALLEY_EDGE).mean()),
           "wilcoxon_p": float(w.pvalue), "mean_diff_pct": float(g_lin.mean() - g_full.mean())}
    log(f"D2: linear-stop start d {np.median(d_lin):.3f} (full {np.median(d_full):.3f}) -> SA {g_lin.mean():.4f}% "
        f"vs full {g_full.mean():.4f}% (p={w.pvalue:.2g}, {time.perf_counter() - t0:.0f}s)")
    return res, {"d2_seeds": seeds, "d2_k_lin": k_lin, "d2_gap_linear_stop": g_lin, "d2_gap_full": g_full,
                 "d2_d_linear_stop": d_lin, "d2_d_full": d_full, "d2_d_after": d_after, "d2_E_linear_stop": e_lin}


def make_out_dir(args) -> Path:
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_", 1)[0][1:]) for p in kind_root.iterdir()
            if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    name = f"v{max(nums, default=0) + 1}_G22_D1P{args.d1_pools}_D2P{args.d2_pools}_sat{SAT_LINEAR_END}"
    out = kind_root / (name + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d1-pools", type=int, default=4)
    ap.add_argument("--d2-pools", type=int, default=60)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not (1 <= args.d1_pools <= 100 and 1 <= args.d2_pools <= 100):
        ap.error("pools must be in 1..100")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")
    ctx = load_context("G22")
    refs, (vals, vecs) = load_refs(ctx), top_eigvecs(ctx.J_cim)
    runner = CimRunner(ctx)
    out_dir = make_out_dir(args)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    d1, a1 = part_d1(ctx, runner, vecs, refs, args.d1_pools, log)
    log(f"D1: k_lin median {d1['k_lin']['median']:.0f}; at k_lin d {d1['at_linear_end']['d']['median']:.3f} "
        f"E {d1['at_linear_end']['E']['median']:+.3f}; final d {d1['final_round']['d']['median']:.3f} "
        f"E {d1['final_round']['E']['median']:+.3f}")
    d2, a2 = part_d2(ctx, runner, vecs, refs, args.d2_pools, log)
    summary = {"meta": {"dataset": "G22", "bks": float(ctx.bks), "cim_rounds": CIM_ROUNDS, "sweeps": SWEEPS,
                        "t_start": 2.0, "sat_linear_end": SAT_LINEAR_END, "d1_pools": args.d1_pools,
                        "d2_pools": args.d2_pools, "stage_seed_base": STAGE_SEED_BASE,
                        "prereg": "docs/20261114/prereg_cim_stage_v1.md", "eigenvalues": vals.tolist(),
                        "threads": int(os.environ["NUMBA_NUM_THREADS"])},
               "results": {"d1_trace": d1, "d2_linear_stop": d2}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", **a1, **a2)
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
