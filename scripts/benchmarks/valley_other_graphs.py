"""valley_other_graphs.py — G22 で見えた「CIM の解は最良解の谷の中にいて、谷は u1 の側にある」が、他の問題でも成り立つか(実験 E)。

11/16 進捗報告会のアクション:「G22 以外の問題でも結果が成り立つかを検証し、可視化する」。
G22 の実験 A〜D と同じ手順を、1 つの問題について 1 回の実行でまとめて行う。

  R. 参照解(既知最良の代わり): ランダム開始の長い SA を回し、最も良いカット値に届いた解を集める。
     G55・G70・K2000 では文献の BKS に SA が届かないため、「この計算で見つかった最良の解」を基準にする。
  P. CIM: 256 本 × P プール(2400 ラウンド)。各プールのカット値上位 16 本を「CIM の解」とし、
     選ぶ前の解の代表として、各プールの seed 順の先頭 16 本も残す(1 本ずつの CIM)。
  S. 同じ本数のランダム解。CIM の解とランダム解から SA(T0 = 2.0、5000 スイープ)。
  B2(検証1). 参照解から距離 d だけ崩した開始解で同じ SA。d ごとの到達点で、谷の大きさを決める。
  C(検証2). J の上位固有ベクトル u1・u2 との一致度を、参照解・CIM の解・SA 後(谷の中/外)で比べる。
     対照として sign(u1)・sign(u2) から同じ SA。

距離は全反転を同一視した正規化ハミング距離 min(h, n − h) / n を、参照解のうち最も近いものまで測る。

  python scripts/benchmarks/valley_other_graphs.py --graph G55 [--pools 100] [--runs-per-d 1600]
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
from scipy.stats import mannwhitneyu  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import load_context, run_cim  # noqa: E402
from scripts.benchmarks.bks_set_distance import canonical, dist_matrix  # noqa: E402
from scripts.benchmarks.eigvec_valley import top_eigvecs  # noqa: E402

EXPERIMENT_KIND = "valley_other_graphs"
GRAPHS = ("G22", "G55", "G70", "K2000")
SWEEPS, T_START, T_END = 5000, 2.0, 0.001     # G22 の実験 A〜D と同じ SA
REF_SWEEPS = 42104                            # G22 で BKS 解を集めた長い SA と同じ
# 参照解を集める長い SA の開始温度:ランダム解から 5000 スイープで一番良かった温度(2026-11-16 の下調べ、各 48 本)
REF_T_START = {"G22": 2.0, "G55": 1.0, "G70": 1.0, "K2000": 50.0}
CIM_ROUNDS, CIM_K, CIM_TOP = 2400, 256, 16    # sa_trajectory v3 と同じ
CIM_SEED_START = 1000
VALLEY_EDGE = 0.3                             # G22 の検証1 で決めた谷の境(他の問題では B2 の結果で確かめる)
D_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
EDGES_REPORTED = (0.1, 0.2, 0.3, 0.4)
# seed(既存の 60M〜500M 台と重ならない)。問題ごとに 100k ずらす。
SEED_BASE = {"ref": 600_000_000, "random_init": 610_000_000, "sa_cim": 620_000_000, "sa_random": 630_000_000,
             "b2": 640_000_000, "spectral": 650_000_000}
GRAPH_OFFSET = {g: i * 10_000_000 // 100 for i, g in enumerate(GRAPHS)}  # 0, 100k, 200k, 300k
PERTURB_SEED = 20261116


def gap_pct(cut: np.ndarray, bks: float) -> np.ndarray:
    return (bks - np.asarray(cut, dtype=np.float64)) / bks * 100


def describe(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(x.size), "mean": float(x.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(x.min()), "max": float(x.max())}


def nearest(sol: np.ndarray, refs: np.ndarray) -> np.ndarray:
    return dist_matrix(sol, refs).min(axis=1).astype(np.float64)


def overlap(S: np.ndarray, v: np.ndarray) -> np.ndarray:
    """|<s, sign(v)>| / n(全反転で変わらない、0〜1)。"""
    S = np.where(np.asarray(S) > 0, 1.0, -1.0)
    return np.abs(S @ np.where(v > 0, 1.0, -1.0)) / S.shape[1]


def seeds_for(ctx, key: str, count: int, start: int = 0) -> np.ndarray:
    return SEED_BASE[key] + GRAPH_OFFSET[ctx.name] + start + np.arange(count, dtype=np.int64)


def run_sa(ctx, init: np.ndarray, seeds: np.ndarray, sweeps: int = SWEEPS,
           t_start: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    cut, signs = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, np.ascontiguousarray(init, dtype=np.int8),
                                  sweeps * ctx.n, t_start=T_START if t_start is None else t_start, t_end=T_END, seeds=seeds)
    signs = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
    cut = np.asarray(cut, dtype=np.float64)
    if not np.array_equal(ctx.score(signs).astype(np.float64), cut):
        raise RuntimeError("rescored cut differs from SA's best cut")
    return cut, signs


def run_batched(ctx, init: np.ndarray, key: str, batch: int, sweeps: int = SWEEPS,
                t_start: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    cuts, signs = [], []
    for s in range(0, len(init), batch):
        c, g = run_sa(ctx, init[s:s + batch], seeds_for(ctx, key, len(init[s:s + batch]), s), sweeps, t_start)
        cuts.append(c)
        signs.append(g)
    return np.concatenate(cuts), np.concatenate(signs)


def part_ref(ctx, runs: int, batch: int, log: Callable[[str], None]) -> tuple[np.ndarray, dict, dict]:
    """長い SA で、最も良いカット値に届いた解(重複なし)を参照解にする。"""
    t0 = time.perf_counter()
    init = np.random.default_rng(SEED_BASE["ref"] + GRAPH_OFFSET[ctx.name]).choice(
        np.array([-1, 1], np.int8), (runs, ctx.n))
    cut, signs = run_batched(ctx, init, "ref", batch, REF_SWEEPS, REF_T_START[ctx.name])
    best = cut.max()
    refs = np.unique(canonical(signs[cut == best]), axis=0)
    res = {"runs": runs, "sweeps": REF_SWEEPS, "t_start": REF_T_START[ctx.name], "best_cut": float(best), "literature_bks": float(ctx.bks),
           "best_gap_pct": float(gap_pct(np.array([best]), ctx.bks)[0]), "hits": int((cut == best).sum()),
           "distinct": int(len(refs)), "final_gap": describe(gap_pct(cut, ctx.bks))}
    if len(refs) > 1:
        dm = dist_matrix(refs, refs)
        res["ref_pairwise_d_max"] = float(dm[np.triu_indices(len(refs), 1)].max())
    log(f"R: {runs} long SA, best {best:.0f} (BKS {ctx.bks}, gap {res['best_gap_pct']:.3f}%), "
        f"{res['hits']} hits, {len(refs)} distinct ({time.perf_counter() - t0:.0f}s)")
    return refs, res, {"ref_final_cut": cut, "refs": refs}


def part_cim(ctx, pools: int, log: Callable[[str], None]) -> dict:
    """CIM を P プール回し、上位 16 本(選んだ解)と seed 順の先頭 16 本(選ぶ前の解)を返す。"""
    t0 = time.perf_counter()
    all_cut = np.empty((pools, CIM_K), np.int32)
    top = np.empty((pools * CIM_TOP, ctx.n), np.int8)
    raw = np.empty((pools * CIM_TOP, ctx.n), np.int8)
    for p in range(pools):
        _, signs = run_cim(ctx, CIM_ROUNDS, CIM_K, CIM_SEED_START + p * CIM_K + np.arange(CIM_K, dtype=np.int64))
        pm = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
        cut = ctx.score(pm).astype(np.int32)
        all_cut[p] = cut
        top[p * CIM_TOP:(p + 1) * CIM_TOP] = pm[np.argsort(-cut, kind="stable")[:CIM_TOP]]
        raw[p * CIM_TOP:(p + 1) * CIM_TOP] = pm[:CIM_TOP]
        if p % 10 == 0 or p == pools - 1:
            log(f"CIM pool {p + 1}/{pools}: {time.perf_counter() - t0:.0f}s mean={cut.mean():.1f} "
                f"top{CIM_TOP}={np.sort(cut)[-CIM_TOP:].mean():.1f}")
    return {"cim_all_cut": all_cut, "cim_top": top, "cim_raw": raw}


def valley_fracs(d: np.ndarray) -> dict:
    return {f"{e:.1f}": float((d < e).mean()) for e in EDGES_REPORTED}


def part_starts(ctx, refs, cim: dict, batch: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    """CIM の解・ランダム解から SA。開始と最後の距離・カット値をまとめる。"""
    t0 = time.perf_counter()
    n_start = len(cim["cim_top"])
    rnd = np.random.default_rng(SEED_BASE["random_init"] + GRAPH_OFFSET[ctx.name]).choice(
        np.array([-1, 1], np.int8), (n_start, ctx.n))
    fin_cim_cut, fin_cim = run_batched(ctx, cim["cim_top"], "sa_cim", batch)
    fin_rnd_cut, fin_rnd = run_batched(ctx, rnd, "sa_random", batch)
    sets = {"cim_start": cim["cim_top"], "cim_raw": cim["cim_raw"], "cim_final": fin_cim, "random_final": fin_rnd}
    res, arrays = {}, {"sa_cim_final_cut": fin_cim_cut, "sa_random_final_cut": fin_rnd_cut,
                       "sa_cim_final": fin_cim, "sa_random_final": fin_rnd, "random_init": rnd}
    for name, S in sets.items():
        d = nearest(S, refs)
        arrays[f"d_{name}"] = d
        res[name] = {"d": describe(d), "in_valley": valley_fracs(d), "gap": describe(gap_pct(ctx.score(S), ctx.bks))}
    for name, cut in (("cim_final", fin_cim_cut), ("random_final", fin_rnd_cut)):
        res[name]["gap"] = describe(gap_pct(cut, ctx.bks))
        res[name]["gap_ci95"] = float(1.96 * gap_pct(cut, ctx.bks).std(ddof=1) / np.sqrt(cut.size))
    res["sa_gap_cim_vs_random_p"] = float(mannwhitneyu(fin_cim_cut, fin_rnd_cut).pvalue)
    log(f"S: SA from {n_start} CIM / random starts ({time.perf_counter() - t0:.0f}s): gap "
        f"{res['cim_final']['gap']['mean']:.4f}% vs {res['random_final']['gap']['mean']:.4f}%; start d median "
        f"CIM {res['cim_start']['d']['median']:.3f} (raw {res['cim_raw']['d']['median']:.3f}), "
        f"final d median CIM {res['cim_final']['d']['median']:.3f} random {res['random_final']['d']['median']:.3f}")
    return res, arrays


def part_b2(ctx, refs, runs: int, batch: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    """参照解から距離 d だけ崩した開始解で SA。到達点が d でどう変わるか。"""
    rng = np.random.default_rng(PERTURB_SEED + GRAPH_OFFSET[ctx.name])
    res, arrays = {}, {}
    for j, d in enumerate(D_GRID):
        t0 = time.perf_counter()
        k = int(round(d * ctx.n))
        init = np.empty((runs, ctx.n), np.int8)
        for i in range(runs):
            base = refs[i % len(refs)].copy()
            idx = rng.choice(ctx.n, size=k, replace=False)
            base[idx] = -base[idx]
            init[i] = base
        cut = np.empty(runs)
        fin = np.empty((runs, ctx.n), np.int8)
        for s in range(0, runs, batch):
            c, g = run_sa(ctx, init[s:s + batch], seeds_for(ctx, "b2", len(init[s:s + batch]), j * 1_000_000 + s))
            cut[s:s + batch], fin[s:s + batch] = c, g
        g = gap_pct(cut, ctx.bks)
        d1 = nearest(fin, refs)
        res[f"{d:.2f}"] = {"flipped": k, "start_gap": describe(gap_pct(ctx.score(init), ctx.bks)),
                           "final_gap": describe(g), "final_gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
                           "final_d": describe(d1)}
        arrays[f"b2_d{j}_final_cut"], arrays[f"b2_d{j}_d_final"] = cut, d1
        log(f"B2 d={d:.2f}: final {g.mean():.4f}% ± {res[f'{d:.2f}']['final_gap_ci95']:.4f}, "
            f"final d median {np.median(d1):.3f} ({time.perf_counter() - t0:.0f}s)")
    return res, arrays


def eig_group(S: np.ndarray, vecs: np.ndarray, d: np.ndarray | None) -> dict:
    ov1, ov2 = overlap(S, vecs[:, 0]), overlap(S, vecs[:, 1])
    out = {"n": int(len(S)), "ov1": float(ov1.mean()), "ov2": float(ov2.mean()), "E": float((ov1 - ov2).mean())}
    if d is not None:
        near, far = d < VALLEY_EDGE, d >= VALLEY_EDGE
        for nm, m in (("near", near), ("far", far)):
            if m.sum() >= 5:
                out[nm] = {"n": int(m.sum()), "ov1": float(ov1[m].mean()), "ov2": float(ov2[m].mean())}
        if near.sum() >= 5 and far.sum() >= 5:
            e = ov1 - ov2
            u = mannwhitneyu(e[near], e[far], alternative="two-sided")
            out["auc_E_near_vs_far"] = float(u.statistic / (near.sum() * far.sum()))
            out["p"] = float(u.pvalue)
    return out


def part_c(ctx, refs, vecs, starts: dict, runs: int, batch: int, log: Callable[[str], None]) -> tuple[dict, dict]:
    res = {"refs": eig_group(refs, vecs, None),
           "cim_start": eig_group(starts["cim_top"], vecs, None),
           "cim_raw": eig_group(starts["cim_raw"], vecs, nearest(starts["cim_raw"], refs)),
           "random_final": eig_group(starts["sa_random_final"], vecs, starts["d_random_final"])}
    arrays = {}
    for k in range(2):
        t0 = time.perf_counter()
        s = np.where(vecs[:, k] > 0, 1, -1).astype(np.int8)
        init = np.repeat(s[None, :], runs, axis=0)
        cut = np.empty(runs)
        fin = np.empty((runs, ctx.n), np.int8)
        for b in range(0, runs, batch):
            c, g = run_sa(ctx, init[b:b + batch], seeds_for(ctx, "spectral", len(init[b:b + batch]),
                                                             k * 1_000_000 + b))
            cut[b:b + batch], fin[b:b + batch] = c, g
        g = gap_pct(cut, ctx.bks)
        d1 = nearest(fin, refs)
        res[f"sign_u{k + 1}_sa"] = {"start_d": float(nearest(s[None, :], refs)[0]), "final_gap": describe(g),
                                    "final_gap_ci95": float(1.96 * g.std(ddof=1) / np.sqrt(g.size)),
                                    "in_valley": valley_fracs(d1)}
        arrays[f"c2_u{k + 1}_final_cut"] = cut
        log(f"C2 sign(u{k + 1}): start d {res[f'sign_u{k + 1}_sa']['start_d']:.3f} -> final {g.mean():.4f}% "
            f"({time.perf_counter() - t0:.0f}s)")
    r = res["random_final"]
    log(f"C1: refs ov1 {res['refs']['ov1']:.3f} ov2 {res['refs']['ov2']:.3f}; CIM ov1 {res['cim_start']['ov1']:.3f} "
        f"ov2 {res['cim_start']['ov2']:.3f}; SA-after near/far AUC {r.get('auc_E_near_vs_far', float('nan')):.2f}")
    return res, arrays


def load_reuse(ctx, run: Path, pools: int, log: Callable[[str], None]) -> tuple[np.ndarray, dict, dict, dict]:
    """前の run の参照解・CIM の解を読み、カット値を採点し直して一致を確かめる。"""
    meta = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    if meta["meta"]["graph"] != ctx.name or meta["meta"]["pools"] != pools:
        raise ValueError(f"{run} is not a {ctx.name} run with {pools} pools")
    z = np.load(run / "raw.npz")
    cim = {k: z[k] for k in ("cim_all_cut", "cim_top", "cim_raw")}
    if ctx.score(z["refs"]).max() != meta["results"]["ref"]["best_cut"]:
        raise RuntimeError("reused refs do not rescore to the recorded best cut")
    if not np.array_equal(np.sort(ctx.score(cim["cim_top"][:CIM_TOP])), np.sort(cim["cim_all_cut"][0])[-CIM_TOP:]):
        raise RuntimeError("reused CIM top solutions do not match the recorded cuts")
    log(f"reused refs ({len(z['refs'])}) and CIM solutions from {run}")
    res = {**meta["results"]["ref"], "reused_from": str(run.resolve().relative_to(ROOT))}
    return z["refs"], res, {"ref_final_cut": z["ref_final_cut"], "refs": z["refs"]}, cim


def make_out_dir(args) -> Path:
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name.split("_", 1)[0][1:]) for p in kind_root.iterdir()
            if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    name = (f"v{max(nums, default=0) + 1}_{args.graph}_P{args.pools}_ref{args.ref_runs}_B{args.runs_per_d}_T{args.t_start:g}".replace(".", "p")
            + (f"_{args.tag}" if args.tag else ""))
    out = kind_root / name
    out.mkdir(parents=True)
    return out


def main() -> None:
    global T_START
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True, choices=GRAPHS)
    ap.add_argument("--pools", type=int, default=100)
    ap.add_argument("--ref-runs", type=int, default=960)
    ap.add_argument("--runs-per-d", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=400)
    ap.add_argument("--t-start", type=float, default=T_START,
                    help="SA の開始温度(G22 は 2.0。疎な問題では 2.0 だと開始解を忘れるので下げる)")
    ap.add_argument("--reuse", default="", help="参照解と CIM の解を、前の run(同じ問題)の raw.npz から使う")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if not 0 < args.t_start <= 100:
        ap.error("t-start must be in (0, 100]")
    T_START = args.t_start
    if min(args.pools, args.ref_runs, args.runs_per_d, args.batch) < 1 or args.runs_per_d < 16:
        ap.error("counts must be positive (runs-per-d >= 16)")
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    ctx = load_context(args.graph)
    out_dir = make_out_dir(args)

    def log(msg: str) -> None:
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    t_all = time.perf_counter()
    vals, vecs = top_eigvecs(ctx.J_cim, k=3)
    log(f"{ctx.name}: n={ctx.n} edges={len(ctx.edges)} BKS={ctx.bks}; top eigenvalues {np.round(vals, 4).tolist()}")
    if args.reuse:
        refs, r_res, r_arr, cim = load_reuse(ctx, Path(args.reuse), args.pools, log)
    else:
        refs, r_res, r_arr = part_ref(ctx, args.ref_runs, args.batch, log)
        cim = part_cim(ctx, args.pools, log)
    s_res, s_arr = part_starts(ctx, refs, cim, args.batch, log)
    b_res, b_arr = part_b2(ctx, refs, args.runs_per_d, args.batch, log)
    c_res, c_arr = part_c(ctx, refs, vecs, {**cim, **s_arr}, args.runs_per_d, args.batch, log)
    summary = {"meta": {"graph": ctx.name, "n": ctx.n, "edges": len(ctx.edges), "literature_bks": float(ctx.bks),
                        "sweeps": SWEEPS, "t_start": T_START, "t_end": T_END, "ref_sweeps": REF_SWEEPS,
                        "cim_rounds": CIM_ROUNDS, "cim_k": CIM_K, "cim_top": CIM_TOP, "pools": args.pools,
                        "valley_edge": VALLEY_EDGE, "eigenvalues": vals.tolist(), "seed_base": SEED_BASE,
                        "graph_offset": GRAPH_OFFSET[ctx.name], "threads": int(os.environ["NUMBA_NUM_THREADS"]),
                        "elapsed_s": round(time.perf_counter() - t_all, 1)},
               "results": {"ref": r_res, "starts": s_res, "b2_perturbed": b_res, "c_eigvec": c_res}}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(out_dir / "raw.npz", eigvals=vals, eigvecs=vecs, **r_arr, **cim, **s_arr, **b_arr, **c_arr)
    log(f"saved: {out_dir} ({time.perf_counter() - t_all:.0f}s)")


if __name__ == "__main__":
    main()
