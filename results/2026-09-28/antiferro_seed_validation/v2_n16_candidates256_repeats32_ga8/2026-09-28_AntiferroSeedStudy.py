"""小規模反強磁性問題で、CIM出力のSA/GA初期解としての性質を実測する。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from numba import set_num_threads

from modules.CIM import build_coupling_matrix, simulate_cim_batch
from modules.GA import simulate_ga_batch, tabu_refine_batch, _grouping_crossover
from modules.SA import simulate_sa_warm
from modules.verify import compute_cut_from_edges

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_KIND = "antiferro_seed_validation"
CIM_PARAMS = dict(kappa=130.0, L=0.05, gamma=42.09, eta=10 ** (-11/10),
                  bandwidth=1e9, photon_energy=1.28e-19, dP_per_round=0.05e-3)
CHECKPOINTS = [32, 128, 1500]
SWEEPS = [1, 10, 100]
N = 16


def graph_edges(diagonal: bool) -> list[tuple[int, int]]:
    """開放4×4格子。対角ありでは9個の正方形に各1本を追加する。"""
    edges = [(4*r+c, 4*r+c+1) for r in range(4) for c in range(3)]
    edges += [(4*r+c, 4*(r+1)+c) for r in range(3) for c in range(4)]
    if diagonal:
        edges += [(4*r+c, 4*(r+1)+c+1) for r in range(3) for c in range(3)]
    assert len(edges) == len(set(edges)) and all(i != j for i, j in edges)
    return edges


def recut(signs, edges):
    """辺リストから全候補のcutを再計算する。"""
    x = np.asarray(signs)
    return sum((x[..., i] != x[..., j]).astype(np.int16) for i, j in edges)


def check_cuts(signs, cuts, edges, bks):
    """全出力を再集計し、独立検算モジュールとも一部照合する。"""
    assert np.array_equal(recut(signs, edges), cuts)
    assert np.all(cuts <= bks)
    flat_s = np.asarray(signs).reshape(-1, N)
    flat_c = np.asarray(cuts).reshape(-1)
    for i in np.linspace(0, len(flat_s)-1, min(8, len(flat_s)), dtype=int):
        assert compute_cut_from_edges(flat_s[i].tolist(), edges) == flat_c[i]


def exact_spectrum(edges):
    """2^16配置を列挙し、隣接行列による別計算とも全配置を照合する。"""
    states = ((np.arange(2**N, dtype=np.uint32)[:, None] >> np.arange(N)) & 1).astype(np.int8)
    values = recut(states, edges)
    adjacency = np.zeros((N, N), dtype=np.int16)
    for i, j in edges:
        adjacency[i, j] = adjacency[j, i] = 1
    spin = (2 * states - 1).astype(np.int16)
    independent = (2*len(edges) - np.sum((spin @ adjacency) * spin, axis=1)) // 4
    assert np.array_equal(values, independent)
    return values, int(values.max()), int(np.sum(values == values.max()))


def gains(signs, edges):
    """符号配置ごとの全単独反転利得を返す。"""
    x = np.where(np.asarray(signs) > 0, 1, -1)
    out = np.zeros(x.shape, dtype=np.int16)
    for i, j in edges:
        z = x[..., i] * x[..., j]
        out[..., i] += z
        out[..., j] += z
    return out


def stats(cuts, bks):
    x = np.asarray(cuts)
    return dict(mean=float(x.mean()), std=float(x.std()), best=float(x.max()),
                worst=float(x.min()), success=float(np.mean(x == bks)))


def population_stats(pop, edges):
    """全反転を同一視した辺状態で多様性を測る。"""
    z = np.stack([pop[..., i] != pop[..., j] for i, j in edges], axis=-1)
    unique = [len(np.unique(a, axis=0)) for a in z]
    distances = [np.mean(z[:, i] != z[:, j], axis=1)
                 for i in range(pop.shape[1]) for j in range(i+1, pop.shape[1])]
    return float(np.mean(unique)), float(np.mean(distances))


def run_controls(out, repeats):
    """前回の手作りA/B配置を同じSA設定で測定する。CIM出力とは分ける。"""
    edges = graph_edges(False)
    ground = np.array([(-1)**(r+c) for r in range(4) for c in range(4)], dtype=np.int8)
    point = ground.copy()
    point[5] *= -1
    wall = ground.copy()
    wall[[4*r+c for r in range(4) for c in range(2)]] *= -1
    states = np.stack([point, wall])
    assert np.all(recut(states, edges) == 20)
    temps = [0.1, 0.5, 2.0]
    cuts = np.empty((3, 3, 2, repeats))
    solutions = np.empty((3, 3, 2, repeats, N), dtype=np.int8)
    for ti, temp in enumerate(temps):
        for bi, sweeps in enumerate(SWEEPS):
            for si, state in enumerate(states):
                c, s = simulate_sa_warm(N, edges, None, np.tile(state, (repeats, 1)),
                                       N*sweeps, t_start=temp, t_end=0.01,
                                       seeds=np.arange(repeats, dtype=np.int64)+700000)
                check_cuts(s, c, edges, 24)
                cuts[ti, bi, si], solutions[ti, bi, si] = c, s
    np.savez_compressed(out / "controls.npz", initial=states, cuts=cuts,
                        solutions=solutions, temperatures=temps, sweeps=SWEEPS)
    return [dict(temperature=t, sweeps=b, configuration=name, **stats(cuts[ti, bi, si], 24))
            for ti, t in enumerate(temps) for bi, b in enumerate(SWEEPS)
            for si, name in enumerate(["point", "wall"])]


def run_graph(out, key, diagonal, args):
    """実CIM、SA、メメティックGAと交叉の診断を同一グラフで走らせる。"""
    start = time.perf_counter()
    edges = graph_edges(diagonal)
    spectrum, optimum, degeneracy = exact_spectrum(edges)
    J = build_coupling_matrix(N, edges, coupling=-0.03)
    cim_states, cim_cuts, elapsed = [], [], []
    for rounds in CHECKPOINTS:
        t0 = time.perf_counter()
        c, s = simulate_cim_batch(N, J, edges, num_rounds=rounds,
                                  num_trials=args.candidates, **CIM_PARAMS,
                                  seeds=np.arange(args.candidates, dtype=np.int64))
        elapsed.append(time.perf_counter()-t0)
        check_cuts(s, c, edges, optimum)
        cim_states.append(s)
        cim_cuts.append(c)
        print(key, "CIM", rounds, stats(c, optimum), flush=True)
    cim_states, cim_cuts = np.asarray(cim_states), np.asarray(cim_cuts)
    random_states = np.random.default_rng(9128).integers(0, 2, (args.candidates, N), dtype=np.int8)
    source_states = np.concatenate([random_states[None], cim_states], axis=0)
    initial_cuts = recut(source_states, edges)
    sa_cuts = np.empty((4, len(SWEEPS), args.candidates, args.repeats))
    sa_states = np.empty((*sa_cuts.shape, N), dtype=np.int8)
    sa_seeds = np.arange(args.candidates*args.repeats, dtype=np.int64)+100000
    for si, states in enumerate(source_states):
        for bi, sweeps in enumerate(SWEEPS):
            c, s = simulate_sa_warm(N, edges, None, np.repeat(states, args.repeats, axis=0),
                                   N*sweeps, t_start=0.5, t_end=0.01, seeds=sa_seeds)
            check_cuts(s, c, edges, optimum)
            sa_cuts[si, bi] = c.reshape(args.candidates, args.repeats)
            sa_states[si, bi] = s.reshape(args.candidates, args.repeats, N)

    # 一個体ごとの局所探索前後も保存。集団としての最良値とは区別する。
    ts_budgets = [4, 32]
    ts_cuts, ts_states = [], []
    for budget in ts_budgets:
        c, s = tabu_refine_batch(N, edges, None, source_states[-1], ts_iters=budget,
                                cr=32, gamma_pert=2, alpha_tenure=2,
                                seeds=np.arange(args.candidates, dtype=np.int64)+300000)
        check_cuts(s, c, edges, optimum)
        ts_cuts.append(c)
        ts_states.append(s)

    # 各独立runの8個体には別のCIM seedを割当てる。
    popsize = 8
    ga_trials = args.candidates // popsize
    ga_seeds = np.arange(ga_trials, dtype=np.int64)+500000
    ga_history = []
    ga_population_history = []
    ga_rows = []
    for ts_budget in ts_budgets:
        for source_idx, name in [(0, "random"), (3, "cim1500")]:
            observations = []
            initial_pop = source_states[source_idx].reshape(ga_trials, popsize, N)
            def observe(generation, pops, fits):
                check_cuts(pops, fits, edges, optimum)
                u, d = population_stats(pops, edges)
                observations.append(pops.copy())
                ga_rows.append(dict(ts_iters=ts_budget, source=name, generation=generation,
                                    mean_best=float(fits.max(axis=1).mean()),
                                    success=float(np.mean(fits.max(axis=1) == optimum)),
                                    mean_unique=u, edge_distance=d))
            c, s, history = simulate_ga_batch(
                N, edges, None, ga_trials, pop_size=popsize,
                max_generations=args.ga_generations, ts_iters=ts_budget,
                cr=32, gamma_pert=2, alpha_tenure=2, beta_quality=0.6,
                init_population=initial_pop, seeds=ga_seeds, align_rng_with_cold=True,
                return_history=True, population_observer=observe)
            check_cuts(s, c, edges, optimum)
            ga_history.append(history)
            ga_population_history.append(np.asarray(observations))

    # 同じcutの実CIM親について、現行交叉が親を超えるかを測る。
    # 128-roundの全候補から親を抽出。全反転だけ異なる同じ解は除外。
    parent_pool = cim_states[1]
    pairs = [(i, j) for i in range(args.candidates) for j in range(i+1, args.candidates)
             if cim_cuts[1, i] == cim_cuts[1, j]
             and 0 < np.sum(parent_pool[i] != parent_pool[j]) < N]
    rng = np.random.default_rng(8928)
    if len(pairs) > 128:
        pairs = [pairs[i] for i in rng.choice(len(pairs), 128, replace=False)]
    offspring, pair_info = [], []
    for i, j in pairs:
        za = np.array([parent_pool[i, u] != parent_pool[i, v] for u, v in edges])
        zb = np.array([parent_pool[j, u] != parent_pool[j, v] for u, v in edges])
        pair_info.append([i, j, cim_cuts[1, i], np.mean(za != zb)])
        for _ in range(args.repeats):
            offspring.append(_grouping_crossover(parent_pool[i].astype(np.int8),
                                                parent_pool[j].astype(np.int8), rng))
    offspring = np.asarray(offspring, dtype=np.int8).reshape(-1, N)
    if len(offspring):
        off_cuts = recut(offspring, edges).reshape(len(pairs), args.repeats)
        refined_cuts, refined_states = tabu_refine_batch(
            N, edges, None, offspring, ts_iters=4, cr=32, gamma_pert=2,
            alpha_tenure=2, seeds=np.arange(len(offspring), dtype=np.int64)+400000)
        check_cuts(refined_states, refined_cuts, edges, optimum)
        refined_cuts = refined_cuts.reshape(len(pairs), args.repeats)
    else:
        off_cuts = refined_cuts = np.zeros((0, args.repeats))
        refined_states = np.zeros((0, N), dtype=np.int8)

    np.savez_compressed(out / f"{key}.npz", edges=edges, exact_cuts=spectrum,
                        source_states=source_states, initial_cuts=initial_cuts,
                        gains=gains(source_states, edges), sa_cuts=sa_cuts, sa_states=sa_states,
                        ts_cuts=ts_cuts, ts_states=ts_states,
                        ga_history=ga_history, ga_populations=ga_population_history,
                        parent_pairs=np.asarray(pair_info).reshape(-1, 4),
                        offspring=offspring, offspring_cuts=off_cuts,
                        offspring_ts_cuts=refined_cuts, offspring_ts_states=refined_states)
    rows = [dict(source=name, sweeps=0, **stats(initial_cuts[si], optimum))
            for si, name in enumerate(["random", "cim32", "cim128", "cim1500"])]
    rows += [dict(source=name, sweeps=b, **stats(sa_cuts[si, bi], optimum))
             for si, name in enumerate(["random", "cim32", "cim128", "cim1500"])
             for bi, b in enumerate(SWEEPS)]
    info = dict(n=N, m=len(edges), optimum=optimum, optimum_configurations=degeneracy,
                coupling=-0.03, edges=edges, cim_params=CIM_PARAMS,
                cim_seconds=elapsed, sa_rows=rows, ga_rows=ga_rows,
                ts_rows=[dict(iterations=b, **stats(c, optimum)) for b, c in zip(ts_budgets, ts_cuts)],
                crossover_pairs=len(pairs), total_seconds=time.perf_counter()-start)
    print(key, "SA from CIM1500", rows[-3:], flush=True)
    print(key, "TS", info["ts_rows"], flush=True)
    return info


def main():
    """全比較を実行し、上書きなしの保存先へ生データを保存する。"""
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--control-repeats", type=int, default=1024)
    parser.add_argument("--ga-generations", type=int, default=8)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.candidates < 8 or args.candidates % 8:
        parser.error("candidatesは8以上の8の倍数")
    if min(args.repeats, args.control_repeats) < 1 or args.ga_generations < 0:
        parser.error("反復数は正、世代数は非負")
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_]+", args.tag):
        parser.error("tagは半角英数字とアンダースコアのみ")
    set_num_threads(4)
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name.split("_", 1)[0][1:]) for p in root.iterdir()
                if p.is_dir() and re.match(r"^v\d+_", p.name)]
    version = max(versions, default=0)+1
    desc = f"n16_candidates{args.candidates}_repeats{args.repeats}_ga{args.ga_generations}"
    if args.tag:
        desc += "_"+args.tag
    out = root / f"v{version}_{desc}"
    out.mkdir(exist_ok=False)
    print("OUTPUT", out, flush=True)
    source_hashes = {}
    for src in [Path(__file__), ROOT/"modules/CIM.py", ROOT/"modules/SA.py", ROOT/"modules/GA.py"]:
        source_hashes[src.relative_to(ROOT).as_posix()] = hashlib.sha256(src.read_bytes()).hexdigest()
        shutil.copy2(src, out/src.name)
    summary = dict(date=date.today().isoformat(), args=vars(args), source_hashes=source_hashes,
                   cim_checkpoints=CHECKPOINTS, sa_sweeps=SWEEPS, sa_temperature=[0.5, 0.01],
                   interpretation="Fixed downstream budgets; not an equal total wall-time speed comparison.",
                   graphs={})
    # JITを先に暖め、主比較のCIM時間から初回コンパイルを除く。
    warm_edges = graph_edges(False)
    simulate_cim_batch(N, build_coupling_matrix(N, warm_edges, coupling=-.03), warm_edges,
                       num_rounds=1, num_trials=1, **CIM_PARAMS, seeds=np.array([99999]))
    for key, diag in [("square", False), ("triangular", True)]:
        summary["graphs"][key] = run_graph(out, key, diag, args)
    summary["controls"] = run_controls(out, args.control_repeats)
    (out/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("DONE", out, flush=True)

