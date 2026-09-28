"""保存済みCIM解の同一cut比較を独立SA乱数で追試し、交叉の対照を追加する。"""

from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from numba import set_num_threads

from modules.SA import simulate_sa_warm
from modules.GA import tabu_refine_batch


def diagnose(out: Path):
    """最適値への非悪化経路長を厳密計算し、独立乱数で到達率を測る。"""
    set_num_threads(4)
    targets = [out/"diagnostics.npz", out/"diagnostics.json", out/Path(__file__).name]
    if any(p.exists() for p in targets):
        raise FileExistsError("診断結果が存在するため上書きしません")
    summary = json.loads((out/"summary.json").read_text(encoding="utf-8"))
    data = np.load(out/"triangular.npz")
    cuts = data["exact_cuts"]
    optimum = int(cuts.max())
    assert optimum == 24
    # cut>=23の状態部分グラフで全最適解からBFS。途中でcutを下げない経路を測る。
    distance = np.full(len(cuts), -1, dtype=np.int16)
    next_state = np.full(len(cuts), -1, dtype=np.int32)
    targets_ids = np.flatnonzero(cuts == optimum)
    distance[targets_ids] = 0
    queue = deque(targets_ids.tolist())
    while queue:
        state = queue.popleft()
        for i in range(16):
            neighbor = state ^ (1 << i)
            if cuts[neighbor] >= optimum-1 and distance[neighbor] < 0:
                distance[neighbor] = distance[state]+1
                next_state[neighbor] = state
                queue.append(neighbor)
    initial = data["source_states"][3]
    ids = np.sum(initial.astype(np.int64)*(1 << np.arange(16)), axis=1)
    candidate_distance = distance[ids]
    edges = summary["graphs"]["triangular"]["edges"]
    repeats = 256
    seeds = np.arange(len(initial)*repeats, dtype=np.int64)+2000000
    result_cuts, result_states = simulate_sa_warm(
        16, edges, None, np.repeat(initial, repeats, axis=0), 160,
        t_start=0.5, t_end=0.01, seeds=seeds)
    recount = sum((result_states[:, i] != result_states[:, j]).astype(int) for i, j in edges)
    assert np.array_equal(recount, result_cuts)
    result_cuts = result_cuts.reshape(len(initial), repeats)
    groups = []
    for d in sorted(set(candidate_distance.tolist())):
        mask = (candidate_distance == d) & (data["initial_cuts"][3] == 23)
        if mask.any():
            groups.append(dict(distance=int(d), candidates=int(mask.sum()),
                               success=float(np.mean(result_cuts[mask] == optimum)),
                               successes=int(np.sum(result_cuts[mask] == optimum)),
                               attempts=int(mask.sum()*repeats)))

    # 成功率の大小で選ばず、経路長1と8の各群で最小seedの配置を例示する。
    examples = []
    path_ids = []
    for d in [1, 8]:
        seed_idx = int(np.flatnonzero(candidate_distance == d)[0])
        state_id = int(ids[seed_idx])
        route = [state_id]
        while next_state[route[-1]] >= 0:
            route.append(int(next_state[route[-1]]))
        route_cuts = cuts[route].tolist()
        assert route_cuts[-1] == 24 and min(route_cuts) == 23
        path_ids.append(route)
        examples.append(dict(cim_seed=seed_idx, distance=d,
                             success=float(np.mean(result_cuts[seed_idx] == optimum)),
                             successes=int(np.sum(result_cuts[seed_idx] == optimum)),
                             attempts=repeats, state_id=state_id,
                             route_ids=route, route_cuts=route_cuts,
                             flip_vertices=[int((a^b).bit_length()-1) for a,b in zip(route[:-1],route[1:])]))

    cross = {}
    baseline_cuts = {}
    for key in ["square", "triangular"]:
        d = np.load(out/f"{key}.npz")
        pairs = d["parent_pairs"]
        reps = d["offspring_cuts"].shape[1]
        parents = np.concatenate([np.stack([d["source_states"][2, int(pair[k % 2])]
                                           for k in range(reps)]) for pair in pairs])
        g_edges = summary["graphs"][key]["edges"]
        base_c, base_s = tabu_refine_batch(
            16, g_edges, None, parents, ts_iters=4, cr=32, gamma_pert=2,
            alpha_tenure=2, seeds=np.arange(len(parents), dtype=np.int64)+400000)
        assert np.array_equal(base_c, sum((base_s[:, i] != base_s[:, j]).astype(int) for i,j in g_edges))
        base_c = base_c.reshape(len(pairs), reps)
        baseline_cuts[key] = base_c
        best_parent = pairs[:, 2, None]
        cross[key] = dict(pairs=len(pairs), repeats=reps,
                          parent_mean=float(best_parent.mean()),
                          raw_child_mean=float(d["offspring_cuts"].mean()),
                          child_ts_mean=float(d["offspring_ts_cuts"].mean()),
                          parent_ts_mean=float(base_c.mean()),
                          raw_child_improvement=float(np.mean(d["offspring_cuts"] > best_parent)),
                          child_ts_improvement=float(np.mean(d["offspring_ts_cuts"] > best_parent)),
                          parent_ts_improvement=float(np.mean(base_c > best_parent)),
                          child_minus_parent_ts=float(np.mean(d["offspring_ts_cuts"]-base_c)))
    np.savez_compressed(out/"diagnostics.npz", exact_distance=distance, next_state=next_state,
                        candidate_distance=candidate_distance, holdout_cuts=result_cuts,
                        holdout_states=result_states.reshape(len(initial), repeats, 16),
                        square_parent_ts_cuts=baseline_cuts["square"],
                        triangular_parent_ts_cuts=baseline_cuts["triangular"])
    result = dict(holdout_repeats=repeats, seed_start=2000000, sweeps=10,
                  temperature=[0.5,0.01], groups=groups, examples=examples, crossover=cross,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interpretation="Exploratory feature chosen using primary run; holdout repeats validate SA probabilities for the same fixed states, not unseen graphs.")
    (out/"diagnostics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(__file__, out/Path(__file__).name)
    print(json.dumps(result, ensure_ascii=False, indent=2))

