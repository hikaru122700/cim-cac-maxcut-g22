"""小規模 MAX-CUT 問題の生成と、全列挙による地形の厳密量(論文化計画 v2 のステップ 2)。

状態はスピン 0 を 0 に固定した 2^(n-1) 個で表す(全反転を同一視)。状態番号 x の
ビット i-1 がスピン i(i>=1)。スピン 0 の反転は、残り全スピンの反転と同じ状態になる。

厳密量(候補 s、cut c(s)、最適値 opt):
  障壁   B(s) = c(s) - l*(s)。l*(s) は「cut >= l の状態だけを 1-flip でたどって
         s から最適解に行ける」最大の l。B=0 なら cut を下げずに最適解に行ける。
  経路長 L(s) = cut >= l*(s) の状態だけを通る、s から最適解までの最短手数。
  距離   d(s) = 最も近い最適解までのハミング距離(全反転を同一視)。
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from numba import njit

FAMILIES = ("reg3", "pm1", "trilat")
PROBLEMS_PER_FAMILY = 10
SIZES = {"reg3": (20, 22, 24), "pm1": (20, 22, 24), "trilat": ((4, 5), (4, 6))}
TRILAT_DIAGONAL_PROB = 0.6
PM1_MEAN_DEGREE = 4.0


def make_problem(family: str, index: int) -> dict:
    """(family, index) から決定的に問題を作る。index が偶数なら開発用、奇数なら検証用。"""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}")
    seed = 1000 * (FAMILIES.index(family) + 1) + index
    rng = np.random.default_rng(seed)
    sizes = SIZES[family]
    size = sizes[index % len(sizes)]
    if family == "reg3":
        g = nx.random_regular_graph(3, size, seed=seed)
        while not nx.is_connected(g):
            seed += 100_000
            g = nx.random_regular_graph(3, size, seed=seed)
        edges = sorted(tuple(sorted(e)) for e in g.edges())
        weights = [1.0] * len(edges)
        n = size
    elif family == "pm1":
        n = size
        p = PM1_MEAN_DEGREE / (n - 1)
        while True:
            g = nx.gnp_random_graph(n, p, seed=int(rng.integers(0, 2**31)))
            if nx.is_connected(g):
                break
        edges = sorted(tuple(sorted(e)) for e in g.edges())
        weights = rng.choice([-1.0, 1.0], len(edges)).tolist()
    else:
        rows, cols = size
        n = rows * cols
        edges = [(cols * r + c, cols * r + c + 1) for r in range(rows) for c in range(cols - 1)]
        edges += [(cols * r + c, cols * (r + 1) + c) for r in range(rows - 1) for c in range(cols)]
        for r in range(rows - 1):
            for c in range(cols - 1):
                if rng.random() < TRILAT_DIAGONAL_PROB:
                    if rng.random() < 0.5:
                        edges.append((cols * r + c, cols * (r + 1) + c + 1))
                    else:
                        edges.append((cols * r + c + 1, cols * (r + 1) + c))
        edges = sorted(tuple(sorted(e)) for e in edges)
        weights = [1.0] * len(edges)
    return {
        "name": f"{family}_{index:02d}",
        "family": family,
        "index": index,
        "split": "dev" if index % 2 == 0 else "test",
        "n": int(n),
        "edges": [(int(a), int(b)) for a, b in edges],
        "weights": [float(w) for w in weights],
    }


def all_problems() -> list[dict]:
    return [make_problem(f, i) for f in FAMILIES for i in range(PROBLEMS_PER_FAMILY)]


def csr(n: int, edges, weights):
    nbr = [[] for _ in range(n)]
    for (a, b), w in zip(edges, weights):
        nbr[a].append((b, w))
        nbr[b].append((a, w))
    indptr = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        indptr[v + 1] = indptr[v] + len(nbr[v])
    indices = np.array([u for v in range(n) for u, _ in nbr[v]], dtype=np.int64)
    data = np.array([w for v in range(n) for _, w in nbr[v]], dtype=np.float64)
    return indptr, indices, data


def state_ids(signs01: np.ndarray) -> np.ndarray:
    """(K, n) の 0/1 状態を、スピン 0 を 0 に揃えた状態番号にする。"""
    s = np.asarray(signs01, dtype=np.int64)
    s = np.where(s[:, :1] == 1, 1 - s, s)
    weights = 1 << np.arange(s.shape[1] - 1, dtype=np.int64)
    return s[:, 1:] @ weights


def ids_to_states(ids: np.ndarray, n: int) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64)
    bits = (ids[:, None] >> np.arange(n - 1)) & 1
    return np.concatenate([np.zeros((len(ids), 1), dtype=np.int8), bits.astype(np.int8)], axis=1)


@njit(cache=True)
def enumerate_cuts(n, indptr, indices, data):
    """全 2^(n-1) 状態の cut(整数重み前提)を Gray 符号順に増分計算する。"""
    num = 1 << (n - 1)
    cuts = np.empty(num, dtype=np.int32)
    s = np.zeros(n, dtype=np.int8)
    gains = np.zeros(n, dtype=np.float64)
    for v in range(n):
        for p in range(indptr[v], indptr[v + 1]):
            gains[v] += data[p]
    cut = 0.0
    cuts[0] = 0
    for k in range(1, num):
        tz = 0
        while (k >> tz) & 1 == 0:
            tz += 1
        v = tz + 1
        cut += gains[v]
        sv_old = s[v]
        gains[v] = -gains[v]
        for p in range(indptr[v], indptr[v + 1]):
            u = indices[p]
            if s[u] == sv_old:
                gains[u] -= 2.0 * data[p]
            else:
                gains[u] += 2.0 * data[p]
        s[v] = 1 - sv_old
        cuts[k ^ (k >> 1)] = int(np.round(cut))
    return cuts


@njit(cache=True)
def _find(parent, x):
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:
        nxt = parent[x]
        parent[x] = root
        x = nxt
    return root


@njit(cache=True)
def barrier_levels(n, cuts, cand_ids, opt):
    """各候補の l*(s)(cut >= l で最適解とつながる最大の l)を、cut の高い順に状態を足して求める。"""
    num = cuts.shape[0]
    full_mask = num - 1
    order = np.argsort(-cuts, kind="mergesort")
    parent = np.arange(num, dtype=np.int64)
    size = np.ones(num, dtype=np.int64)
    has_opt = cuts == opt
    added = np.zeros(num, dtype=np.bool_)
    unset = np.iinfo(np.int32).min
    level = np.full(cand_ids.shape[0], unset, dtype=np.int64)
    remaining = cand_ids.shape[0]
    i = 0
    while i < num and remaining > 0:
        lvl = cuts[order[i]]
        j = i
        while j < num and cuts[order[j]] == lvl:
            x = order[j]
            added[x] = True
            for b in range(n):
                y = x ^ ((1 << b) if b < n - 1 else full_mask)
                if added[y]:
                    rx = _find(parent, x)
                    ry = _find(parent, y)
                    if rx != ry:
                        if size[rx] < size[ry]:
                            rx, ry = ry, rx
                        parent[ry] = rx
                        size[rx] += size[ry]
                        has_opt[rx] = has_opt[rx] or has_opt[ry]
            j += 1
        i = j
        for c in range(cand_ids.shape[0]):
            if level[c] == unset and added[cand_ids[c]] and has_opt[_find(parent, cand_ids[c])]:
                level[c] = lvl
                remaining -= 1
    return level


@njit(cache=True)
def bfs_from_optima(n, cuts, min_cut, opt):
    """cut >= min_cut の状態だけを通る、最適解からの最短手数(届かなければ -1)。"""
    num = cuts.shape[0]
    full_mask = num - 1
    dist = np.full(num, -1, dtype=np.int32)
    queue = np.empty(num, dtype=np.int64)
    head = 0
    tail = 0
    for x in range(num):
        if cuts[x] == opt:
            dist[x] = 0
            queue[tail] = x
            tail += 1
    while head < tail:
        x = queue[head]
        head += 1
        for b in range(n):
            y = x ^ ((1 << b) if b < n - 1 else full_mask)
            if dist[y] < 0 and cuts[y] >= min_cut:
                dist[y] = dist[x] + 1
                queue[tail] = y
                tail += 1
    return dist


def nearest_optimum_distance(cand_ids: np.ndarray, opt_ids: np.ndarray, n: int) -> np.ndarray:
    """全反転を同一視したハミング距離で、最も近い最適解までの距離。"""
    full = (1 << (n - 1)) - 1
    out = np.empty(len(cand_ids), dtype=np.int64)
    for k, c in enumerate(cand_ids):
        x = np.bitwise_xor(opt_ids, c)
        d = np.array([bin(int(v)).count("1") for v in x])
        dc = np.array([bin(int(v) ^ full).count("1") + 1 for v in x])  # スピン 0 も反転
        out[k] = int(min(d.min(), dc.min()))
    return out


def exact_landscape(n: int, edges, weights, cand_ids: np.ndarray) -> dict:
    """候補ごとの障壁・経路長・最適解までの距離と、問題全体の量を返す。"""
    indptr, indices, data = csr(n, edges, weights)
    cuts = enumerate_cuts(n, indptr, indices, data)
    opt = int(cuts.max())
    opt_ids = np.flatnonzero(cuts == opt).astype(np.int64)
    cand_ids = np.asarray(cand_ids, dtype=np.int64)
    lvl = barrier_levels(n, cuts, cand_ids, opt)
    barrier = cuts[cand_ids].astype(np.int64) - lvl
    path = np.empty(len(cand_ids), dtype=np.int64)
    for level in np.unique(lvl):
        dist = bfs_from_optima(n, cuts, int(level), opt)
        m = lvl == level
        path[m] = dist[cand_ids[m]]
    return {
        "cuts": cuts,
        "opt": opt,
        "n_opt": int(len(opt_ids)),
        "opt_ids": opt_ids,
        "barrier": barrier,
        "path_length": path,
        "opt_distance": nearest_optimum_distance(cand_ids, opt_ids, n),
    }
