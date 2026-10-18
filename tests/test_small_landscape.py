"""scripts/benchmarks/small_landscape.py の厳密量を、素朴な計算と照合する。"""
import itertools
from collections import deque

import numpy as np
import pytest

from scripts.benchmarks.small_landscape import (
    all_problems,
    csr,
    enumerate_cuts,
    exact_landscape,
    ids_to_states,
    make_problem,
    state_ids,
)


def _naive_cut(state, edges, weights):
    return sum(w for (a, b), w in zip(edges, weights) if state[a] != state[b])


def _small_pm1(n=8, seed=0):
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i, j in itertools.combinations(range(n), 2) if rng.random() < 0.5]
    weights = rng.choice([-1.0, 1.0], len(edges)).tolist()
    return n, edges, weights


def _neighbors(x, n):
    full = (1 << (n - 1)) - 1
    return [x ^ (1 << b) for b in range(n - 1)] + [x ^ full]


def _naive_barrier_and_path(cuts, x, n):
    opt = cuts.max()
    for level in range(int(cuts[x]), int(cuts.min()) - 1, -1):
        seen = {x: 0}
        q = deque([x])
        while q:
            y = q.popleft()
            if cuts[y] == opt:
                return int(cuts[x]) - level, seen[y]
            for z in _neighbors(y, n):
                if z not in seen and cuts[z] >= level:
                    seen[z] = seen[y] + 1
                    q.append(z)
    raise AssertionError("unreachable")


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_enumeration_matches_naive_cut(seed):
    n, edges, weights = _small_pm1(seed=seed)
    cuts = enumerate_cuts(n, *csr(n, edges, weights))
    states = ids_to_states(np.arange(1 << (n - 1)), n)
    naive = np.array([_naive_cut(s, edges, weights) for s in states])
    np.testing.assert_array_equal(cuts, naive)


@pytest.mark.parametrize("seed", [0, 3])
def test_barrier_and_path_match_naive_search(seed):
    n, edges, weights = _small_pm1(n=9, seed=seed)
    cand = np.arange(1 << (n - 1), dtype=np.int64)
    land = exact_landscape(n, edges, weights, cand)
    for x in cand[::3]:
        b, p = _naive_barrier_and_path(land["cuts"], int(x), n)
        assert land["barrier"][x] == b
        assert land["path_length"][x] == p
    assert np.all(land["barrier"][land["cuts"] == land["opt"]] == 0)
    assert np.all(land["path_length"][land["cuts"] == land["opt"]] == 0)


def test_state_ids_identify_global_flip():
    s = np.array([[0, 1, 1, 0, 1], [1, 0, 0, 1, 0]], dtype=np.int8)
    ids = state_ids(s)
    assert ids[0] == ids[1]
    np.testing.assert_array_equal(ids_to_states(ids[:1], 5)[0], s[0])


def test_opt_distance_is_zero_for_optima_and_symmetric():
    n, edges, weights = _small_pm1(n=8, seed=5)
    cand = np.arange(1 << (n - 1), dtype=np.int64)
    land = exact_landscape(n, edges, weights, cand)
    assert np.all(land["opt_distance"][land["opt_ids"]] == 0)
    assert land["opt_distance"].max() <= n // 2


def test_problem_set_is_deterministic_and_split():
    a, b = all_problems(), all_problems()
    assert [p["edges"] for p in a] == [p["edges"] for p in b]
    assert len(a) == 30
    assert sum(p["split"] == "dev" for p in a) == 15
    assert all(18 <= p["n"] <= 24 for p in a)
    with pytest.raises(ValueError):
        make_problem("grid", 0)
