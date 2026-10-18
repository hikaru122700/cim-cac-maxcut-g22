"""modules/tabu_bucket.py(バケット版 TS)のテスト。"""
import numpy as np
import pytest

from modules.GA import (
    _build_csr,
    _cut_of,
    _resolve_ts_impl,
    load_graph,
    simulate_ga_batch,
    tabu_refine_batch,
)
from modules.tabu_bucket import max_weighted_degree, tabu_search_batch_bucket


def _csr(n, edges, weights):
    ea = np.array([e[0] for e in edges], dtype=np.int64)
    eb = np.array([e[1] for e in edges], dtype=np.int64)
    ew = np.asarray(weights, dtype=np.float64)
    return _build_csr(n, ea, eb, ew)


def _random_pm1_graph(n, m, seed):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        a, b = sorted(rng.integers(0, n, 2))
        if a != b:
            pairs.add((int(a), int(b)))
    edges = sorted(pairs)
    weights = rng.choice([-1.0, 1.0], len(edges)).tolist()
    return n, edges, weights


def _run_checked(n, edges, weights, ts_iters, cr, gamma, alpha, num=4, seed0=0):
    indptr, indices, data = _csr(n, edges, weights)
    offset = max_weighted_degree(n, indptr, data)
    init = np.random.default_rng(seed0).integers(0, 2, (num, n)).astype(np.int8)
    seeds = np.arange(seed0, seed0 + num, dtype=np.int64)
    signs, cuts, bad = tabu_search_batch_bucket(
        n, indptr, indices, data, init, ts_iters, cr, gamma, alpha, seeds, offset, True
    )
    return (indptr, indices, data), signs, cuts, bad


@pytest.mark.parametrize("case", ["G22", "pm1_dense_perturb", "pm1_sparse"])
def test_bucket_state_matches_full_scan_every_iteration(case):
    """毎反復、バケットから選ぶ手・同点数・aspiration が全走査の結果と一致する。"""
    if case == "G22":
        n, edges, weights = load_graph("input/G22.txt", return_weights=True)
        csr, signs, cuts, bad = _run_checked(n, edges, weights, 3000, 3000, 150, 15, num=2)
    elif case == "pm1_dense_perturb":
        n, edges, weights = _random_pm1_graph(60, 600, 1)
        csr, signs, cuts, bad = _run_checked(n, edges, weights, 4000, 40, 10, 3)
    else:
        n, edges, weights = _random_pm1_graph(300, 330, 2)
        csr, signs, cuts, bad = _run_checked(n, edges, weights, 4000, 200, 20, 5)
    assert bad.sum() == 0
    for t in range(len(cuts)):
        assert abs(_cut_of(n, *csr, signs[t]) - cuts[t]) < 1e-9


def test_bucket_is_deterministic_for_same_seed():
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    init = np.random.default_rng(3).integers(0, 2, (3, n)).astype(np.int8)
    kw = dict(ts_iters=5000, seeds=np.arange(3), ts_impl="bucket")
    c1, s1 = tabu_refine_batch(n, edges, weights, init, **kw)
    c2, s2 = tabu_refine_batch(n, edges, weights, init, **kw)
    np.testing.assert_array_equal(c1, c2)
    np.testing.assert_array_equal(s1, s2)


def test_bucket_quality_matches_scan():
    """同じ規則なので、乱数が違っても到達品質の平均は統計的に区別できない(4 SE 以内)。"""
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    num = 32
    init = np.random.default_rng(4).integers(0, 2, (num, n)).astype(np.int8)
    kw = dict(ts_iters=20000, seeds=np.arange(num))
    cb, _ = tabu_refine_batch(n, edges, weights, init, ts_impl="bucket", **kw)
    cs, _ = tabu_refine_batch(n, edges, weights, init, ts_impl="scan", **kw)
    se = np.sqrt(cb.var(ddof=1) / num + cs.var(ddof=1) / num)
    assert abs(cb.mean() - cs.mean()) < 4 * se


def test_resolve_ts_impl():
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    indptr, _, data = _csr(n, edges, weights)
    assert _resolve_ts_impl("auto", True, n, indptr, data) == max_weighted_degree(n, indptr, data)
    assert _resolve_ts_impl("auto", False, n, indptr, data) == -1
    assert _resolve_ts_impl("scan", True, n, indptr, data) == -1
    frac = data * 0.5
    assert _resolve_ts_impl("auto", True, n, indptr, frac) == -1
    with pytest.raises(ValueError):
        _resolve_ts_impl("bucket", True, n, indptr, frac)
    with pytest.raises(ValueError):
        _resolve_ts_impl("bucket", False, n, indptr, data)
    with pytest.raises(ValueError):
        _resolve_ts_impl("fast", True, n, indptr, data)


def test_ga_runs_with_bucket_and_scan():
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    indptr, indices, data = _csr(n, edges, weights)
    kw = dict(num_trials=2, pop_size=4, max_generations=3, ts_iters=2000, seeds=np.arange(2))
    for impl in ("bucket", "scan"):
        cuts, signs = simulate_ga_batch(n, edges, weights, **kw, ts_impl=impl)
        for t in range(2):
            assert abs(_cut_of(n, indptr, indices, data, signs[t].astype(np.int8)) - cuts[t]) < 1e-9
