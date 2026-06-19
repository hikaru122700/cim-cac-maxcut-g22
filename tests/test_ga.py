"""modules/GA.py(メメティック MAX-CUT)の回帰テスト。"""
import numpy as np

from modules.GA import (
    load_graph,
    simulate_ga_batch,
    tabu_refine_batch,
    _build_csr,
    _cut_of,
)


def _csr(n, edges, weights):
    ea = np.array([e[0] for e in edges], dtype=np.int64)
    eb = np.array([e[1] for e in edges], dtype=np.int64)
    ew = np.asarray(weights, dtype=np.float64)
    return _build_csr(n, ea, eb, ew)


def test_reported_cut_matches_recompute():
    """報告カット値が独立な再計算と一致する(増分更新の正しさ)。"""
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    cuts, signs = tabu_refine_batch(
        n, edges, weights,
        np.random.default_rng(0).integers(0, 2, (3, n)).astype(np.int8),
        ts_iters=5000, seeds=np.arange(3),
    )
    indptr, indices, data = _csr(n, edges, weights)
    for t in range(3):
        chk = _cut_of(n, indptr, indices, data, signs[t].astype(np.int8))
        assert abs(chk - cuts[t]) < 1e-6


def test_memetic_reaches_near_bks_g22():
    """フルメメティックが G22 で BKS(13359)近傍に到達する。"""
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    cuts, _ = simulate_ga_batch(
        n, edges, weights, num_trials=4, pop_size=8,
        max_generations=60, ts_iters=20000, seeds=np.arange(4),
    )
    assert cuts.max() >= 13330  # BKS 13359 の 99.8% 以上


def test_warm_start_improves_random():
    """warm-start(良い初期解)からの TS は乱数初期より良いか同等。"""
    n, edges, weights = load_graph("input/G22.txt", return_weights=True)
    rng = np.random.default_rng(1)
    rand_init = rng.integers(0, 2, (2, n)).astype(np.int8)
    c_rand, signs = tabu_refine_batch(
        n, edges, weights, rand_init, ts_iters=8000, seeds=np.arange(2)
    )
    # 一度磨いた解を再度 warm-start すると劣化しない
    c_warm, _ = tabu_refine_batch(
        n, edges, weights, signs.astype(np.int8), ts_iters=8000,
        seeds=np.arange(2, 4),
    )
    assert c_warm.max() >= c_rand.max()
