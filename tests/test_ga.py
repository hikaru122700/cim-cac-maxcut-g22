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


def test_ga_matched_rng_alignment_changes_only_first_individual():
    """coldの第1個体をwarmに渡せば、aligned実行はcoldと完全一致する。"""
    n = 10
    edges = [(i, (i + 1) % n) for i in range(n)]
    weights = [1.0] * len(edges)
    seeds = np.arange(3, dtype=np.int64)
    warm = np.empty((len(seeds), n), dtype=np.int8)
    for t, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        warm[t] = (rng.random(n) < 0.5).astype(np.int8)

    kwargs = dict(
        n=n,
        edges=edges,
        weights=weights,
        num_trials=len(seeds),
        pop_size=4,
        max_generations=3,
        ts_iters=200,
        cr=50,
        gamma_pert=2,
        alpha_tenure=2,
        beta_quality=0.6,
        seeds=seeds,
    )
    cold_cuts, cold_signs = simulate_ga_batch(**kwargs)
    warm_cuts, warm_signs = simulate_ga_batch(
        **kwargs,
        init_signs=warm,
        align_rng_with_cold=True,
    )
    np.testing.assert_array_equal(warm_cuts, cold_cuts)
    np.testing.assert_array_equal(warm_signs, cold_signs)


def test_ga_full_population_alignment_matches_cold_population():
    """cold生成と同じ全初期集団を渡せば、aligned実行は完全一致する。"""
    n = 10
    pop_size = 4
    edges = [(i, (i + 1) % n) for i in range(n)]
    weights = [1.0] * len(edges)
    seeds = np.arange(3, dtype=np.int64)
    population = np.empty((len(seeds), pop_size, n), dtype=np.int8)
    for t, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        for j in range(pop_size):
            population[t, j] = (rng.random(n) < 0.5).astype(np.int8)

    kwargs = dict(
        n=n,
        edges=edges,
        weights=weights,
        num_trials=len(seeds),
        pop_size=pop_size,
        max_generations=3,
        ts_iters=200,
        cr=50,
        gamma_pert=2,
        alpha_tenure=2,
        beta_quality=0.6,
        seeds=seeds,
    )
    cold_cuts, cold_signs = simulate_ga_batch(**kwargs)
    full_cuts, full_signs = simulate_ga_batch(
        **kwargs,
        init_population=population,
        align_rng_with_cold=True,
    )
    np.testing.assert_array_equal(full_cuts, cold_cuts)
    np.testing.assert_array_equal(full_signs, cold_signs)


def test_ga_rejects_conflicting_warm_inputs():
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    with np.testing.assert_raises(ValueError):
        simulate_ga_batch(
            n,
            edges,
            [1.0] * len(edges),
            1,
            pop_size=2,
            max_generations=0,
            ts_iters=10,
            init_signs=np.ones((1, n), dtype=np.int8),
            init_population=np.ones((1, 2, n), dtype=np.int8),
            seeds=np.array([0]),
        )
