"""高速化した warm-start SA が、高速化前のカーネルと完全に同じ結果を返すことを確かめる。

_reference_warm は高速化前の modules/SA.py::_simulate_sa_batch_warm をそのまま写したもの(コンパイル条件も同じ)。
"""
import numpy as np
import pytest
from numba import njit, prange

from modules.SA import _build_csr_sa, simulate_sa_warm


@njit(cache=False, fastmath=True, parallel=True)
def _reference_warm(n, num_iters, num_trials, adj_indptr, adj_indices, adj_weights,
                    t_start, t_end, init_signs, seeds, consume_init_rng):
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.int8)
    log_ratio = np.log(t_end / t_start)
    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])
        if consume_init_rng:
            for i in range(n):
                np.random.random()
        x = np.empty(n, dtype=np.int8)
        for i in range(n):
            x[i] = init_signs[trial_idx, i]
        cur_cut = 0.0
        for i in range(n):
            for k in range(adj_indptr[i], adj_indptr[i + 1]):
                j = adj_indices[k]
                if j > i and x[i] != x[j]:
                    cur_cut += adj_weights[k]
        best_cut = cur_cut
        best_x = x.copy()
        for it in range(num_iters):
            progress = it / num_iters
            T = t_start * np.exp(log_ratio * progress)
            v = np.random.randint(0, n)
            delta = 0.0
            for k in range(adj_indptr[v], adj_indptr[v + 1]):
                u = adj_indices[k]
                w = adj_weights[k]
                if x[v] == x[u]:
                    delta += w
                else:
                    delta -= w
            if delta > 0:
                x[v] = -x[v]
                cur_cut += delta
            elif T > 0:
                if np.random.random() < np.exp(delta / T):
                    x[v] = -x[v]
                    cur_cut += delta
            if cur_cut > best_cut:
                best_cut = cur_cut
                for i in range(n):
                    best_x[i] = x[i]
        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_x[i]
    return best_cuts_out, best_signs_out


def _graph(n: int, m: int, signed: bool, seed: int):
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < m:
        a, b = rng.integers(0, n, 2)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    edges = sorted(pairs)
    weights = list(rng.choice([-1.0, 1.0], m)) if signed else None
    return edges, weights


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("align", [False, True])
@pytest.mark.parametrize("t_start", [2.0, 0.5])
def test_fast_warm_matches_reference(signed, align, t_start):
    n, trials = 300, 12
    edges, weights = _graph(n, 1500, signed, seed=7)
    rng = np.random.default_rng(3)
    init = rng.choice(np.array([-1, 1], np.int8), (trials, n))
    seeds = 130_000_000 + np.arange(trials, dtype=np.int64)
    iters = 300 * n
    cut, signs = simulate_sa_warm(n, edges, weights, init, iters, t_start=t_start, t_end=0.001,
                                  seeds=seeds, align_rng_with_cold=align)
    ip, ix, w = _build_csr_sa(n, edges, weights)
    ref_cut, ref_signs = _reference_warm(n, iters, trials, ip, ix, w, t_start, 0.001,
                                         np.ascontiguousarray(init), seeds, align)
    np.testing.assert_array_equal(cut, ref_cut)
    np.testing.assert_array_equal(signs, ref_signs)


def test_fast_warm_does_not_modify_init():
    n = 50
    edges, _ = _graph(n, 200, False, seed=1)
    init = np.random.default_rng(0).choice(np.array([-1, 1], np.int8), (3, n))
    before = init.copy()
    simulate_sa_warm(n, edges, None, init, 50 * n, t_start=2.0, t_end=0.001, seeds=np.arange(3))
    np.testing.assert_array_equal(init, before)
