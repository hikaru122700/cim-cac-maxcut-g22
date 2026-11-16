"""実験 A・B(sa_path_bks_distance / start_distance_causal)の補助関数のテスト。"""

import numpy as np

from scripts.benchmarks.bks_set_distance import dist_matrix
from scripts.benchmarks.sa_path_bks_distance import _nearest_and_start, make_checkpoints
from scripts.benchmarks.start_distance_causal import binned, ols, perturb


def _signs(rng, shape):
    return rng.choice(np.array([-1, 1], np.int8), shape)


def test_perturb_flips_exactly_k_and_keeps_input():
    rng = np.random.default_rng(0)
    base = _signs(rng, 200)
    keep = base.copy()
    out = perturb(base, 30, np.random.default_rng(1))
    assert np.array_equal(base, keep)
    assert int((out != base).sum()) == 30
    assert np.isclose(dist_matrix(out[None], base[None])[0, 0], 30 / 200)


def test_nearest_and_start_matches_dist_matrix_with_flip_symmetry():
    rng = np.random.default_rng(2)
    refs = _signs(rng, (5, 101))
    x = -refs[3].copy()  # 全反転は同じ解
    x[:7] = -x[:7]
    start = _signs(rng, 101)
    near, d_start = _nearest_and_start(x, start, refs)
    assert np.isclose(near, dist_matrix(x[None], refs).min())
    assert np.isclose(near, 7 / 101)
    assert np.isclose(d_start, dist_matrix(x[None], start[None])[0, 0])


def test_make_checkpoints_starts_at_zero_and_ends_at_total():
    cps = make_checkpoints(100, 50, count=20)
    assert cps[0] == 0 and cps[-1] == 50 * 100
    assert np.all(np.diff(cps) > 0)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=500), rng.normal(size=500)
    y = 1.0 + 2.0 * a - 0.5 * b
    fit = ols(y, {"a": a, "b": b})
    assert np.isclose(fit["intercept"]["coef"], 1.0)
    assert np.isclose(fit["a"]["coef"], 2.0)
    assert np.isclose(fit["b"]["coef"], -0.5)


def test_binned_skips_small_bins():
    d = np.r_[np.full(20, 0.05), np.full(5, 0.15)]
    gap = np.r_[np.full(20, 0.1), np.full(5, 0.3)]
    rows = binned(d, gap)
    assert len(rows) == 1 and rows[0]["n"] == 20 and np.isclose(rows[0]["gap_mean"], 0.1)
