import numpy as np

from scripts.benchmarks.cim_stage_valley import linear_end_index, sample_rounds, signs_of, trace_stats


def test_linear_end_index_first_crossing():
    sat = np.array([0.0, 0.001, 0.009, 0.01, 0.5, 0.002])
    assert linear_end_index(sat) == 3
    assert linear_end_index(np.zeros(4)) == 3


def test_sample_rounds_dense_early_and_ends_at_last():
    r = sample_rounds(2400)
    assert r[0] == 0 and r[-1] == 2399
    assert np.all(np.diff(r) > 0)
    assert np.all(np.diff(r[r < 600]) == 2)


def test_signs_of_maps_zero_to_minus():
    assert signs_of(np.array([0.3, -0.1, 0.0])).tolist() == [1, -1, -1]


def test_trace_stats_aligned_state():
    n = 40
    u1 = np.ones(n) / np.sqrt(n)
    u2 = np.r_[np.ones(20), -np.ones(20)] / np.sqrt(n)
    vecs = np.column_stack([u1, u2, np.tile([1.0, -1.0], 20) / np.sqrt(n)])
    c = np.vstack([2.0 * u1, 0.5 * u2])
    refs = np.ones((1, n), np.int8)
    st = trace_stats(c, np.array([0.0, 0.1]), vecs, refs)
    assert np.isclose(st["ov1"][0], 1.0) and np.isclose(st["w1"][0], 1.0)
    assert np.isclose(st["d_bks"][0], 0.0) and np.isclose(st["d_bks"][1], 0.5)
    assert np.isclose(st["w2"][1], 1.0) and np.allclose(st["w_top10"], 1.0)
