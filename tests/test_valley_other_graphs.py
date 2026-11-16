"""実験 E(valley_other_graphs)・F(cim_bks_approach)・G(eigmix_start)の小さな関数のテスト。"""

import numpy as np
import pytest

from scripts.benchmarks import cim_bks_approach as F
from scripts.benchmarks import eigmix_start as G
from scripts.benchmarks import valley_other_graphs as E


def test_overlap_is_flip_invariant_and_bounded():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(50)
    s = np.where(rng.standard_normal((4, 50)) > 0, 1, -1).astype(np.int8)
    ov = E.overlap(s, v)
    assert np.all((ov >= 0) & (ov <= 1))
    assert np.allclose(ov, E.overlap(-s, v))
    assert E.overlap(np.where(v > 0, 1, -1)[None, :], v)[0] == pytest.approx(1.0)


def test_nearest_distance_uses_closest_reference_and_global_flip():
    ref = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 1, -1, 1, -1, 1, -1]], np.int8)
    sol = np.array([[-1, -1, -1, -1, -1, -1, -1, 1]], np.int8)   # 全反転すると ref[0] と 1 スピン違い
    assert E.nearest(sol, ref)[0] == pytest.approx(1 / 8)


def test_valley_fracs_counts_strictly_below_edge():
    d = np.array([0.05, 0.15, 0.25, 0.35])
    fr = E.valley_fracs(d)
    assert fr["0.1"] == 0.25 and fr["0.2"] == 0.5 and fr["0.3"] == 0.75 and fr["0.4"] == 1.0


def test_seed_ranges_do_not_overlap_between_graphs_and_parts():
    class Ctx:
        def __init__(self, name):
            self.name = name
    starts = []
    for g in E.GRAPHS:
        for key in E.SEED_BASE:
            s = E.seeds_for(Ctx(g), key, 1600, 6 * 1_000_000)   # B2 の最後の d まで
            starts.append((int(s[0]), int(s[-1])))
    starts.sort()
    for (a0, a1), (b0, _) in zip(starts, starts[1:]):
        assert a1 < b0


def test_gap_pct():
    assert E.gap_pct(np.array([100.0, 99.0]), 100.0).tolist() == pytest.approx([0.0, 1.0])


def test_eigmix_starts_k1_is_sign_of_u1_up_to_flip():
    rng = np.random.default_rng(1)
    vecs = np.linalg.qr(rng.standard_normal((20, 5)))[0]
    vals = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    s = G.make_starts(vecs, vals, 1, False, 8, rng, 0.0)
    u1 = np.where(vecs[:, 0] > 0, 1, -1)
    assert all(np.array_equal(r, u1) or np.array_equal(r, -u1) for r in s)


def test_eigmix_growth_weights_shrink_lower_modes():
    rng = np.random.default_rng(2)
    vecs = np.linalg.qr(rng.standard_normal((30, 3)))[0]
    vals = np.array([1.0, 0.5, 0.0])
    # τ が大きいと u1 の重みが圧倒的になり、sign(u1)(か全反転)に一致する
    s = G.make_starts(vecs, vals, 3, True, 16, rng, tau=200.0)
    u1 = np.where(vecs[:, 0] > 0, 1, -1)
    assert all(np.array_equal(r, u1) or np.array_equal(r, -u1) for r in s)


def test_group_path_splits_selected_and_others():
    rounds = np.arange(0, 2400, 100)
    d = np.zeros((4, rounds.size))
    d[:2] = 0.1
    d[2:] = 0.4
    sel = np.array([True, True, False, False])
    row = F.group_path(d, rounds, sel)[-1]
    assert row["selected"]["d_median"] == pytest.approx(0.1)
    assert row["selected"]["in_valley"] == 1.0
    assert row["others"]["in_valley"] == 0.0
