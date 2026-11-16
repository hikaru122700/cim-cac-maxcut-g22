import numpy as np
import scipy.sparse as sp

from scripts.benchmarks.eigvec_valley import eig_stats, overlap, top_eigvecs


def test_overlap_is_flip_symmetric_and_bounded():
    rng = np.random.default_rng(0)
    s = rng.choice([-1, 1], size=(5, 40)).astype(np.int8)
    v = rng.normal(size=40)
    a, b = overlap(s, v), overlap(-s, v)
    assert np.allclose(a, b)
    assert np.all((a >= 0) & (a <= 1))
    assert np.isclose(overlap(np.sign(v)[None, :], v)[0], 1.0)


def test_top_eigvecs_order_and_sign():
    J = sp.diags([3.0, 1.0, 2.0, -1.0, 0.5])
    vals, vecs = top_eigvecs(J, k=2)
    assert np.allclose(vals, [3.0, 2.0])
    assert np.allclose(np.abs(vecs[:, 0]), [1, 0, 0, 0, 0])
    assert np.all(vecs[0] >= 0)


def test_eig_stats_splits_valleys():
    n = 50
    u1 = np.ones(n)
    u2 = np.r_[np.ones(25), -np.ones(25)]
    vecs = np.column_stack([u1, u2]) / np.sqrt(n)
    near = np.tile(u1, (6, 1))
    far = np.tile(u2, (6, 1))
    S = np.vstack([near, far])
    d = np.r_[np.zeros(6), np.full(6, 0.45)]
    out = eig_stats(S, vecs, d)
    assert out["near"]["E_mean"] > 0 > out["far"]["E_mean"]
    assert out["E_near_vs_far"]["auc"] == 1.0
    assert np.isclose(out["frac_far"], 0.5)
