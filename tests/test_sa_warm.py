import numpy as np

from modules.SA import simulate_sa_warm


def test_sa_warm_rng_alignment_is_reproducible():
    n = 6
    edges = [(i, (i + 1) % n) for i in range(n)]
    init = np.tile(np.array([1, -1, 1, -1, 1, -1], dtype=np.int8), (2, 1))
    kwargs = dict(
        n=n,
        edges=edges,
        weights=None,
        init_signs=init,
        num_iters=100,
        t_start=1.0,
        t_end=0.01,
        seeds=np.arange(2),
        align_rng_with_cold=True,
    )
    cuts_a, signs_a = simulate_sa_warm(**kwargs)
    cuts_b, signs_b = simulate_sa_warm(**kwargs)
    np.testing.assert_array_equal(cuts_a, cuts_b)
    np.testing.assert_array_equal(signs_a, signs_b)
