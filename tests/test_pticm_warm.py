import numpy as np
import pytest

from modules.PT_ICM import simulate_pticm_batch


def _cut(edges, weights, signs):
    return sum(
        w for (a, b), w in zip(edges, weights) if signs[a] != signs[b]
    )


def test_warm_start_is_included_in_best_so_far():
    n = 6
    edges = [(i, (i + 1) % n) for i in range(n)]
    weights = [1.0] * len(edges)
    optimum = np.array([1, -1, 1, -1, 1, -1], dtype=np.int8)

    cuts, signs, _ = simulate_pticm_batch(
        n,
        edges,
        weights,
        num_trials=2,
        num_sweeps=1,
        num_temps=4,
        init_signs=optimum,
        init_mode="single",
        seeds=np.arange(2),
    )

    assert np.all(cuts >= 6)
    assert all(_cut(edges, weights, s) == c for s, c in zip(signs, cuts))


@pytest.mark.parametrize("mode", ["single", "ladder"])
def test_warm_modes_accept_zero_one_spins(mode):
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    cuts, signs, _ = simulate_pticm_batch(
        n,
        edges,
        None,
        num_trials=2,
        num_sweeps=2,
        num_temps=3,
        init_signs=np.array([0, 1, 0, 1], dtype=np.int8),
        init_mode=mode,
        perturb_max=0.4,
        seeds=np.arange(2),
    )
    assert cuts.shape == (2,)
    assert signs.shape == (2, n)


def test_warm_start_validates_shape_and_mode():
    with pytest.raises(ValueError):
        simulate_pticm_batch(
            4,
            [(0, 1)],
            None,
            2,
            num_sweeps=1,
            init_signs=np.ones(3),
        )
    with pytest.raises(ValueError):
        simulate_pticm_batch(
            4,
            [(0, 1)],
            None,
            2,
            num_sweeps=1,
            init_signs=np.ones(4),
            init_mode="all",
        )
