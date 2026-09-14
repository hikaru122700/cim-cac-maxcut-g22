"""Observation must not alter the GA, and supplied populations must be honored."""
import numpy as np
import pytest

from modules.GA import simulate_ga_batch
from scripts.benchmarks.ga_diversity_trajectory import population_metrics

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
WEIGHTS = [1., -1., 2., 1., -2., 3., 1.]
KW = dict(n=6, edges=EDGES, weights=WEIGHTS, num_trials=2, pop_size=4,
          max_generations=5, ts_iters=20, cr=5, seeds=np.array([100, 200]), return_history=True)


def test_observer_is_noninvasive_and_scores_are_correct():
    expected = simulate_ga_batch(**KW)
    generations = []

    def observe(g, pops, fits):
        generations.append(g)
        independent = sum(w * (pops[:, :, a] != pops[:, :, b])
                          for (a, b), w in zip(EDGES, WEIGHTS))
        np.testing.assert_allclose(independent, fits)
        pops[:] = 0  # copies must protect internal solver state
        fits[:] = -999

    actual = simulate_ga_batch(**KW, population_observer=observe)
    assert generations == [-1, 0, 1, 2, 3, 4, 5]
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_full_population_matches_default_and_rng_stream():
    initial = np.stack([(np.random.default_rng(s).random((4, 6)) < .5).astype(np.int8)
                        for s in KW["seeds"]])
    expected = simulate_ga_batch(**KW)
    actual = simulate_ga_batch(**KW, init_population=initial)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)


def test_supplied_population_is_observed_before_ts():
    initial = np.ones((2, 4, 6), dtype=np.int8)
    initial[:, :, 0] = -1
    seen = []
    simulate_ga_batch(**KW, init_population=initial,
                      population_observer=lambda g, p, f: seen.append(p) if g == -1 else None)
    np.testing.assert_array_equal(seen[0], initial > 0)
    with pytest.raises(ValueError):
        simulate_ga_batch(**KW, init_population=initial, init_signs=initial[:, 0])
    with pytest.raises(ValueError):
        simulate_ga_batch(**KW, init_population=initial[0])


def test_metrics_respect_complements_and_run_boundaries():
    pops = np.array([[[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 1, 1]],
                     [[0, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0]]], dtype=np.int8)
    metrics = population_metrics(pops)
    np.testing.assert_allclose(metrics[0], [1/3, 1/6, 0, 2])
    np.testing.assert_array_equal(metrics[1], [0, 0, 0, 1])
