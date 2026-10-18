"""child_observer's spin trace preserves trajectories and reports actual parents/children."""
import numpy as np
from modules.GA import simulate_ga_batch

EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)]
WEIGHTS = [1., -1., 2., 1., -2., 3., 1.]
KW = dict(n=6, edges=EDGES, weights=WEIGHTS, num_trials=3, pop_size=4,
          max_generations=12, ts_iters=20, cr=5, seeds=np.array([100, 200, 300]),
          return_history=True)


def score(s):
    return sum(w * (s[..., a] != s[..., b]) for (a, b), w in zip(EDGES, WEIGHTS))


def test_child_observer_trace_preserves_results_and_tracks_updates():
    expected = simulate_ga_batch(**KW)
    populations = {}
    events = []

    def on_population(g, p, f):
        populations[g] = (p, f)

    def on_child(g, e):
        prev, prev_fits = populations[g - 1]
        for t in range(KW["num_trials"]):
            np.testing.assert_array_equal(e["parent_spins"][t], prev[t, e["parent_idx"][t]])
        np.testing.assert_array_equal(e["pop_fits_before"], prev_fits)
        parent_fits = score(e["parent_spins"])
        np.testing.assert_array_equal(parent_fits[:, 0], e["parent_a"])
        np.testing.assert_array_equal(parent_fits[:, 1], e["parent_b"])
        np.testing.assert_array_equal(score(e["child_raw_spins"]), e["child_raw"])
        np.testing.assert_array_equal(score(e["child_ts_spins"]), e["child_ts"])
        assert np.all(e["child_ts"] >= e["child_raw"])
        events.append({k: v.copy() for k, v in e.items()})
        # The callback must not be able to mutate the solver's current/future state.
        for value in e.values():
            value[:] = 0

    actual = simulate_ga_batch(**KW, population_observer=on_population, child_observer=on_child)
    for a, b in zip(expected, actual):
        np.testing.assert_array_equal(a, b)
    assert len(events) == KW["max_generations"]
    for g, e in enumerate(events, start=1):
        prev, _ = populations[g-1]
        current, fits = populations[g]
        np.testing.assert_array_equal(fits, e["pop_fits_after"])
        for t in range(KW["num_trials"]):
            if e["accepted"][t]:
                assert np.any(np.all(current[t] == e["child_ts_spins"][t], axis=1))
            else:
                np.testing.assert_array_equal(prev[t], current[t])
