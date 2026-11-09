"""sa_trajectory の SA が modules/SA.py と同じ結果を返すことの確認。"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

# sa_trajectory は numba の読み込み前にスレッド数を設定するので、modules.SA より先に読む
from scripts.benchmarks.sa_trajectory import compare_final, crossover_sweep, final_stats, make_checkpoints, run_sa_paths  # isort: skip
from modules.SA import simulate_sa_warm  # isort: skip


def small_graph(n: int = 40, seed: int = 0) -> SimpleNamespace:
    rng = np.random.default_rng(seed)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.2]
    return SimpleNamespace(n=n, edges=edges, weights=None)


def test_final_best_matches_repo_sa_exactly() -> None:
    ctx = small_graph()
    init = np.random.default_rng(1).choice(np.array([-1, 1], np.int8), (8, ctx.n))
    seeds = np.arange(8, dtype=np.int64) + 7
    for t_start in (0.5, 2.0):
        repo, _ = simulate_sa_warm(ctx.n, ctx.edges, None, init, 20 * ctx.n, t_start=t_start, t_end=0.001, seeds=seeds)
        _, best, _, final = run_sa_paths(ctx, init, seeds, 20, t_start)
        assert np.array_equal(final, repo)
        assert np.array_equal(best[:, -1], repo.astype(np.float32))


def test_best_so_far_never_decreases_and_starts_at_initial_cut() -> None:
    ctx = small_graph()
    init = np.ones((4, ctx.n), dtype=np.int8)  # 全頂点が同じ側なので cut = 0
    iters, best, cur, _ = run_sa_paths(ctx, init, np.arange(4, dtype=np.int64), 20, 2.0)
    assert iters[0] == 0
    assert np.all(best[:, 0] == 0) and np.all(cur[:, 0] == 0)
    assert np.all(np.diff(best, axis=1) >= 0)
    assert np.all(best >= cur)


def test_checkpoints_end_at_full_budget() -> None:
    cp = make_checkpoints(2000, 5000)
    assert cp[0] == 0 and cp[-1] == 5000 * 2000
    assert np.all(np.diff(cp) > 0)


def test_compare_final_uses_group_means() -> None:
    cim = np.repeat(np.arange(10.0), 4) + 100
    rnd = np.repeat(np.arange(10.0), 4)
    res = compare_final(cim, rnd, 4)
    assert res["mean_diff"] == 100
    assert res["best_of_m_diff"] == 100
    assert res["mean_diff_ci95"][0] > 0
    stats = final_stats(cim, 4, 200.0)
    assert stats["n_runs"] == 40
    assert stats["best_of_m_mean"] == cim.reshape(-1, 4).max(axis=1).mean()
    assert res["mannwhitney_p_group_means"] < 0.01


def test_crossover_returns_first_catch_up_or_none() -> None:
    iters = np.array([0, 10, 20, 30])
    cim = np.array([[5.0, 6.0, 7.0, 8.0]])
    assert crossover_sweep(iters, 10, cim, np.array([[0.0, 1.0, 7.0, 9.0]])) == 2.0
    assert crossover_sweep(iters, 10, cim, np.array([[0.0, 1.0, 2.0, 3.0]])) is None
