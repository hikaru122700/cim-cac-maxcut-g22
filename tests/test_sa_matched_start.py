"""sa_matched_start の開始解の対応づけ(カット値で重複なしに選ぶ)の確認。"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmarks.sa_matched_start import distance, match_by_cut  # isort: skip


def test_exact_values_are_matched_without_reuse() -> None:
    targets = np.array([10.0, 12.0, 10.0])
    pool = np.array([9.0, 10.0, 12.0, 10.0, 11.0])
    idx = match_by_cut(targets, pool)
    assert len(set(idx.tolist())) == targets.size
    assert np.array_equal(pool[idx], targets)


def test_best_target_gets_the_scarce_good_solution() -> None:
    # 良い値(12)は pool に 1 個しかない。良い target から順に選ぶので 12 の target が取る
    targets = np.array([11.0, 12.0])
    pool = np.array([12.0, 10.0, 11.0])
    idx = match_by_cut(targets, pool)
    assert pool[idx[1]] == 12.0
    assert pool[idx[0]] == 11.0


def test_pool_smaller_than_targets_is_rejected() -> None:
    with pytest.raises(ValueError):
        match_by_cut(np.array([1.0, 2.0]), np.array([1.0]))


def test_distance_treats_global_flip_as_identical() -> None:
    a = np.array([[1, -1, 1, 1]], dtype=np.int8)
    assert distance(a, -a)[0] == 0.0
    assert distance(a, a)[0] == 0.0
    b = np.array([[1, 1, 1, 1]], dtype=np.int8)
    assert distance(a, b)[0] == 0.25
