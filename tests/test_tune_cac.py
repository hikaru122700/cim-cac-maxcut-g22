"""
scripts/tune_cac.py の純粋ロジック部分の単体テスト。

重い simulate_cac_batch を呼ばずに、以下の振る舞いを検証する:
  - apply_override : 不変な新しい config を返す
  - HyperparamGrid  : 既定値に対する乗数展開
  - rank_by_sensitivity : mean_cut の max-min スプレッドで優先度付け
  - sensitivity_phase : 各パラメータを独立に評価、優先度順を返す
  - optimization_phase : 優先度順に 1 パラメータずつ逐次最適化
"""

from __future__ import annotations

from typing import Callable

import pytest

from scripts.tune_cac import (
    CACConfig,
    EvalResult,
    HyperparamGrid,
    apply_override,
    optimization_phase,
    rank_by_sensitivity,
    sensitivity_phase,
)


# ============================================================
#  Test fixtures
# ============================================================
def _make_base_config() -> CACConfig:
    """GSET 既定値にざっくり近い base config (数値はテスト用)"""
    return CACConfig(
        p=0.5,
        alpha=3.0,
        rho=1.0,
        delta=1.3e-3,
        beta0_error=0.15,
        gamma_growth=1e-3,
        tau=18000.0,
    )


def _make_result(config: CACConfig, mean_cut: float) -> EvalResult:
    """テスト専用: mean_cut だけ指定できるヘルパー"""
    return EvalResult(
        config=config,
        mean_cut=mean_cut,
        max_cut=int(mean_cut),
        min_cut=int(mean_cut),
        std_cut=0.0,
        num_optimal=0,
        eval_time_sec=0.0,
    )


# ============================================================
#  HyperparamGrid
# ============================================================
class TestHyperparamGrid:
    def test_candidates_scale_default_by_multipliers(self) -> None:
        grid = HyperparamGrid(name="alpha", multipliers=(0.5, 1.0, 2.0))
        assert grid.candidates(3.0) == (1.5, 3.0, 6.0)

    def test_candidates_preserve_order(self) -> None:
        grid = HyperparamGrid(name="rho", multipliers=(2.0, 0.5, 1.0))
        assert grid.candidates(1.0) == (2.0, 0.5, 1.0)

    def test_empty_multipliers_yield_empty_candidates(self) -> None:
        grid = HyperparamGrid(name="tau", multipliers=())
        assert grid.candidates(9000.0) == ()


# ============================================================
#  apply_override
# ============================================================
class TestApplyOverride:
    def test_returns_new_instance_without_mutating_original(self) -> None:
        base = _make_base_config()
        new = apply_override(base, "alpha", 5.0)
        assert new.alpha == 5.0
        assert base.alpha == 3.0  # immutability
        assert new.rho == base.rho  # other fields preserved

    def test_raises_on_unknown_field(self) -> None:
        base = _make_base_config()
        with pytest.raises(TypeError):
            apply_override(base, "nonexistent_field", 1.0)


# ============================================================
#  rank_by_sensitivity
# ============================================================
class TestRankBySensitivity:
    def test_sorts_params_by_max_min_spread_descending(self) -> None:
        cfg = _make_base_config()
        high_var = [
            _make_result(cfg, 100.0),
            _make_result(cfg, 500.0),
            _make_result(cfg, 200.0),
        ]  # spread = 400
        low_var = [
            _make_result(cfg, 300.0),
            _make_result(cfg, 310.0),
            _make_result(cfg, 305.0),
        ]  # spread = 10
        ranking = rank_by_sensitivity({"alpha": high_var, "rho": low_var})
        assert ranking == ["alpha", "rho"]

    def test_single_param_ranks_trivially(self) -> None:
        cfg = _make_base_config()
        ranking = rank_by_sensitivity({"alpha": [_make_result(cfg, 100.0)]})
        assert ranking == ["alpha"]

    def test_three_params_ordered_by_spread(self) -> None:
        cfg = _make_base_config()
        results = {
            "low":  [_make_result(cfg, 100.0), _make_result(cfg, 102.0)],  # 2
            "high": [_make_result(cfg, 100.0), _make_result(cfg, 200.0)],  # 100
            "mid":  [_make_result(cfg, 100.0), _make_result(cfg, 150.0)],  # 50
        }
        ranking = rank_by_sensitivity(results)
        assert ranking == ["high", "mid", "low"]


# ============================================================
#  sensitivity_phase
# ============================================================
class TestSensitivityPhase:
    def test_evaluates_each_candidate_for_each_grid(self) -> None:
        base = _make_base_config()
        grids = [
            HyperparamGrid("alpha", (0.5, 1.0, 2.0)),
            HyperparamGrid("rho", (0.5, 1.0, 2.0)),
        ]
        calls: list[CACConfig] = []

        def fake_eval(cfg: CACConfig) -> EvalResult:
            calls.append(cfg)
            return _make_result(cfg, mean_cut=100.0)

        priority, results = sensitivity_phase(base, grids, fake_eval)

        assert len(calls) == 6  # 2 grids x 3 candidates
        assert "alpha" in results and "rho" in results
        assert len(results["alpha"]) == 3
        assert len(results["rho"]) == 3
        assert set(priority) == {"alpha", "rho"}

    def test_priority_reflects_higher_variance_param(self) -> None:
        base = _make_base_config()
        grids = [
            HyperparamGrid("alpha", (0.5, 1.0, 2.0)),   # 高感度
            HyperparamGrid("rho", (0.5, 1.0, 2.0)),     # 低感度
        ]

        def fake_eval(cfg: CACConfig) -> EvalResult:
            if cfg.alpha != 3.0:
                # alpha を動かすと大きく変化
                return _make_result(cfg, mean_cut=1000.0 * cfg.alpha)
            # rho を動かすと微小変化のみ
            return _make_result(cfg, mean_cut=100.0 + cfg.rho * 0.1)

        priority, _ = sensitivity_phase(base, grids, fake_eval)
        assert priority[0] == "alpha"

    def test_independent_eval_uses_base_config_for_other_params(self) -> None:
        """sensitivity_phase では他パラメータは常に base_config の値で固定されるべき"""
        base = _make_base_config()
        grids = [HyperparamGrid("alpha", (0.5, 1.0, 2.0))]
        seen: list[CACConfig] = []

        def fake_eval(cfg: CACConfig) -> EvalResult:
            seen.append(cfg)
            return _make_result(cfg, mean_cut=0.0)

        sensitivity_phase(base, grids, fake_eval)
        for cfg in seen:
            assert cfg.rho == base.rho
            assert cfg.delta == base.delta
            assert cfg.tau == base.tau


# ============================================================
#  optimization_phase
# ============================================================
class TestOptimizationPhase:
    def test_picks_best_candidate_for_single_param(self) -> None:
        base = _make_base_config()
        grids = {"alpha": HyperparamGrid("alpha", (0.5, 1.0, 2.0))}
        priority = ["alpha"]

        def fake_eval(cfg: CACConfig) -> EvalResult:
            return _make_result(cfg, mean_cut=cfg.alpha * 10.0)

        best, history = optimization_phase(base, grids, priority, fake_eval)
        # 最大 alpha 候補 = 3.0 * 2.0 = 6.0
        assert best.alpha == pytest.approx(6.0)
        assert len(history) == 3

    def test_sequentially_updates_best_before_next_param(self) -> None:
        """2 つのパラメータを順に最適化するとき、2 番目の評価は 1 番目の最適値を使うべき"""
        base = _make_base_config()
        grids = {
            "alpha": HyperparamGrid("alpha", (1.0, 2.0)),  # 候補 3.0, 6.0
            "rho":   HyperparamGrid("rho",   (1.0, 2.0)),  # 候補 1.0, 2.0
        }
        priority = ["alpha", "rho"]

        seen: list[CACConfig] = []

        def fake_eval(cfg: CACConfig) -> EvalResult:
            seen.append(cfg)
            # alpha=6.0, rho=2.0 が最良
            return _make_result(cfg, mean_cut=cfg.alpha * 100 + cfg.rho * 10)

        best, _ = optimization_phase(base, grids, priority, fake_eval)
        assert best.alpha == pytest.approx(6.0)
        assert best.rho == pytest.approx(2.0)

        # rho の評価 (3 番目以降) では alpha が既に 6.0 にロックされているはず
        rho_phase = seen[2:]
        for cfg in rho_phase:
            assert cfg.alpha == pytest.approx(6.0)

    def test_history_records_every_evaluation(self) -> None:
        base = _make_base_config()
        grids = {
            "alpha": HyperparamGrid("alpha", (0.5, 1.0, 2.0)),
            "rho":   HyperparamGrid("rho",   (0.5, 1.0, 2.0)),
        }
        priority = ["alpha", "rho"]

        def fake_eval(cfg: CACConfig) -> EvalResult:
            return _make_result(cfg, mean_cut=1.0)

        _, history = optimization_phase(base, grids, priority, fake_eval)
        assert len(history) == 6  # 2 params x 3 candidates

    def test_missing_grid_for_priority_param_raises(self) -> None:
        base = _make_base_config()
        grids = {"alpha": HyperparamGrid("alpha", (1.0,))}
        priority = ["alpha", "rho"]  # rho のグリッドが無い

        def fake_eval(cfg: CACConfig) -> EvalResult:
            return _make_result(cfg, mean_cut=0.0)

        with pytest.raises(KeyError):
            optimization_phase(base, grids, priority, fake_eval)
