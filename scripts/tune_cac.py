"""
CAC ハイパーパラメータチューナー (Method B, Hanyu et al. 2025)

参考論文:
    Hanyu, Katagiri, Mukunoki, Hoshino,
    "Towards Generalized Parameter Tuning in Coherent Ising Machines:
     A Portfolio-Based Approach",
    arXiv:2507.20295 (2025).

Method B の 2 段構成:

  Phase 1 (感度評価):
    各ハイパーパラメータを独立に候補値群で評価し、
    mean_cut の max-min スプレッドで感度を定量化する。

  Phase 2 (逐次最適化):
    感度の高い順にパラメータを 1 つずつ最適化する。
    各パラメータで勝った値を以降の評価に固定し、次のパラメータへ進む。

使い方 (CLI):

    uv run python -m scripts.tune_cac

出力:
    - phase1 / phase2 の各評価結果を標準出力 + results/tune_cac_log.csv に保存
    - 最終的な best config での 100 trial 再評価結果

本実装のスコープ:
    - 対象: alpha, rho, delta, gamma_growth, tau
    - 固定: p (分岐パラメータ), beta0_error (d_0 依存), n_x/n_e/dt_x/dt_e/e_max
      (論文 Supp の GSET 設定から動かさない)
    - ベンチマーク: G22 (固定). 複数インスタンスへの拡張は将来課題
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix


# ============================================================
#  Data types
# ============================================================
@dataclass(frozen=True)
class CACConfig:
    """CAC シミュレータに渡す全パラメータ (不変)。"""

    # 探索対象 (Method B で動かす)
    p: float
    alpha: float
    rho: float
    delta: float
    beta0_error: float
    gamma_growth: float
    tau: float

    # 固定パラメータ (論文 Supp Table 2 の GSET 既定)
    n_x_inner: int = 6
    n_e_inner: int = 4
    dt_x: float = 2.0 ** -6
    dt_e: float = 2.0 ** -4
    e_max: float = 32.0


@dataclass(frozen=True)
class HyperparamGrid:
    """パラメータ名と、既定値に対する乗数タプル。"""

    name: str
    multipliers: tuple[float, ...]

    def candidates(self, default: float) -> tuple[float, ...]:
        """既定値に各乗数を掛けた候補値タプルを返す。"""
        return tuple(m * default for m in self.multipliers)


@dataclass(frozen=True)
class EvalResult:
    """1 つの config に対する評価結果 (不変)。"""

    config: CACConfig
    mean_cut: float
    max_cut: int
    min_cut: int
    std_cut: float
    num_optimal: int
    eval_time_sec: float


# シグネチャ: CACConfig -> EvalResult
EvalFn = Callable[[CACConfig], EvalResult]


# ============================================================
#  Pure logic (テスト対象)
# ============================================================
def apply_override(config: CACConfig, name: str, value: float) -> CACConfig:
    """指定フィールドだけ差し替えた新しい CACConfig を返す (不変)。

    未知のフィールド名が渡された場合は ``dataclasses.replace`` が TypeError を送出する。
    """
    valid_names = {f.name for f in fields(config)}
    if name not in valid_names:
        raise TypeError(f"Unknown CACConfig field: {name!r}")
    return replace(config, **{name: value})


def rank_by_sensitivity(
    results: Mapping[str, Sequence[EvalResult]],
) -> list[str]:
    """各パラメータを mean_cut の max-min スプレッドで降順ソートして返す。

    スプレッドが大きい = そのパラメータ次第で性能が大きく変わる = 高感度 = 優先度高。
    """
    scores: dict[str, float] = {}
    for name, res_list in results.items():
        if not res_list:
            scores[name] = 0.0
            continue
        means = [r.mean_cut for r in res_list]
        scores[name] = max(means) - min(means)
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


def sensitivity_phase(
    base_config: CACConfig,
    grids: Sequence[HyperparamGrid],
    eval_fn: EvalFn,
) -> tuple[list[str], dict[str, list[EvalResult]]]:
    """Phase 1: 各パラメータを独立に評価し、優先度順を返す。

    各グリッドについて:
      - base_config を複製し、対象パラメータのみ候補値で上書き
      - eval_fn で評価
      - 結果を辞書に蓄積
    最後に rank_by_sensitivity で優先度順を決定する。
    """
    results: dict[str, list[EvalResult]] = {}
    for grid in grids:
        default_value = getattr(base_config, grid.name)
        per_param: list[EvalResult] = []
        for value in grid.candidates(default_value):
            cfg = apply_override(base_config, grid.name, value)
            per_param.append(eval_fn(cfg))
        results[grid.name] = per_param
    priority = rank_by_sensitivity(results)
    return priority, results


def optimization_phase(
    base_config: CACConfig,
    grids: Mapping[str, HyperparamGrid],
    priority: Sequence[str],
    eval_fn: EvalFn,
) -> tuple[CACConfig, list[EvalResult]]:
    """Phase 2: 優先度順にパラメータを 1 つずつ最適化。

    各パラメータについて、現時点の current config から候補値を生成し
    (既にロック済みのパラメータはそのまま使われる)、mean_cut 最大の
    候補で config を更新してから次のパラメータに進む。

    Returns:
        (best_config, history): すべての評価結果を時系列で並べた履歴も返す。
    """
    current = base_config
    history: list[EvalResult] = []
    for name in priority:
        if name not in grids:
            raise KeyError(f"No grid defined for priority param: {name!r}")
        grid = grids[name]
        default_value = getattr(current, name)
        per_param: list[EvalResult] = []
        for value in grid.candidates(default_value):
            cfg = apply_override(current, name, value)
            result = eval_fn(cfg)
            per_param.append(result)
            history.append(result)
        if not per_param:
            # 候補が空なら何もせず次へ
            continue
        best = max(per_param, key=lambda r: r.mean_cut)
        current = best.config
    return current, history


# ============================================================
#  Eval function builder (simulate_cac_batch を呼ぶ)
# ============================================================
def build_eval_fn(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_outer_steps: int,
    num_trials: int,
    seed_base: int,
    target_cut: int = 13359,
) -> EvalFn:
    """simulate_cac_batch を呼ぶ EvalFn を返す。

    重い import (numba など) をテストコレクション時に走らせないため
    遅延 import する。
    """
    # 遅延 import: テストから build_eval_fn を呼ばない限り numba は起動しない
    from CAC import simulate_cac_batch

    def eval_fn(config: CACConfig) -> EvalResult:
        seeds = np.array(
            [seed_base + i for i in range(num_trials)], dtype=np.int64
        )
        t0 = time.time()
        cuts, _ = simulate_cac_batch(
            n=n,
            J=J,
            edges=edges,
            num_outer_steps=num_outer_steps,
            num_trials=num_trials,
            p=config.p,
            alpha=config.alpha,
            rho=config.rho,
            delta=config.delta,
            beta0_error=config.beta0_error,
            gamma_growth=config.gamma_growth,
            tau=config.tau,
            n_x_inner=config.n_x_inner,
            n_e_inner=config.n_e_inner,
            dt_x=config.dt_x,
            dt_e=config.dt_e,
            e_max=config.e_max,
            seeds=seeds,
        )
        elapsed = time.time() - t0
        return EvalResult(
            config=config,
            mean_cut=float(cuts.mean()),
            max_cut=int(cuts.max()),
            min_cut=int(cuts.min()),
            std_cut=float(cuts.std()),
            num_optimal=int((cuts == target_cut).sum()),
            eval_time_sec=elapsed,
        )

    return eval_fn


# ============================================================
#  CSV logging
# ============================================================
def write_history_csv(
    output_path: Path,
    phase_label: str,
    history: Sequence[EvalResult],
    param_focus: Sequence[str] | None = None,
    mode: str = "a",
) -> None:
    """評価履歴を CSV に追記する。

    Columns: phase, param_focus, alpha, rho, delta, gamma_growth, tau,
             beta0_error, p, mean_cut, max_cut, min_cut, std_cut, num_optimal, eval_sec
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists or mode == "w":
            writer.writerow([
                "phase", "param_focus",
                "alpha", "rho", "delta", "gamma_growth", "tau",
                "beta0_error", "p",
                "mean_cut", "max_cut", "min_cut", "std_cut",
                "num_optimal", "eval_sec",
            ])
        focus_str = ",".join(param_focus) if param_focus else ""
        for r in history:
            c = r.config
            writer.writerow([
                phase_label, focus_str,
                f"{c.alpha:.6g}", f"{c.rho:.6g}", f"{c.delta:.6g}",
                f"{c.gamma_growth:.6g}", f"{c.tau:.6g}",
                f"{c.beta0_error:.6g}", f"{c.p:.6g}",
                f"{r.mean_cut:.4f}", r.max_cut, r.min_cut,
                f"{r.std_cut:.4f}", r.num_optimal,
                f"{r.eval_time_sec:.2f}",
            ])


# ============================================================
#  Default search grids
# ============================================================
def default_grids() -> dict[str, HyperparamGrid]:
    """5 パラメータの既定探索グリッド (既定値に対する乗数)。

    Method B は感度分析の結果で順番が決まるので、各グリッドは
    対称な乗数 (0.5, 0.75, 1.0, 1.5, 2.0) を採用。中心 1.0 が論文既定。
    """
    mult = (0.5, 0.75, 1.0, 1.5, 2.0)
    return {
        "alpha":        HyperparamGrid("alpha", mult),
        "rho":          HyperparamGrid("rho", mult),
        "delta":        HyperparamGrid("delta", mult),
        "gamma_growth": HyperparamGrid("gamma_growth", mult),
        "tau":          HyperparamGrid("tau", mult),
    }


# ============================================================
#  CLI main
# ============================================================
def _build_base_config_from_graph(n: int, J: csr_matrix) -> CACConfig:
    """compute_gset_parameters から既定 CACConfig を構築する。"""
    # 遅延 import
    from CAC import compute_gset_parameters

    g = compute_gset_parameters(J, n)
    return CACConfig(
        p=g["p"],
        alpha=g["alpha"],
        rho=g["rho"],
        delta=g["delta"],
        beta0_error=g["beta0_error"],
        gamma_growth=g["gamma_growth"],
        tau=g["tau"],
        n_x_inner=g["n_x_inner"],
        n_e_inner=g["n_e_inner"],
        dt_x=g["dt_x"],
        dt_e=g["dt_e"],
        e_max=g["e_max"],
    )


def _print_config(label: str, cfg: CACConfig) -> None:
    print(f"  [{label}] " + ", ".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in asdict(cfg).items()
    ))


def main(
    graph_path: str = "input/G22.txt",
    screen_outer_steps: int = 20000,
    screen_trials: int = 20,
    final_outer_steps: int = 50000,
    final_trials: int = 100,
    seed_base: int = 0,
    output_csv: str = "results/tune_cac_log.csv",
) -> CACConfig:
    """Method B をエンドツーエンドで実行し、最終的な best config を返す。

    既定では G22, 軽量評価 20 trial × 20000 steps でスクリーニング、
    最後に full budget (100 trial × 50000 steps) で再評価する。
    """
    # 遅延 import (テストから main を呼ばない限り numba は起動しない)
    from CIM import build_coupling_matrix, load_graph

    print("=" * 60)
    print("CAC Method B hyperparameter tuner (Hanyu et al. 2025)")
    print("=" * 60)

    # ---- グラフ読み込み ----
    n, k_edges, _adj, edges = load_graph(graph_path)
    print(f"Graph: {graph_path}  N={n}  K={k_edges}")
    J = build_coupling_matrix(n, edges, -1.0)

    # ---- base config ----
    base = _build_base_config_from_graph(n, J)
    print("\nBase config (GSET defaults):")
    _print_config("base", base)

    # ---- 軽量評価用の eval_fn ----
    screen_eval = build_eval_fn(
        n=n,
        J=J,
        edges=edges,
        num_outer_steps=screen_outer_steps,
        num_trials=screen_trials,
        seed_base=seed_base,
    )

    grids = default_grids()

    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # 新規作成 (既存は上書き)
    if csv_path.exists():
        csv_path.unlink()

    # ============================================================
    #  Phase 1: 感度評価
    # ============================================================
    print("\n" + "-" * 60)
    print("Phase 1: sensitivity assessment")
    print(f"  budget: {screen_trials} trials x {screen_outer_steps} steps per eval")
    print(f"  grids: {list(grids.keys())}")
    print("-" * 60)
    t_phase1 = time.time()
    priority, phase1_results = sensitivity_phase(
        base, list(grids.values()), screen_eval
    )
    t_phase1_elapsed = time.time() - t_phase1
    print(f"\nPhase 1 done in {t_phase1_elapsed:.1f} sec")
    print("Sensitivity (max-min spread of mean_cut):")
    for name in priority:
        means = [r.mean_cut for r in phase1_results[name]]
        spread = max(means) - min(means)
        print(f"  {name:15s}  spread = {spread:.2f}  "
              f"(min={min(means):.1f}  max={max(means):.1f})")
    print(f"\nPriority order: {priority}")

    # CSV にログ
    for name, res_list in phase1_results.items():
        write_history_csv(csv_path, "phase1", res_list, param_focus=[name])

    # ============================================================
    #  Phase 2: 逐次最適化
    # ============================================================
    print("\n" + "-" * 60)
    print("Phase 2: sequential optimization (priority order)")
    print("-" * 60)
    t_phase2 = time.time()
    best_cfg, phase2_history = optimization_phase(
        base, grids, priority, screen_eval
    )
    t_phase2_elapsed = time.time() - t_phase2
    print(f"\nPhase 2 done in {t_phase2_elapsed:.1f} sec")
    print("Best config after phase 2:")
    _print_config("best", best_cfg)

    # CSV にログ
    write_history_csv(csv_path, "phase2", phase2_history, param_focus=priority)

    # ============================================================
    #  Final: full budget で再評価
    # ============================================================
    print("\n" + "-" * 60)
    print("Final: full-budget re-evaluation of best config")
    print(f"  budget: {final_trials} trials x {final_outer_steps} steps")
    print("-" * 60)
    final_eval = build_eval_fn(
        n=n,
        J=J,
        edges=edges,
        num_outer_steps=final_outer_steps,
        num_trials=final_trials,
        seed_base=seed_base,
    )
    t_final = time.time()
    baseline_result = final_eval(base)
    tuned_result = final_eval(best_cfg)
    t_final_elapsed = time.time() - t_final

    print(f"\nFinal evaluation done in {t_final_elapsed:.1f} sec")
    print(f"Baseline (GSET defaults):")
    print(f"  mean={baseline_result.mean_cut:.2f}  "
          f"max={baseline_result.max_cut}  "
          f"optimal_hits={baseline_result.num_optimal}/{final_trials}")
    print(f"Tuned (Method B):")
    print(f"  mean={tuned_result.mean_cut:.2f}  "
          f"max={tuned_result.max_cut}  "
          f"optimal_hits={tuned_result.num_optimal}/{final_trials}")

    # CSV にログ (final はラベル別)
    write_history_csv(csv_path, "final_baseline", [baseline_result])
    write_history_csv(csv_path, "final_tuned", [tuned_result])

    delta_mean = tuned_result.mean_cut - baseline_result.mean_cut
    print(f"\nΔ mean_cut (tuned − baseline) = {delta_mean:+.2f}")
    print(f"Log saved to: {csv_path}")

    return best_cfg


if __name__ == "__main__":
    main()
