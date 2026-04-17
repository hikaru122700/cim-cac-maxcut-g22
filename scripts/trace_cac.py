"""
CAC 1-trial トレーサー(純粋 Python + numpy ベクトル化)。

JIT バッチ実装 (CAC._simulate_cac_batch) と同じダイナミクスを、
途中状態を記録できるように numpy で書き下したもの。1 trial 専用。

目的: スナップショットを記録して可視化 (scripts/visualize) に渡すこと。

既存 JIT コード (CAC.py) には一切手を加えない。
内部ループが単純なポイントワイズ更新 (x[i] だけが i-index 依存) なので、
JIT の逐次 for ループと numpy ベクトル化は数学的に同一。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix

from scripts.visualize import SpinFrame, TrajectorySnapshot


@dataclass(frozen=True)
class TraceResult:
    """トレース 1 trial の結果。"""

    final_cut: int
    snapshots: tuple[TrajectorySnapshot, ...]
    spin_frames: tuple[SpinFrame, ...]
    wall_time_sec: float


def trace_cac_single_trial(
    n: int,
    J: csr_matrix,
    edges: np.ndarray,
    num_outer_steps: int,
    config: Mapping[str, Any],
    seed: int = 0,
    snapshot_interval: int = 50,
    spin_frame_interval: int = 250,
) -> TraceResult:
    """CAC を 1 trial だけ純粋 Python で走らせ、スナップショットを記録する。

    Args:
        n: スピン数
        J: 結合行列 (scipy.sparse.csr_matrix)
        edges: shape=(K, 2), 各行が [i, j] の辺リスト
        num_outer_steps: 外ループ反復数
        config: CAC ハイパーパラメータ辞書 (CAC.compute_gset_parameters の出力)
        seed: RNG シード
        snapshot_interval: 通常のスナップショット周期(ステップ数)。
                           β_inj リセット時と改善時は必ず記録される。
        spin_frame_interval: per-spin 振幅フレームの記録周期。snapshot_interval
                            より粗くしないと HTML が巨大化する。改善時は必ず記録。

    Returns:
        TraceResult(final_cut, snapshots, spin_frames, wall_time_sec)
    """
    # ハイパラ展開
    p = float(config["p"])
    alpha = float(config["alpha"])
    rho = float(config["rho"])
    delta = float(config["delta"])
    beta0_error = float(config["beta0_error"])
    gamma_growth = float(config["gamma_growth"])
    tau = float(config["tau"])
    n_x_inner = int(config["n_x_inner"])
    n_e_inner = int(config["n_e_inner"])
    dt_x = float(config["dt_x"])
    dt_e = float(config["dt_e"])
    e_max = float(config["e_max"])

    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = edges_np[:, 0]
    edge_b = edges_np[:, 1]
    num_edges = edges_np.shape[0]

    # 状態初期化 (JIT 版と同じ)
    rng = np.random.default_rng(seed)
    x = 1e-3 * rng.standard_normal(n)
    e = np.ones(n, dtype=np.float64)

    beta_inj = 0.0
    nu_c = 0
    a_t = alpha
    H_opt = float(num_edges)
    best_cut = 0
    best_signs = np.zeros(n, dtype=bool)

    snapshots: list[TrajectorySnapshot] = []
    spin_frames: list[SpinFrame] = []

    t_start = time.time()

    for nu in range(num_outer_steps):
        # Step 1: snapshot x_prev
        x_prev = x.copy()
        x_prev_sq = x_prev * x_prev

        # Step 2: Jx = J @ x_prev  (scipy sparse matvec)
        Jx = J @ x_prev

        # Step 3: I = beta_inj * e * Jx
        I_inj = beta_inj * e * Jx

        # Step 4: current cut & H (sign(x_prev))
        signs = x_prev > 0.0
        cut = int(np.sum(signs[edge_a] != signs[edge_b]))
        H = float(num_edges - 2 * cut)

        # Step 5: x の内ループ (n_x 回, I 固定)
        for _ in range(n_x_inner):
            dx = (p - 1.0) * x - x * x * x + I_inj
            x = x + dx * dt_x

        # Step 6: e の内ループ (n_e 回, x_prev_sq と a_t 固定)
        for _ in range(n_e_inner):
            de = -beta0_error * (x_prev_sq - a_t) * e
            e = np.clip(e + de * dt_e, -e_max, e_max)

        # Step 7: beta_inj 線形成長
        beta_inj += gamma_growth

        # Step 8: a_t 更新 (次ループの e 更新で使用)
        dH = H - H_opt
        a_t_new = alpha + rho * np.tanh(delta * dH)

        # Step 9: reset 判定 (JIT 版と同じく improvement より先)
        reset = (nu - nu_c) > tau
        if reset:
            nu_c = nu
            beta_inj = 0.0

        # Step 10: 改善判定
        improvement = H < H_opt
        if improvement:
            H_opt = H
            nu_c = nu
            best_cut = cut
            best_signs = signs.copy()

        a_t = a_t_new

        # スナップショット記録判定
        if (
            nu % snapshot_interval == 0
            or nu == num_outer_steps - 1
            or reset
            or improvement
        ):
            snapshots.append(TrajectorySnapshot(
                step=nu,
                cut=cut,
                best_cut=best_cut,
                mean_abs_x=float(np.abs(x).mean()),
                std_abs_x=float(np.abs(x).std()),
                mean_e=float(e.mean()),
                std_e=float(e.std()),
                beta_inj=float(beta_inj),
                a_t=float(a_t),
                num_positive=int(np.sum(x > 0.0)),
                beta_reset=bool(reset),
                improvement=bool(improvement),
            ))

        # AHC 風プレーヤー用 per-spin フレーム (x_prev_sq を用いず x を記録)
        # サイズ節約のため spin_frame_interval (= snapshot_interval の倍数) で
        # 間引きつつ、改善時は必ず記録。
        if (
            nu % max(1, spin_frame_interval) == 0
            or nu == num_outer_steps - 1
            or improvement
        ):
            x_abs_max = float(np.max(np.abs(x))) if x.size else 0.0
            # int8 量子化 (-127..127)。scale = x_abs_max / 127
            if x_abs_max > 0.0:
                scale = x_abs_max / 127.0
                x_q = np.clip(
                    np.round(x / scale), -127, 127
                ).astype(np.int8)
            else:
                scale = 1.0
                x_q = np.zeros(n, dtype=np.int8)
            spin_frames.append(SpinFrame(
                step=nu,
                cut=cut,
                scale=float(scale),
                x_q=tuple(int(v) for v in x_q.tolist()),
            ))

    elapsed = time.time() - t_start
    # best_signs は現在未使用だが、将来的に空間パターン可視化用に保持
    _ = best_signs
    return TraceResult(
        final_cut=best_cut,
        snapshots=tuple(snapshots),
        spin_frames=tuple(spin_frames),
        wall_time_sec=elapsed,
    )
