"""
CAC (Chaotic Amplitude Control) for MAX-CUT G22.

論文: Leleu, Khoyratee, Levi, Hamerly, Kohno, Aihara,
  "Scaling advantage of chaotic amplitude control for high-performance
   combinatorial optimization",
  Communications Physics 4, 266 (2021).
  (arXiv:2009.04084 の Supplementary Materials S2.3, S2.8 に疑似コードとパラメータあり)

========================================================================
【CAC の要点】
通常の CIM は対称飽和でローカル最適にトラップされやすい。CAC は以下で脱出する:

  (1) パルスごとのエラー変数 e_i で結合強度を動的制御
  (2) 目標振幅 a(t) を現在の解品質 ΔH に応じて動的変調
  (3) 結合ランプ β_inj(t) を 0 から線形増加 → τ 経過で 0 リセット
      (カオス的な周期的再立ち上げで準最適解を揺さぶる)

========================================================================
【疑似コード (Supplementary S2.3 を整理)】

  x_prev ← x                              # 外ループ先頭でスナップショット
  Jx ← Ω @ x_prev                         # sparse matvec (1回/外ループ)
  I ← β_inj · e · Jx                       # 注入項 (時変 β_inj を使用)
  σ ← sign(x_prev); H ← K − 2·cut         # Ising エネルギー (= K − 2·cut)

  for _ in range(n_x):                    # x の内ループ n_x 回
      Δx ← (−1 + p) · x − x³ + I
      x  ← x + Δx · dt_x

  for _ in range(n_e):                    # e の内ループ n_e 回(x_prev² と a_t 固定)
      Δe ← −β₀ · (x_prev² − a_t) · e
      e  ← e + Δe · dt_e
      e  ← clip(e, −e_max, e_max)         # Python 実装固有 (e_max = 32)

  β_inj ← β_inj + γ · dt_x                # 線形成長
  a_t   ← α + ρ · tanh(δ · (H − H_opt))   # 目標振幅変調

  if ν − ν_c > τ / dt_x:                  # 長時間未改善 → β_inj リセット
      β_inj ← 0; ν_c ← ν
  if H < H_opt:                           # 改善 → 時計再スタート
      H_opt ← H; ν_c ← ν; 最良解を保存

========================================================================
【変数の区別(紛らわしいので整理)】

  β₀ (beta0_error)  : 定数、誤差項の rate。GSET では 3/d₀。
                      論文本文 Eq.(2) の ξ に対応。
  β_inj (beta_inj)  : 時変、注入項の結合スケール。初期 0 から γ で増加、
                      τ 経過で 0 に戻る。本文中の Γ によるランプ。
  γ (gamma_growth) : β_inj の線形成長率。GSET では 2/N。
  τ (tau)           : β_inj のリセット時間窓(連続時間単位)。GSET では 9N。
  p                 : 分岐パラメータ(定数)。GSET では 1 − 400 · d₁⁻²·⁵。
  α (alpha)         : 目標振幅² の中心値。GSET では 3.0。
  ρ (rho)           : 目標振幅変調の深さ。GSET では 1.0。
  δ (delta)         : ΔH への感度。GSET では 2.6/N。
  d₀                : mean_i Σ_j |Ω_ij| = 平均重み付き次数 (G22 では ≈ 20)
  d₁                : max(d₀, 10)

【論文 Table 1: G22 での CAC 性能 (FPGA)】
  C_opt = 13359, C_CAC = 13359 (p₀ = 0.11, 100 run 中 11% が最適到達)
  TTS_CAC = 157 sec, <t_CAC> = 3.33 sec (FPGA, 50 MHz)
"""

import time

import numpy as np
import wandb
from numba import njit, prange
from scipy.sparse import csr_matrix

from .CIM import build_coupling_matrix, load_graph
from .verify import run_all_checks


# ============================================================
#  Numba JIT + 並列化された CAC 本体
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cac_batch(
    n: int,
    num_outer_steps: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    p: float,
    alpha: float,
    rho: float,
    delta: float,
    beta0_error: float,      # 論文 β₀ = 3/d₀ (GSET): 誤差項の rate
    gamma_growth: float,     # 論文 γ = 2/N (GSET): β_inj の線形成長率
    tau: float,              # 論文 τ = 9N (GSET): β_inj リセット時間窓
    n_x_inner: int,          # 論文 n_x = 6: x の内ループ回数
    n_e_inner: int,          # 論文 n_e = 4: e の内ループ回数
    dt_x: float,             # 論文 dt_x = 2⁻⁶
    dt_e: float,             # 論文 dt_e = 2⁻⁴
    e_max: float,            # 論文 e_max = 32 (Python 実装)
    seeds: np.ndarray,
):
    """CAC (Leleu 2021 GSET-用パラメータ) を num_trials 分並列実行。"""
    best_cuts_out = np.zeros(num_trials, dtype=np.int64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)

    num_edges = edge_a.shape[0]
    # β_inj リセット判定: nu - nu_c > tau (tau は外ループ回数として直接指定)
    # 論文の "τ = 9N" を連続時間と解釈すると β が暴走するため、外ループ回数と解釈
    tau_iters = tau

    # ---- trial ごとに並列実行 ----
    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        # 状態ベクトル (trial ごと独立)
        x = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x[i] = 1e-3 * np.random.standard_normal()
        x_prev = np.zeros(n, dtype=np.float64)
        x_prev_sq = np.zeros(n, dtype=np.float64)
        e = np.ones(n, dtype=np.float64)
        Jx = np.zeros(n, dtype=np.float64)
        I_inj = np.zeros(n, dtype=np.float64)

        # CAC 制御変数
        beta_inj = 0.0       # 時変結合スケール
        nu_c = 0             # 最後の H_opt 更新 or リセット時刻 (外ループ index)
        a_t = alpha          # 初期目標振幅² (ΔH=0 → a=α)

        # ベスト追跡
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = 0
        # H = K − 2·cut. cut=0 のとき H = K (最大), best なら小さい
        H_opt = float(num_edges)  # 初期値: 最悪(全て不成立)

        # ---- 外ループ ----
        for nu in range(num_outer_steps):
            # Step 1: x をスナップショット (内ループで x が変わっても x_prev² は固定)
            for i in range(n):
                x_prev[i] = x[i]
                x_prev_sq[i] = x[i] * x[i]

            # Step 2: Sparse matvec Jx = Ω @ x_prev (CSR 手書きループ)
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * x_prev[J_indices[jj]]
                Jx[i] = acc

            # Step 3: 注入項 I = β_inj · e · Jx (内ループ中は固定)
            for i in range(n):
                I_inj[i] = beta_inj * e[i] * Jx[i]

            # Step 4: 現在の Ising エネルギー H (cut 換算)
            cut = 0
            for idx in range(num_edges):
                if (x_prev[edge_a[idx]] > 0.0) != (x_prev[edge_b[idx]] > 0.0):
                    cut += 1
            H = float(num_edges - 2 * cut)

            # Step 5: x の内ループ (n_x 回、I 固定)
            for _ in range(n_x_inner):
                for i in range(n):
                    xi = x[i]
                    dx = (p - 1.0) * xi - xi * xi * xi + I_inj[i]
                    x[i] = xi + dx * dt_x

            # Step 6: e の内ループ (n_e 回、x_prev² と a_t は前の外ループの値で固定)
            for _ in range(n_e_inner):
                for i in range(n):
                    ei = e[i]
                    de = -beta0_error * (x_prev_sq[i] - a_t) * ei
                    new_ei = ei + de * dt_e
                    if new_ei > e_max:
                        new_ei = e_max
                    elif new_ei < -e_max:
                        new_ei = -e_max
                    e[i] = new_ei

            # Step 7: β_inj の線形成長
            # 疑似コードの "β ← β + λ dt" は外ループ単位時間 dt=1 として運用するのが
            # 著者参考 Python 実装 (arXiv Supp) の慣例 → dt_x ではなく 1 を使う
            beta_inj += gamma_growth

            # Step 8: 目標振幅 a_t を更新 (次回ループの e 更新で使用)
            dH = H - H_opt  # ≥ 0 (H_opt は最小)
            a_t = alpha + rho * np.tanh(delta * dH)

            # Step 9: 長時間未改善 → β_inj リセット
            if (nu - nu_c) > tau_iters:
                nu_c = nu
                beta_inj = 0.0

            # Step 10: 最良解更新 → 時計リスタート
            if H < H_opt:
                H_opt = H
                nu_c = nu
                best_cut = cut
                for i in range(n):
                    best_signs[i] = x_prev[i] > 0.0

        # trial 終了、結果を出力
        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


def simulate_cac_batch(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_outer_steps: int,
    num_trials: int,
    p: float,
    alpha: float,
    rho: float,
    delta: float,
    beta0_error: float,
    gamma_growth: float,
    tau: float,
    n_x_inner: int,
    n_e_inner: int,
    dt_x: float,
    dt_e: float,
    e_max: float,
    seeds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """CAC を num_trials 分、並列実行する公開ラッパー。"""
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    return _simulate_cac_batch(
        n,
        num_outer_steps,
        num_trials,
        J.data,
        J.indices,
        J.indptr,
        edge_a,
        edge_b,
        float(p),
        float(alpha),
        float(rho),
        float(delta),
        float(beta0_error),
        float(gamma_growth),
        float(tau),
        int(n_x_inner),
        int(n_e_inner),
        float(dt_x),
        float(dt_e),
        float(e_max),
        seeds_arr,
    )


def compute_gset_parameters(J: csr_matrix, n: int) -> dict:
    """論文 Supp Table 2 の GSET 用パラメータを、問題インスタンスから計算する。

    d_0 = mean_i Σ_j |ω_ij|  (平均重み付き次数)
    d_1 = max(d_0, 10)
    p     = 1 − 400 · d_1^(−2.5)
    β₀   = 3 / d_0                (誤差項の rate)
    γ    = 2 / N                  (β_inj の成長率)
    τ    = 9 · N                  (β_inj リセット時間窓)
    δ    = 2.6 / N                (ΔH 感度)
    α    = 3.0                    (目標振幅² 中心)
    ρ    = 1.0                    (振幅変調の深さ)
    """
    # 各行の絶対値和(= |ω_ij| の行和)の平均
    row_abs_sums = np.abs(J).sum(axis=1)
    d_0 = float(np.asarray(row_abs_sums).mean())
    d_1 = max(d_0, 10.0)

    return {
        "d_0": d_0,
        "d_1": d_1,
        "p": 1.0 - 400.0 * (d_1 ** -2.5),
        "alpha": 3.0,
        "rho": 1.0,
        "delta": 2.6 / n,
        "beta0_error": 3.0 / d_0,
        "gamma_growth": 2.0 / n,
        "tau": 9.0 * n,
        "n_x_inner": 6,
        "n_e_inner": 4,
        "dt_x": 2.0 ** -6,
        "dt_e": 2.0 ** -4,
        "e_max": 32.0,
    }


def main():
    """CAC を 100 trial 並列実行して G22 の結果を集計。"""
    # ==== ハイパーパラメータ ====
    # 論文 Supp Table 2 の GSET 用を compute_gset_parameters で自動計算。
    # num_outer_steps は CPU で現実的な時間に収まる値に設定(論文 FPGA は ~10⁶ 相当)。
    base_config = {
        "coupling": -1.0,           # J_ij for edges (ω_ij ∈ {−1, 0})
        "num_outer_steps": 50000,   # 外ループ回数 (調整可)
        "num_trials": 100,
        "seed_base": 0,
    }

    # ==== グラフ読み込み ====
    filepath = "input/G22.txt"
    n, k_edges, adj, edges = load_graph(filepath)
    print(f"N={n}, K={k_edges}")

    # 結合行列
    J = build_coupling_matrix(n, edges, base_config["coupling"])

    # 問題インスタンスから GSET パラメータを計算
    gset_params = compute_gset_parameters(J, n)
    print(f"d_0 = {gset_params['d_0']:.2f}, d_1 = {gset_params['d_1']:.2f}")
    print(f"p = {gset_params['p']:.4f}")
    print(f"beta0 (3/d_0) = {gset_params['beta0_error']:.4f}")
    print(f"gamma (2/N)   = {gset_params['gamma_growth']:.6f}")
    print(f"tau (9N)      = {gset_params['tau']:.1f}")
    print(f"delta (2.6/N) = {gset_params['delta']:.6f}")

    # wandb 用の full config
    config = {**base_config, **gset_params}
    wandb.init(project="cim-cac", config=config)
    cfg = wandb.config

    # シード配列
    seeds = np.array(
        [cfg.seed_base + i for i in range(cfg.num_trials)], dtype=np.int64
    )

    # ==== CAC 並列実行 ====
    print(f"Running CAC: {cfg.num_trials} trials in parallel (Numba prange)...")
    t_start = time.time()
    best_cuts_arr, best_signs_arr = simulate_cac_batch(
        n=n,
        J=J,
        edges=edges,
        num_outer_steps=cfg.num_outer_steps,
        num_trials=cfg.num_trials,
        p=cfg.p,
        alpha=cfg.alpha,
        rho=cfg.rho,
        delta=cfg.delta,
        beta0_error=cfg.beta0_error,
        gamma_growth=cfg.gamma_growth,
        tau=cfg.tau,
        n_x_inner=cfg.n_x_inner,
        n_e_inner=cfg.n_e_inner,
        dt_x=cfg.dt_x,
        dt_e=cfg.dt_e,
        e_max=cfg.e_max,
        seeds=seeds,
    )
    t_elapsed = time.time() - t_start
    print(
        f"Finished in {t_elapsed:.2f} sec "
        f"({t_elapsed / cfg.num_trials * 1000:.1f} ms/trial)"
    )

    # ==== trial ごとの結果を wandb に逐次ログ ====
    running_mean = 0.0
    running_max = 0
    cum_sum = 0
    for trial in range(cfg.num_trials):
        cut = int(best_cuts_arr[trial])
        cum_sum += cut
        running_mean = cum_sum / (trial + 1)
        running_max = max(running_max, cut)
        wandb.log({
            "trial": trial + 1,
            "seed": int(seeds[trial]),
            "trial_cut": cut,
            "running_mean": running_mean,
            "running_max": running_max,
        })

    # ==== 統計 ====
    cuts_arr = best_cuts_arr.astype(np.int64)
    mean_cut = float(cuts_arr.mean())
    std_cut = float(cuts_arr.std())
    max_cut = int(cuts_arr.max())
    min_cut = int(cuts_arr.min())
    median_cut = float(np.median(cuts_arr))

    num_optimal = int((cuts_arr == 13359).sum())
    success_rate = num_optimal / cfg.num_trials

    best_overall_trial = int(np.argmax(cuts_arr))
    best_overall_cut = int(cuts_arr[best_overall_trial])
    best_overall_x: list[int] = best_signs_arr[best_overall_trial].astype(np.int64).tolist()

    print("=" * 60)
    print(f"Results over {cfg.num_trials} trials:")
    print(f"  mean  = {mean_cut:.2f}  (CIM paper: 13275)")
    print(f"  best  = {max_cut}      (known best / paper CAC: 13359)")
    print(f"  worst = {min_cut}")
    print(f"  std   = {std_cut:.2f}")
    print(f"  median= {median_cut:.1f}")
    print(f"  optimal (13359) hits = {num_optimal}/{cfg.num_trials} (p_0 = {success_rate:.3f})  (paper: 0.11)")
    print(f"  best trial: #{best_overall_trial + 1} (seed={cfg.seed_base + best_overall_trial})")
    print("=" * 60)

    # ==== 検算 ====
    run_all_checks(best_overall_x, n, k_edges, adj, edges, best_overall_cut)

    # ==== wandb summary ====
    wandb.summary["mean_cut"] = mean_cut
    wandb.summary["best_cut"] = max_cut
    wandb.summary["worst_cut"] = min_cut
    wandb.summary["std_cut"] = std_cut
    wandb.summary["median_cut"] = median_cut
    wandb.summary["optimal_hits"] = num_optimal
    wandb.summary["success_rate_p0"] = success_rate
    wandb.summary["best_trial_idx"] = best_overall_trial
    wandb.summary["ratio_to_known_best"] = max_cut / 13359
    wandb.summary["known_best"] = 13359
    wandb.summary["paper_cac_p0"] = 0.11

    # ヒストグラム
    wandb.log({"cut_histogram": wandb.Histogram(cuts_arr.tolist())})

    # 全 trial テーブル
    trial_table = wandb.Table(
        columns=["trial", "seed", "cut"],
        data=[
            [i + 1, int(seeds[i]), int(cuts_arr[i])]
            for i in range(cfg.num_trials)
        ],
    )
    wandb.log({"trials_table": trial_table})

    wandb.finish()


if __name__ == "__main__":
    main()
