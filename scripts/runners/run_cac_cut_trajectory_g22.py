"""CAC の outer step ごとのカット推移(100 trial 中の最大)を G22 で記録・作図する。

modules/CAC.py の `_simulate_cac_batch` を正確にミラーし、唯一
  各 outer step 終了時点の running best_cut を cut_hist[trial, nu] に記録する
点だけを足した計装カーネルを内蔵する(本体実装には手を入れない)。

各 step で 100 trial の running best_cut の **最大値** を取り、その時間発展を描く。
パラメータ・seed・予算は run_pumpbench_real_cim_g22.py の CAC 条件と同一
(compute_gset_parameters, num_outer_steps=50000, 100 trial, seeds=arange)。

実行(プロジェクトルートから):
  C:\\Python313\\python.exe scripts/runners/run_cac_cut_trajectory_g22.py
  (または .venv の python)
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters  # noqa: E402

EXPERIMENT_KIND = "cac_cut_trajectory_g22"
KNOWN_BEST = 13359
NUM_OUTER_STEPS = 50000
TRIALS = 100


@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cac_traj(
    n, num_outer_steps, num_trials,
    J_data, J_indices, J_indptr, edge_a, edge_b,
    p, alpha, rho, delta, beta0_error, gamma_growth, tau,
    n_x_inner, n_e_inner, dt_x, dt_e, e_max, seeds,
    cut_hist,           # (num_trials, num_outer_steps) int32: 各 step の running best_cut
):
    num_edges = edge_a.shape[0]
    tau_iters = tau
    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])
        x = np.zeros(n, dtype=np.float64)
        for i in range(n):
            x[i] = 1e-3 * np.random.standard_normal()
        x_prev = np.zeros(n, dtype=np.float64)
        x_prev_sq = np.zeros(n, dtype=np.float64)
        e = np.ones(n, dtype=np.float64)
        Jx = np.zeros(n, dtype=np.float64)
        I_inj = np.zeros(n, dtype=np.float64)

        beta_inj = 0.0
        nu_c = 0
        a_t = alpha
        best_cut = 0
        H_opt = float(num_edges)

        for nu in range(num_outer_steps):
            for i in range(n):
                x_prev[i] = x[i]
                x_prev_sq[i] = x[i] * x[i]
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * x_prev[J_indices[jj]]
                Jx[i] = acc
            for i in range(n):
                I_inj[i] = beta_inj * e[i] * Jx[i]
            cut = 0
            for idx in range(num_edges):
                if (x_prev[edge_a[idx]] > 0.0) != (x_prev[edge_b[idx]] > 0.0):
                    cut += 1
            H = float(num_edges - 2 * cut)
            for _ in range(n_x_inner):
                for i in range(n):
                    xi = x[i]
                    dx = (p - 1.0) * xi - xi * xi * xi + I_inj[i]
                    x[i] = xi + dx * dt_x
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
            beta_inj += gamma_growth
            dH = H - H_opt
            a_t = alpha + rho * np.tanh(delta * dH)
            if (nu - nu_c) > tau_iters:
                nu_c = nu
                beta_inj = 0.0
            if H < H_opt:
                H_opt = H
                nu_c = nu
                best_cut = cut
            # ★ 計装: この step 終了時点の running best_cut を記録
            cut_hist[trial_idx, nu] = best_cut
    return cut_hist


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J = build_coupling_matrix(n, edges, -1.0)            # CAC の正しい結合
    params = compute_gset_parameters(J, n)
    params.pop("d_0", None); params.pop("d_1", None)
    seeds = np.arange(TRIALS, dtype=np.int64)

    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    cut_hist = np.zeros((TRIALS, NUM_OUTER_STEPS), dtype=np.int32)

    print(f"G22 N={n} E={k_edges}  outer_steps={NUM_OUTER_STEPS} trials={TRIALS}")
    print(f"[CAC] p={params['p']:.4f} beta0={params['beta0_error']:.4f} tau={params['tau']:.0f}")

    import time
    t0 = time.time()
    _simulate_cac_traj(
        n, NUM_OUTER_STEPS, TRIALS, J.data, J.indices, J.indptr, edge_a, edge_b,
        float(params["p"]), float(params["alpha"]), float(params["rho"]),
        float(params["delta"]), float(params["beta0_error"]),
        float(params["gamma_growth"]), float(params["tau"]),
        int(params["n_x_inner"]), int(params["n_e_inner"]),
        float(params["dt_x"]), float(params["dt_e"]), float(params["e_max"]),
        seeds, cut_hist)
    dt = time.time() - t0

    # 各 step での 100 trial 中の最大 running best cut
    max_traj = cut_hist.max(axis=0)
    mean_traj = cut_hist.mean(axis=0)
    steps = np.arange(1, NUM_OUTER_STEPS + 1)
    final_max = int(max_traj[-1])
    reach_step = int(np.argmax(max_traj >= final_max) + 1)
    print(f"  done [{dt:.1f}s]  最終 最大カット={final_max}  到達 step={reach_step}")

    # ---- 出力ディレクトリ(CLAUDE.md 規約) ----
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in kind_root.iterdir():
        if q.is_dir() and q.name.startswith("v") and q.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(q.name.split("_", 1)[0][1:]))
    out_dir = kind_root / f"v{v + 1}_steps{NUM_OUTER_STEPS}_{TRIALS}trial_maxtraj"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "cut_trajectory.npz",
                        max_traj=max_traj, mean_traj=mean_traj)

    # ---- 作図: 左=線形 / 右=log-x(序盤の急上昇を見る) ----
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax in (axL, axR):
        ax.plot(steps, max_traj, color="#c0392b", lw=2.0,
                label="100 trial 中の最大 running best cut")
        ax.plot(steps, mean_traj, color="#7f8c8d", lw=1.2, alpha=0.8,
                label="100 trial 平均(参考)")
        ax.axhline(KNOWN_BEST, color="red", ls="--", lw=1.2,
                   label=f"既知ベスト {KNOWN_BEST}")
        ax.axhline(final_max, color="#c0392b", ls=":", lw=1.0,
                   label=f"最終最大 {final_max}")
        ax.set_xlabel("outer step", fontsize=12)
        ax.set_ylabel("カット値", fontsize=12)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(alpha=0.3)
    axR.set_xscale("log")
    axL.set_title("線形軸", fontsize=12)
    axR.set_title("log 軸(序盤拡大)", fontsize=12)
    axL.legend(fontsize=8, loc="lower right")
    fig.suptitle(f"CAC の outer step ごとのカット推移(100 trial 中の最大)— G22  "
                 f"最終最大={final_max}(到達 step≈{reach_step})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "cut_trajectory_max.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
