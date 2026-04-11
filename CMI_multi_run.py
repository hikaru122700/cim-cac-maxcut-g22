"""
CIM を 100回実行するラッパースクリプト。

論文 Inoue & Yoshida (2022) の Fig.8 再現用。
各試行 (trial) で乱数シードを変えて simulate_cim を呼び出し、
得られた cut 数の分布を集計して wandb にログする。

wandb には以下を記録する:
  - trial ごとの cut 数 (history)
  - 走行中の running mean / running max
  - 最終統計 (平均, 最良, 最悪, 標準偏差)
  - 分布のヒストグラム (論文 Fig.8 に相当)
  - 全 trial の cut 数テーブル
"""

import time

import numpy as np
import wandb

from CIM import (
    build_coupling_matrix,
    load_graph,
    simulate_cim_batch,
)
from scripts.verify import run_all_checks


def main():
    # ==== ハイパーパラメータ ====
    # 物理パラメータは CIM.py と同じ。num_trials と seed_base を追加。
    config = {
        # 物理パラメータ (論文 Section 3)
        "kappa": 130.0,             # W^(-1/2) m^(-1)
        "L": 0.05,                  # m  (5 cm)
        "gamma": 42.09,             # W^(-1)
        "loss_dB": 11.0,            # dB
        "bandwidth": 1.0e9,         # Hz  (1 GHz)
        "photon_energy_J": 1.28e-19, # J   (ℏω at 1550nm, ノイズ単位変換)
        "dP_per_round": 0.05e-3,    # W/round  (0.05 mW)
        "coupling": -0.03,          # J_ij for G22 edges

        # シミュレーション設定
        "num_rounds": 1500,         # 1 trial あたりのラウンド数
        "num_trials": 100,          # 試行回数 (論文 Fig.8 は 100 回)
        "seed_base": 0,             # seed は seed_base + trial_idx で決まる
    }

    # ==== wandb 初期化 (1 つの run に 100 trial をまとめる) ====
    wandb.init(project="cim-max-cut-multi", config=config)
    cfg = wandb.config

    # 損失(dB) → 透過率 η
    eta = 10.0 ** (-cfg.loss_dB / 10.0)
    wandb.config.update({"eta": eta}, allow_val_change=True)

    # ==== グラフと結合行列は 1 回だけ構築して使いまわす ====
    filepath = "input/G22.txt"
    n, k_edges, adj, edges = load_graph(filepath)
    print(f"N={n}, K={k_edges}, eta={eta:.4f}")

    J = build_coupling_matrix(n, edges, cfg.coupling)

    # ==== 100 試行を Numba 並列で一気に実行 ====
    # 各 trial は seed_base + trial_idx を seed として渡す。
    # simulate_cim_batch が @njit(parallel=True) で CPU コアに分散実行する。
    seeds = np.array(
        [cfg.seed_base + i for i in range(cfg.num_trials)], dtype=np.int64
    )

    print(f"Running {cfg.num_trials} trials in parallel (Numba prange)...")
    t_start = time.time()
    best_cuts_arr, best_signs_arr = simulate_cim_batch(
        n=n,
        J=J,
        edges=edges,
        num_rounds=cfg.num_rounds,
        num_trials=cfg.num_trials,
        kappa=cfg.kappa,
        L=cfg.L,
        gamma=cfg.gamma,
        eta=eta,
        bandwidth=cfg.bandwidth,
        photon_energy=cfg.photon_energy_J,
        dP_per_round=cfg.dP_per_round,
        seeds=seeds,
    )
    t_elapsed = time.time() - t_start
    print(f"Finished in {t_elapsed:.2f} sec ({t_elapsed/cfg.num_trials*1000:.1f} ms/trial)")

    # ==== 全 trial の結果を trial 順に wandb にロギング ====
    # 並列実行後、trial 順に running_mean / running_max を計算して逐次ログ
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

    # 全体最良解
    best_overall_trial = int(np.argmax(best_cuts_arr))
    best_overall_cut = int(best_cuts_arr[best_overall_trial])
    best_overall_x: list[int] = best_signs_arr[best_overall_trial].astype(np.int64).tolist()

    # ==== 全 trial の統計 ====
    cuts_arr = best_cuts_arr.astype(np.int64)
    mean_cut = float(cuts_arr.mean())
    std_cut = float(cuts_arr.std())
    max_cut = int(cuts_arr.max())
    min_cut = int(cuts_arr.min())
    median_cut = float(np.median(cuts_arr))

    print("=" * 60)
    print(f"Results over {cfg.num_trials} trials:")
    print(f"  mean  = {mean_cut:.2f}  (paper: 13275)")
    print(f"  best  = {max_cut}      (paper: 13321)")
    print(f"  worst = {min_cut}")
    print(f"  std   = {std_cut:.2f}")
    print(f"  median= {median_cut:.1f}")
    print(f"  best trial: #{best_overall_trial + 1} (seed={cfg.seed_base + best_overall_trial})")
    print("=" * 60)

    # ==== 全体最良解の検算 ====
    run_all_checks(best_overall_x, n, k_edges, adj, edges, best_overall_cut)

    # ==== wandb summary と ヒストグラム ====
    wandb.summary["mean_cut"] = mean_cut
    wandb.summary["best_cut"] = max_cut
    wandb.summary["worst_cut"] = min_cut
    wandb.summary["std_cut"] = std_cut
    wandb.summary["median_cut"] = median_cut
    wandb.summary["best_trial_idx"] = best_overall_trial
    wandb.summary["ratio_to_known_best"] = max_cut / 13359
    wandb.summary["paper_mean"] = 13275
    wandb.summary["paper_best"] = 13321
    wandb.summary["known_best"] = 13359

    # ヒストグラム (論文 Fig.8 相当)
    wandb.log({"cut_histogram": wandb.Histogram(cuts_arr.tolist())})

    # 全 trial の結果テーブル
    trial_table = wandb.Table(
        columns=["trial", "seed", "cut"],
        data=[
            [i + 1, cfg.seed_base + i, int(cuts_arr[i])]
            for i in range(cfg.num_trials)
        ],
    )
    wandb.log({"trials_table": trial_table})

    wandb.finish()


if __name__ == "__main__":
    main()
