"""
compare.py — CIM vs CAC vs SA を G22 で比較する。

各手法を 100 trial ずつ実行し、解品質の分布・ベスト更新曲線・統計を比較。
結果は results/ 配下に画像で保存される。

========================================================================
【なぜこの条件設定か】
- CIM は 1500 ラウンドで物理的に収束する(それ以上回しても改善しない)
- CAC は外ループ数に応じて対数的に改善する
- SA は per-trial の time_limit に応じて改善する

単純に「総実行時間を揃える」とどれかが過剰・不足になるので、本スクリプトでは
「各手法を現実的な設定で 100 trial 回す」方針とした。壁時計時間も併記するので、
スクリプト冒頭の config を書き換えれば time-aligned 比較にも簡単に変更できる。

デフォルト設定:
  CIM: num_rounds=1500 (論文ベースラインで収束)
  CAC: num_outer_steps=100000 (CPU で妥当、論文の FPGA 運用の約 1/1000)
  SA : time_limit=2.0 sec/trial (SA が十分探索できる時間)
========================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境用
import matplotlib.pyplot as plt
import numpy as np

from modules.CAC import compute_gset_parameters, simulate_cac_batch
from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch


# ============================================================
#  SA (main.py から抽出した独立版: wandb 非依存)
# ============================================================
def run_sa_trial(
    n: int,
    adj: list[list[int]],
    edges: list[tuple[int, int]],
    t_start: float,
    t_end: float,
    time_limit: float,
    seed: int,
) -> int:
    """1 trial の SA を実行して best_cut を返す。

    main.py の simulated_annealing と同一アルゴリズム(1-flip + 指数冷却)。
    wandb ログを取り除き、外部から seed を渡せるようにしたローカル版。
    """
    random.seed(seed)
    # 初期解
    x = [random.randint(0, 1) for _ in range(n)]
    current_cut = sum(1 for a, b in edges if x[a] != x[b])
    best_cut = current_cut

    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= time_limit:
            break

        # 温度スケジュール (指数冷却)
        progress = elapsed / time_limit
        temperature = t_start * ((t_end / t_start) ** progress)

        # ランダムに頂点を選んで反転を試みる
        v = random.randint(0, n - 1)
        # delta: 反転した場合のカット数変化量
        delta = 0
        for u in adj[v]:
            if x[v] == x[u]:
                delta += 1
            else:
                delta -= 1

        # 受理判定
        if delta > 0:
            x[v] ^= 1
            current_cut += delta
        elif temperature > 0:
            prob = math.exp(delta / temperature)
            if random.random() < prob:
                x[v] ^= 1
                current_cut += delta

        if current_cut > best_cut:
            best_cut = current_cut

    return best_cut


def run_sa_batch(
    n: int,
    adj: list[list[int]],
    edges: list[tuple[int, int]],
    num_trials: int,
    time_per_trial: float,
    seed_base: int = 0,
) -> np.ndarray:
    """SA を num_trials 回、seed を変えて直列実行。"""
    cuts = np.zeros(num_trials, dtype=np.int64)
    for trial in range(num_trials):
        cuts[trial] = run_sa_trial(
            n, adj, edges,
            t_start=2.0, t_end=0.001,
            time_limit=time_per_trial,
            seed=seed_base + trial,
        )
    return cuts


# ============================================================
#  メイン
# ============================================================
def main():
    # ==== 設定 ====
    NUM_TRIALS = 100
    SEED_BASE = 0

    # CIM: num_rounds=1500 で論文と同じ(物理的に収束)
    CIM_NUM_ROUNDS = 1500
    CIM_COUPLING = -0.03       # CIM は弱結合 J_ij = -0.03
    CIM_PARAMS = dict(
        kappa=130.0, L=0.05, gamma=42.09,
        eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
    )

    # CAC: num_outer_steps=100000 で CPU に現実的な計算量
    CAC_NUM_OUTER_STEPS = 100_000
    CAC_COUPLING = -1.0        # CAC は生の ±1 結合(β_inj でスケール)

    # SA: 1 trial あたり 2 秒
    SA_TIME_PER_TRIAL = 2.0

    KNOWN_BEST = 13359

    # ==== グラフ読み込み ====
    n, k_edges, adj, edges = load_graph("input/G22.txt")
    print(f"Graph: N={n}, K={k_edges}")

    # 乱数シード配列 (CIM/CAC は numpy / numba なので int64 の seed)
    seeds = np.arange(SEED_BASE, SEED_BASE + NUM_TRIALS, dtype=np.int64)

    # ==== CIM 実行 ====
    print(f"\n[CIM] {NUM_TRIALS} trials (num_rounds={CIM_NUM_ROUNDS})...")
    J_cim = build_coupling_matrix(n, edges, CIM_COUPLING)
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n,
        J=J_cim,
        edges=edges,
        num_rounds=CIM_NUM_ROUNDS,
        num_trials=NUM_TRIALS,
        seeds=seeds,
        **CIM_PARAMS,
    )
    cim_time = time.time() - t0
    print(f"  time: {cim_time:.2f} sec  ({cim_time / NUM_TRIALS * 1000:.1f} ms/trial)")
    print(f"  mean={cim_cuts.mean():.1f}  best={cim_cuts.max()}")

    # ==== CAC 実行 ====
    print(f"\n[CAC] {NUM_TRIALS} trials (num_outer_steps={CAC_NUM_OUTER_STEPS})...")
    J_cac = build_coupling_matrix(n, edges, CAC_COUPLING)
    cac_params = compute_gset_parameters(J_cac, n)
    # compute_gset_parameters の診断用キーを除外
    cac_kwargs = {kk: vv for kk, vv in cac_params.items() if kk not in ("d_0", "d_1")}
    t0 = time.time()
    cac_cuts, _ = simulate_cac_batch(
        n=n,
        J=J_cac,
        edges=edges,
        num_outer_steps=CAC_NUM_OUTER_STEPS,
        num_trials=NUM_TRIALS,
        seeds=seeds,
        **cac_kwargs,
    )
    cac_time = time.time() - t0
    print(f"  time: {cac_time:.2f} sec  ({cac_time / NUM_TRIALS * 1000:.1f} ms/trial)")
    print(f"  mean={cac_cuts.mean():.1f}  best={cac_cuts.max()}")

    # ==== SA 実行 (直列) ====
    print(f"\n[SA]  {NUM_TRIALS} trials (time_limit={SA_TIME_PER_TRIAL}s/trial, serial)...")
    t0 = time.time()
    sa_cuts = run_sa_batch(
        n, adj, edges,
        num_trials=NUM_TRIALS,
        time_per_trial=SA_TIME_PER_TRIAL,
        seed_base=SEED_BASE,
    )
    sa_time = time.time() - t0
    print(f"  time: {sa_time:.2f} sec  ({sa_time / NUM_TRIALS * 1000:.1f} ms/trial)")
    print(f"  mean={sa_cuts.mean():.1f}  best={sa_cuts.max()}")

    # ==== 統計サマリ ====
    results = {"CIM": cim_cuts, "CAC": cac_cuts, "SA": sa_cuts}
    times = {"CIM": cim_time, "CAC": cac_time, "SA": sa_time}

    print("\n" + "=" * 78)
    print(
        f"{'Method':<6} {'Mean':>10} {'Best':>8} {'Worst':>8} {'Std':>8} "
        f"{'Ratio':>8} {'Time[s]':>10}"
    )
    print("-" * 78)
    for name in ["CIM", "CAC", "SA"]:
        cuts = results[name]
        print(
            f"{name:<6} {cuts.mean():>10.1f} {int(cuts.max()):>8d} {int(cuts.min()):>8d} "
            f"{cuts.std():>8.1f} {cuts.max() / KNOWN_BEST:>8.4f} {times[name]:>10.2f}"
        )
    print("=" * 78)
    print(f"Known best: {KNOWN_BEST}")
    print(f"Paper CIM:  mean=13275, best=13321")
    print(f"Paper CAC:  best=13359 (p_0=0.11 on FPGA)")

    # ==== 結果を画像化 ====
    os.makedirs("results", exist_ok=True)

    colors = {"CIM": "#1f77b4", "CAC": "#ff7f0e", "SA": "#2ca02c"}

    # --- Figure 1: ヒストグラム (3 パネル、共通 x 軸) ---
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    all_cuts = np.concatenate([cim_cuts, cac_cuts, sa_cuts])
    x_min = int(all_cuts.min()) - 20
    x_max = max(int(all_cuts.max()) + 20, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 35)

    for ax, name in zip(axes, ["CIM", "CAC", "SA"]):
        cuts = results[name]
        ax.hist(
            cuts, bins=bins, color=colors[name], alpha=0.75, edgecolor="black", linewidth=0.5
        )
        ax.axvline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3, label=f"known best {KNOWN_BEST}")
        ax.axvline(cuts.mean(), color="black", linestyle=":", linewidth=1.3, label=f"mean {cuts.mean():.0f}")
        ax.set_title(
            f"{name}\ntime: {times[name]:.1f}s  "
            f"mean: {cuts.mean():.0f}  best: {int(cuts.max())}",
            fontsize=11,
        )
        ax.set_xlabel("cut value")
        ax.set_ylabel("count")
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")

    fig1.suptitle(
        f"CIM vs CAC vs SA — MAX-CUT on G22 ({NUM_TRIALS} trials each)",
        fontsize=13,
    )
    fig1.tight_layout()
    hist_path = os.path.join("results", "compare_histogram.png")
    fig1.savefig(hist_path, dpi=150)
    plt.close(fig1)
    print(f"\nSaved: {hist_path}")

    # --- Figure 2: running best (trial 進行に対するこれまでの最良) ---
    fig2, ax2 = plt.subplots(figsize=(10, 5.5))
    for name in ["CIM", "CAC", "SA"]:
        cuts = results[name]
        running = np.maximum.accumulate(cuts)
        ax2.plot(
            np.arange(1, NUM_TRIALS + 1),
            running,
            label=f"{name}  (wall time: {times[name]:.1f}s)",
            color=colors[name],
            linewidth=2.0,
        )
    ax2.axhline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3, label=f"known best {KNOWN_BEST}")
    ax2.set_xlabel("trial")
    ax2.set_ylabel("best cut so far")
    ax2.set_title("Running best cut vs trial count (G22)")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    running_path = os.path.join("results", "compare_running_best.png")
    fig2.savefig(running_path, dpi=150)
    plt.close(fig2)
    print(f"Saved: {running_path}")

    # --- Figure 3: 全体サマリ (bar plot of mean & best) ---
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    methods = ["CIM", "CAC", "SA"]
    means = [results[m].mean() for m in methods]
    bests = [int(results[m].max()) for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax3.bar(x - width / 2, means, width, label="mean", color=[colors[m] for m in methods], alpha=0.6)
    ax3.bar(x + width / 2, bests, width, label="best", color=[colors[m] for m in methods], alpha=1.0)
    ax3.axhline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3, label=f"known best {KNOWN_BEST}")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("cut value")
    ax3.set_title(f"Mean and best cut over {NUM_TRIALS} trials (G22)")
    # y 範囲を意味のある所だけに絞る
    y_min = min(means) - 50
    y_max = KNOWN_BEST + 30
    ax3.set_ylim(y_min, y_max)
    ax3.legend(loc="lower right")
    ax3.grid(axis="y", alpha=0.3)
    # 各バーの上に数値を書く
    for i, m in enumerate(methods):
        ax3.text(i - width / 2, means[i] + 2, f"{means[i]:.0f}", ha="center", fontsize=9)
        ax3.text(i + width / 2, bests[i] + 2, f"{bests[i]}", ha="center", fontsize=9)
    fig3.tight_layout()
    bar_path = os.path.join("results", "compare_bar.png")
    fig3.savefig(bar_path, dpi=150)
    plt.close(fig3)
    print(f"Saved: {bar_path}")


if __name__ == "__main__":
    main()
