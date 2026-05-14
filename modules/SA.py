"""
inputからG22.txtファイルを読み込む

以下の形式になっている。
N K
A B 1

Nは点数、Kは辺の数
ここで、A Bはつながっていることを示す。

最大カット問題を焼きなまし法(SA)で解く。
"""

import random
import math
import time
import wandb

import numpy as np
from numba import njit, prange

from .verify import (
    compute_cut_from_edges,
    run_all_checks,
)


@njit(cache=True, fastmath=True, parallel=True)
def _simulate_sa_batch_weighted(
    n: int,
    num_iters: int,
    num_trials: int,
    adj_indptr: np.ndarray,    # CSR 風: 各頂点の隣接開始位置
    adj_indices: np.ndarray,   # 隣接頂点のフラットリスト
    adj_weights: np.ndarray,   # 各隣接辺の MAX-CUT 重み
    t_start: float,
    t_end: float,
    seeds: np.ndarray,
):
    """重み付き MAX-CUT 用 SA を num_trials 並列実行する Numba JIT 版。

    隣接情報は CSR 風(indptr, indices, weights)で渡す。各イテレーションで
    頂点 v をランダム選び、delta = -2 * (現状寄与) * (反転で符号反転)
    を O(deg(v)) で計算する。Metropolis 受理判定。
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.int8)

    log_ratio = np.log(t_end / t_start)

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        # 初期解: ランダム ±1
        x = np.empty(n, dtype=np.int8)
        for i in range(n):
            x[i] = 1 if np.random.random() < 0.5 else -1

        # 初期 cut 値を計算
        cur_cut = 0.0
        for i in range(n):
            start = adj_indptr[i]
            end = adj_indptr[i + 1]
            for k in range(start, end):
                j = adj_indices[k]
                if j > i:  # 上三角だけカウント
                    if x[i] != x[j]:
                        cur_cut += adj_weights[k]

        best_cut = cur_cut
        best_x = x.copy()

        for it in range(num_iters):
            # 温度スケジュール (指数冷却)
            progress = it / num_iters
            T = t_start * np.exp(log_ratio * progress)

            # ランダム頂点
            v = np.random.randint(0, n)

            # delta = (反転後の cut 寄与) - (反転前の cut 寄与)
            #       = sum_{u ∈ adj(v)} w(v,u) * ([x_new_v != x_u] - [x_v != x_u])
            # x_new_v = -x_v なので、x_v != x_u が x_v == x_u に切り替わる、逆も然り
            #       = sum_u w(v,u) * sign(x_v == x_u ? +1 : -1)
            delta = 0.0
            start = adj_indptr[v]
            end = adj_indptr[v + 1]
            for k in range(start, end):
                u = adj_indices[k]
                w = adj_weights[k]
                if x[v] == x[u]:
                    delta += w   # 同符号 → 反転で異符号、cut+=w
                else:
                    delta -= w   # 異符号 → 反転で同符号、cut-=w

            # Metropolis 受理
            if delta > 0:
                x[v] = -x[v]
                cur_cut += delta
            elif T > 0:
                if np.random.random() < np.exp(delta / T):
                    x[v] = -x[v]
                    cur_cut += delta

            if cur_cut > best_cut:
                best_cut = cur_cut
                for i in range(n):
                    best_x[i] = x[i]

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_x[i]

    return best_cuts_out, best_signs_out


def simulate_sa_batch(
    n: int,
    edges: list[tuple[int, int]],
    weights: list[float] | None,
    num_iters: int,
    num_trials: int,
    *,
    t_start: float = 2.0,
    t_end: float = 0.001,
    seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SA を num_trials 並列実行する公開ラッパー。重み付き MAX-CUT 対応。

    weights が None の場合、すべての辺で +1 と仮定(unweighted MAX-CUT)。
    """
    if seeds is None:
        seeds = np.arange(num_trials, dtype=np.int64)
    seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    # CSR 風隣接配列を構築
    if weights is None:
        weights = [1.0] * len(edges)
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]
    indices = np.zeros(int(indptr[-1]), dtype=np.int64)
    adj_w = np.zeros(int(indptr[-1]), dtype=np.float64)
    cursor = indptr[:-1].copy()
    for (a, b), w in zip(edges, weights):
        indices[cursor[a]] = b; adj_w[cursor[a]] = w; cursor[a] += 1
        indices[cursor[b]] = a; adj_w[cursor[b]] = w; cursor[b] += 1

    best_cuts, best_signs = _simulate_sa_batch_weighted(
        n, num_iters, num_trials,
        indptr, indices, adj_w,
        float(t_start), float(t_end),
        seeds,
    )
    return best_cuts, best_signs


def load_graph(filepath: str):
    """グラフを読み込み、隣接リストを返す"""
    with open(filepath, "r") as f:
        first_line = f.readline().split()
        n, k = int(first_line[0]), int(first_line[1])
        adj = [[] for _ in range(n)]
        edges = []
        for _ in range(k):
            parts = f.readline().split()
            a, b = int(parts[0]) - 1, int(parts[1]) - 1  # 0-indexed
            adj[a].append(b)
            adj[b].append(a)
            edges.append((a, b))
    return n, k, adj, edges


def compute_delta(x: list[int], adj: list[list[int]], v: int) -> int:
    """頂点vを反転した場合のカット数の変化量を計算
    正なら改善、負なら悪化"""
    delta = 0
    for u in adj[v]:
        if x[v] == x[u]:
            delta += 1   # 同じ→異なる: カット+1
        else:
            delta -= 1   # 異なる→同じ: カット-1
    return delta


def simulated_annealing(
    n: int,
    adj: list[list[int]],
    edges: list[tuple[int, int]],
    t_start: float = 2.0,
    t_end: float = 0.001,
    time_limit: float = 30.0,
    log_interval: int = 10000,
) -> tuple[list[int], int]:
    """焼きなまし法でMAX-CUTを解く"""

    # 初期解: ランダム割り当て
    x = [random.randint(0, 1) for _ in range(n)]
    current_cut = compute_cut_from_edges(x, edges)
    best_x = x[:]
    best_cut = current_cut

    start_time = time.time()
    iteration = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            break

        # 温度スケジュール (指数冷却)
        progress = elapsed / time_limit
        temperature = t_start * ((t_end / t_start) ** progress)

        # ランダムに頂点を選んで反転を試みる
        v = random.randint(0, n - 1)
        delta = compute_delta(x, adj, v)

        # 受理判定
        if delta > 0:
            # 改善: 常に受理
            x[v] ^= 1
            current_cut += delta
        elif temperature > 0:
            # 悪化: 確率的に受理
            prob = math.exp(delta / temperature)
            if random.random() < prob:
                x[v] ^= 1
                current_cut += delta

        # ベスト更新
        if current_cut > best_cut:
            best_cut = current_cut
            best_x = x[:]

        iteration += 1

        # wandbログ
        if iteration % log_interval == 0:
            wandb.log({
                "iteration": iteration,
                "current_cut": current_cut,
                "best_cut": best_cut,
                "temperature": temperature,
                "elapsed": elapsed,
                "progress": progress,
            })

    return best_x, best_cut


def main():
    # ハイパーパラメータ
    config = {
        "t_start": 2.0,
        "t_end": 0.001,
        "time_limit": 30.0,
        "log_interval": 10000,
        "seed": 42,
    }

    # wandb初期化
    wandb.init(project="max-cut-sa", config=config)
    config = wandb.config

    random.seed(config.seed)

    # グラフ読み込み
    filepath = "input/G22.txt"
    n, k, adj, edges = load_graph(filepath)
    print(f"N={n}, K={k}")

    # 焼きなまし法実行
    print("Running Simulated Annealing...")
    best_x, best_cut = simulated_annealing(
        n, adj, edges,
        t_start=config.t_start,
        t_end=config.t_end,
        time_limit=config.time_limit,
        log_interval=config.log_interval,
    )

    print(f"Best cut: {best_cut}")
    print(f"Known best: 13359")
    print(f"Random expected: ~9995")

    # 検算
    run_all_checks(best_x, n, k, adj, edges, best_cut)

    # 最終結果をwandbに記録
    wandb.log({"final_best_cut": best_cut})
    wandb.summary["best_cut"] = best_cut
    wandb.summary["ratio_to_known_best"] = best_cut / 13359

    # 結果を出力
    for xi in best_x:
        print(xi)

    wandb.finish()


if __name__ == "__main__":
    main()
