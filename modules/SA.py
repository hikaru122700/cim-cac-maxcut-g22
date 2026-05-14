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

from .verify import (
    compute_cut_from_edges,
    run_all_checks,
)


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
