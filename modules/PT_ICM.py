"""PT-ICM (Parallel Tempering + Isoenergetic Cluster Move) for MAX-CUT.

参考: Zhu, Ochoa, Katzgraber, "Efficient cluster algorithm for spin glasses in
any space dimension", PRL 115, 077201 (2015).

CIM ベンチマーク文献(Hamerly+ 2019, McMahon+ 2016, Leleu+ 2019 ほか)で
ヒューリスティクスの参照点として頻繁に用いられる SOTA 級 SA 派生手法。

実装の要点
----------
* NT 個の温度 T_0 < T_1 < ... < T_{NT-1}(幾何分布)で並列鎖を回す
* 2 系統 (A, B) 走らせ、ICM (Houdayer cluster move) で交叉する
* ICM はマイナス overlap (q_i = s_A_i * s_B_i = -1) の連結成分内で
  s_A_i ↔ s_B_i をスワップ。境界はすべて q=+1 サイトなので
  joint Hamiltonian H_A+H_B 保存 = **rejection-free**
* Numba JIT + prange(num_trials) で trial レベル並列化
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def _build_csr(n: int, edges: np.ndarray, weights: np.ndarray):
    """edges (E, 2) と weights (E,) から CSR (indptr, indices, w) を構築。"""
    e = edges.shape[0]
    deg = np.zeros(n, dtype=np.int64)
    for k in range(e):
        deg[edges[k, 0]] += 1
        deg[edges[k, 1]] += 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]
    nnz = indptr[n]
    indices = np.zeros(nnz, dtype=np.int64)
    adj_w = np.zeros(nnz, dtype=np.float64)
    cursor = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cursor[i] = indptr[i]
    for k in range(e):
        a = edges[k, 0]
        b = edges[k, 1]
        w = weights[k]
        indices[cursor[a]] = b
        adj_w[cursor[a]] = w
        cursor[a] += 1
        indices[cursor[b]] = a
        adj_w[cursor[b]] = w
        cursor[b] += 1
    return indptr, indices, adj_w


@njit(cache=True, fastmath=True, parallel=True)
def _simulate_pticm_batch(
    n: int,
    num_trials: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    adj_w: np.ndarray,
    T_ladder: np.ndarray,        # ascending, shape (NT,)
    num_sweeps: int,
    sweep_len: int,              # 1 sweep の単一スピン更新回数(通常 n)
    swap_interval: int,
    icm_interval: int,
    sample_interval: int,         # この sweep ごとに best-so-far を記録(>=1)
    num_samples: int,             # = num_sweeps // sample_interval
    seeds: np.ndarray,
):
    """PT-ICM を num_trials 並列実行する内部ルーチン。

    Returns
    -------
    best_cuts : (num_trials,) 各 trial の最終最良カット
    best_signs : (num_trials, n) 最終最良解の ±1 ベクトル
    trajectory : (num_trials, num_samples) 各サンプル時点での best-so-far
    """
    NT = T_ladder.shape[0]
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.int8)
    trajectory = np.zeros((num_trials, num_samples), dtype=np.float64)

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        # ---------------- 初期化 ----------------
        s_A = np.empty((NT, n), dtype=np.int8)
        s_B = np.empty((NT, n), dtype=np.int8)
        cut_A = np.zeros(NT, dtype=np.float64)
        cut_B = np.zeros(NT, dtype=np.float64)

        for k in range(NT):
            for i in range(n):
                s_A[k, i] = 1 if np.random.random() < 0.5 else -1
                s_B[k, i] = 1 if np.random.random() < 0.5 else -1

        # 初期 cut 値
        for k in range(NT):
            ca = 0.0
            cb = 0.0
            for i in range(n):
                start = indptr[i]
                end = indptr[i + 1]
                for kk in range(start, end):
                    j = indices[kk]
                    if j > i:
                        if s_A[k, i] != s_A[k, j]:
                            ca += adj_w[kk]
                        if s_B[k, i] != s_B[k, j]:
                            cb += adj_w[kk]
            cut_A[k] = ca
            cut_B[k] = cb

        # ベスト初期値: 最低温度の方が良いとは限らないので全部走査
        best_cut = cut_A[0]
        best_k = 0
        best_side = 0  # 0 = A, 1 = B
        for k in range(NT):
            if cut_A[k] > best_cut:
                best_cut = cut_A[k]
                best_k = k
                best_side = 0
            if cut_B[k] > best_cut:
                best_cut = cut_B[k]
                best_k = k
                best_side = 1
        best_x = np.empty(n, dtype=np.int8)
        if best_side == 0:
            for i in range(n):
                best_x[i] = s_A[best_k, i]
        else:
            for i in range(n):
                best_x[i] = s_B[best_k, i]

        # ICM 用バッファ
        visited = np.zeros(n, dtype=np.int8)
        queue = np.empty(n, dtype=np.int64)
        cluster = np.empty(n, dtype=np.int64)
        minus_sites = np.empty(n, dtype=np.int64)

        # ---------------- メインループ ----------------
        for sweep in range(num_sweeps):
            # === 1. 各温度・各系統で Metropolis スイープ ===
            for k in range(NT):
                T = T_ladder[k]
                inv_T = 1.0 / T if T > 0 else 1.0e18

                # 系統 A
                for _ in range(sweep_len):
                    v = np.random.randint(0, n)
                    delta = 0.0
                    start = indptr[v]
                    end = indptr[v + 1]
                    for kk in range(start, end):
                        u = indices[kk]
                        if s_A[k, v] == s_A[k, u]:
                            delta += adj_w[kk]
                        else:
                            delta -= adj_w[kk]
                    if delta > 0.0:
                        s_A[k, v] = -s_A[k, v]
                        cut_A[k] += delta
                    else:
                        if np.random.random() < np.exp(delta * inv_T):
                            s_A[k, v] = -s_A[k, v]
                            cut_A[k] += delta

                # 系統 B
                for _ in range(sweep_len):
                    v = np.random.randint(0, n)
                    delta = 0.0
                    start = indptr[v]
                    end = indptr[v + 1]
                    for kk in range(start, end):
                        u = indices[kk]
                        if s_B[k, v] == s_B[k, u]:
                            delta += adj_w[kk]
                        else:
                            delta -= adj_w[kk]
                    if delta > 0.0:
                        s_B[k, v] = -s_B[k, v]
                        cut_B[k] += delta
                    else:
                        if np.random.random() < np.exp(delta * inv_T):
                            s_B[k, v] = -s_B[k, v]
                            cut_B[k] += delta

            # === 2. ベスト更新 ===
            for k in range(NT):
                if cut_A[k] > best_cut:
                    best_cut = cut_A[k]
                    for i in range(n):
                        best_x[i] = s_A[k, i]
                if cut_B[k] > best_cut:
                    best_cut = cut_B[k]
                    for i in range(n):
                        best_x[i] = s_B[k, i]

            # === 3. PT swap (隣接温度間で偶奇交互に試行) ===
            if (sweep + 1) % swap_interval == 0:
                # 系統 A, B それぞれで独立に。
                # β = 1/T で、低温(小 k)が高 β。
                # 受理率: exp((β_k - β_{k+1}) * (cut[k+1] - cut[k]))
                #   高温側が高 cut を引いたら無条件受理。
                for k in range(NT - 1):
                    d_beta = 1.0 / T_ladder[k] - 1.0 / T_ladder[k + 1]
                    # A
                    arg = d_beta * (cut_A[k + 1] - cut_A[k])
                    if arg >= 0.0 or np.random.random() < np.exp(arg):
                        for i in range(n):
                            tmp = s_A[k, i]
                            s_A[k, i] = s_A[k + 1, i]
                            s_A[k + 1, i] = tmp
                        tc = cut_A[k]
                        cut_A[k] = cut_A[k + 1]
                        cut_A[k + 1] = tc
                    # B
                    arg = d_beta * (cut_B[k + 1] - cut_B[k])
                    if arg >= 0.0 or np.random.random() < np.exp(arg):
                        for i in range(n):
                            tmp = s_B[k, i]
                            s_B[k, i] = s_B[k + 1, i]
                            s_B[k + 1, i] = tmp
                        tc = cut_B[k]
                        cut_B[k] = cut_B[k + 1]
                        cut_B[k + 1] = tc

            # === 4. ICM (Houdayer cluster move) ===
            if (sweep + 1) % icm_interval == 0:
                for k in range(NT):
                    # マイナスサイト q_i = s_A * s_B = -1 を列挙
                    n_minus = 0
                    for i in range(n):
                        if s_A[k, i] != s_B[k, i]:
                            minus_sites[n_minus] = i
                            n_minus += 1
                    if n_minus < 2:
                        continue  # クラスタを作るほどの差異がない

                    # ランダムシードサイトから BFS で連結成分を取得
                    seed_idx = np.random.randint(0, n_minus)
                    seed_node = minus_sites[seed_idx]

                    for i in range(n):
                        visited[i] = 0
                    queue[0] = seed_node
                    visited[seed_node] = 1
                    qhead = 0
                    qtail = 1
                    n_cluster = 0
                    while qhead < qtail:
                        v = queue[qhead]
                        qhead += 1
                        cluster[n_cluster] = v
                        n_cluster += 1
                        start = indptr[v]
                        end = indptr[v + 1]
                        for kk in range(start, end):
                            u = indices[kk]
                            if visited[u] == 0 and s_A[k, u] != s_B[k, u]:
                                visited[u] = 1
                                queue[qtail] = u
                                qtail += 1

                    if n_cluster < 1:
                        continue

                    # クラスタ反転による境界 cut 変化を計算
                    # 境界辺 (v in cluster, u not in cluster):
                    #   ΔCut_A: +w if s_A[v]==s_A[u] else -w
                    #   ΔCut_B: +w if s_B[v]==s_B[u] else -w
                    # 内部辺は不変(両端とも反転)
                    dA = 0.0
                    dB = 0.0
                    for cc in range(n_cluster):
                        v = cluster[cc]
                        start = indptr[v]
                        end = indptr[v + 1]
                        for kk in range(start, end):
                            u = indices[kk]
                            if visited[u] == 0:
                                w_uv = adj_w[kk]
                                if s_A[k, v] == s_A[k, u]:
                                    dA += w_uv
                                else:
                                    dA -= w_uv
                                if s_B[k, v] == s_B[k, u]:
                                    dB += w_uv
                                else:
                                    dB -= w_uv

                    # Houdayer rejection-free: dA + dB == 0(マイナスクラスタ
                    # 境界は q_u = +1 サイトに限られるため数学的に保証される)
                    # 念のためそのまま採用し、cut_A/cut_B を更新。
                    for cc in range(n_cluster):
                        v = cluster[cc]
                        s_A[k, v] = -s_A[k, v]
                        s_B[k, v] = -s_B[k, v]
                    cut_A[k] += dA
                    cut_B[k] += dB

                # ICM 後にも best 更新を見ておく(個別 cut が上がる可能性)
                for k in range(NT):
                    if cut_A[k] > best_cut:
                        best_cut = cut_A[k]
                        for i in range(n):
                            best_x[i] = s_A[k, i]
                    if cut_B[k] > best_cut:
                        best_cut = cut_B[k]
                        for i in range(n):
                            best_x[i] = s_B[k, i]

            # === 5. trajectory サンプリング(best-so-far を記録) ===
            if (sweep + 1) % sample_interval == 0:
                sample_idx = (sweep + 1) // sample_interval - 1
                if 0 <= sample_idx < num_samples:
                    trajectory[trial_idx, sample_idx] = best_cut

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_x[i]

    return best_cuts_out, best_signs_out, trajectory


def make_geometric_ladder(t_min: float, t_max: float, num_temps: int) -> np.ndarray:
    """T_min から T_max への昇順幾何温度ラダーを返す(長さ num_temps)。"""
    if num_temps < 2:
        return np.array([t_min], dtype=np.float64)
    return np.geomspace(t_min, t_max, num=num_temps, dtype=np.float64)


def simulate_pticm_batch(
    n: int,
    edges: list[tuple[int, int]],
    weights: list[float] | None,
    num_trials: int,
    *,
    num_sweeps: int = 200,
    sweep_len: int | None = None,
    t_min: float = 0.05,
    t_max: float = 3.0,
    num_temps: int = 12,
    T_ladder: np.ndarray | None = None,
    swap_interval: int = 1,
    icm_interval: int = 5,
    seeds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PT-ICM を num_trials 並列実行する公開 API。

    Parameters
    ----------
    n : 頂点数
    edges : (E,) リスト of (a, b)、0-indexed
    weights : 重みリスト or None(None なら全辺 +1)
    num_trials : 並列試行数
    num_sweeps : 全体スイープ数
    sweep_len : 1 スイープでの単スピン試行数。None なら n
    t_min, t_max, num_temps : 幾何温度ラダーの両端と段数(T_ladder 未指定時)
    T_ladder : 直接指定する場合の昇順温度配列(優先)
    swap_interval : PT swap を試みるスイープ間隔
    icm_interval : ICM cluster move を試みるスイープ間隔
    seeds : 各 trial の乱数シード(None なら 0..num_trials-1)
    """
    if seeds is None:
        seeds = np.arange(num_trials, dtype=np.int64)
    seeds = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    if weights is None:
        weights = [1.0] * len(edges)
    edges_arr = np.asarray(edges, dtype=np.int64)
    weights_arr = np.asarray(weights, dtype=np.float64)

    if T_ladder is None:
        T_ladder = make_geometric_ladder(t_min, t_max, num_temps)
    T_ladder = np.ascontiguousarray(np.asarray(T_ladder, dtype=np.float64))
    if not np.all(np.diff(T_ladder) >= 0):
        # 念のため昇順に並べ替え
        T_ladder = np.sort(T_ladder)

    if sweep_len is None:
        sweep_len = n

    indptr, indices, adj_w = _build_csr(n, edges_arr, weights_arr)

    best_cuts, best_signs = _simulate_pticm_batch(
        n,
        num_trials,
        indptr,
        indices,
        adj_w,
        T_ladder,
        int(num_sweeps),
        int(sweep_len),
        int(swap_interval),
        int(icm_interval),
        seeds,
    )
    return best_cuts, best_signs
