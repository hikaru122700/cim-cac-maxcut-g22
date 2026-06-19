"""GA.py — MAX-CUT 用メメティックアルゴリズム (Memetic GA)。

採用手法: Wu & Hao 2013 の枠組み(グルーピング交叉 + 摂動付き Tabu Search +
距離・品質併用プール更新 "DisQual")を、制約なし MAX-CUT 向けに単一頂点フリップ
近傍で具現化したもの(MACUT / TSHEA 系列)。

参考:
- Q. Wu, J.-K. Hao, "Memetic search for the max-bisection problem,"
  Computers & Operations Research 40(1):166-179, 2013.
- Q. Wu, J.-K. Hao, "A Memetic Approach for the Max-Cut Problem," PPSN 2012,
  LNCS 7492:297-306.
- Q. Wu, Y. Wang, Z. Lu, "A tabu search based hybrid evolutionary algorithm for
  the max-cut problem," Applied Soft Computing 34:827-837, 2015.

========================================================================
【手法の骨子】
スピン s in {0,1}^n。カット f(s)=Σ_{(i,j)∈E} w_ij [s_i≠s_j]。
- 局所探索 = Tabu Search(単一フリップ近傍, 動的 tabu tenure, aspiration, 摂動)
- 移動利得 Δ_v = (同じ側の隣接重み和) − (反対側の隣接重み和)。フリップで cut += Δ_v。
  反転後の増分更新は O(deg(v))。
- 交叉 = グルーピング(共通分割継承)交叉。両親の合意した側を子に引き継ぎ、
  残りを貪欲に両親から等距離になるよう割り当てる。
- 集団更新 = DisQual。子を仮に入れ、品質と最小ハミング距離の重み付き score が
  最小の個体を捨てる(多様性維持)。
========================================================================

公開 API:
- simulate_ga_batch(...)     : num_trials 個の独立メメティック実行(分布取得用)
- tabu_refine_batch(...)     : 与えた初期解を TS のみで磨く(ハイブリッド warm-start /
                               TS-only アブレーション用)
- load_graph / build_weights : 入力グラフ読み込み(CIM.load_graph 互換)
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from numba import njit, prange


# tabu tenure の周期ステップパターン(15 ブロック, 100 iter ごと)。
# tenure = alpha * pattern[(iter//100) % 15]。{alpha,2a,4a,8a} を巡回。
_TENURE_PATTERN = np.array(
    [1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1], dtype=np.int64
)


# ============================================================
#  グラフ入出力(CIM.load_graph と互換の (n,k,adj,edges))
# ============================================================
def load_graph(filepath: str, return_weights: bool = False):
    """G-set / K2000 形式を読み込む。

    先頭行 "N K"、以降 "a b w"(1-indexed)。w 省略時は +1。
    return_weights=True なら (n, edges, weights) を返す。
    そうでなければ (n, k, adj, edges)。
    """
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    with open(filepath, "r") as f:
        first = f.readline().split()
        n, k = int(first[0]), int(first[1])
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            a = int(parts[0]) - 1
            b = int(parts[1]) - 1
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            edges.append((a, b))
            weights.append(w)
    if return_weights:
        return n, edges, weights
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    return n, k, adj, edges


@njit(cache=True)
def _build_csr(n, edge_a, edge_b, edge_w):
    """無向 CSR を構築。各頂点の隣接(indices)と辺重み(data)。"""
    e = edge_a.shape[0]
    deg = np.zeros(n, dtype=np.int64)
    for k in range(e):
        deg[edge_a[k]] += 1
        deg[edge_b[k]] += 1
    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]
    nnz = indptr[n]
    indices = np.zeros(nnz, dtype=np.int64)
    data = np.zeros(nnz, dtype=np.float64)
    cursor = indptr[:-1].copy()
    for k in range(e):
        a = edge_a[k]
        b = edge_b[k]
        w = edge_w[k]
        indices[cursor[a]] = b
        data[cursor[a]] = w
        cursor[a] += 1
        indices[cursor[b]] = a
        data[cursor[b]] = w
        cursor[b] += 1
    return indptr, indices, data


@njit(cache=True)
def _init_gains(n, indptr, indices, data, s):
    """全頂点の移動利得 Δ と現在のカット値を初期化。

    Δ_v = Σ_{u∈N(v)} w_vu * (+1 if s_u==s_v else -1)
    cut = Σ_{(i,j)} w * [s_i != s_j]  (= 各辺 1 回でカウント)
    """
    gains = np.zeros(n, dtype=np.float64)
    cut = 0.0
    for v in range(n):
        g = 0.0
        sv = s[v]
        for p in range(indptr[v], indptr[v + 1]):
            u = indices[p]
            w = data[p]
            if s[u] == sv:
                g += w
            else:
                g -= w
                cut += w * 0.5  # 辺を 2 回見るので 0.5
        gains[v] = g
    return gains, cut


@njit(cache=True, inline="always")
def _apply_flip(v, s, gains, indptr, indices, data):
    """頂点 v をフリップし、自身と隣接の利得を増分更新。s は更新される。"""
    sv_old = s[v]
    gains[v] = -gains[v]
    for p in range(indptr[v], indptr[v + 1]):
        u = indices[p]
        w = data[p]
        if s[u] == sv_old:
            gains[u] -= 2.0 * w
        else:
            gains[u] += 2.0 * w
    s[v] = 1 - sv_old


@njit(cache=True)
def _tabu_search_one(
    n, indptr, indices, data, s, ts_iters, cr, gamma_pert, alpha_tenure
):
    """単一解 s(0/1, 破壊的に更新される作業コピー)に対する Tabu Search。

    動的 tenure・aspiration・摂動付き。best 解と best カットを返す。
    """
    gains, cut = _init_gains(n, indptr, indices, data, s)
    best_s = s.copy()
    best_cut = cut
    tabu_until = np.zeros(n, dtype=np.int64)
    no_improve = 0

    for it in range(ts_iters):
        tenure = alpha_tenure * _TENURE_PATTERN[(it // 100) % 15]

        # 非 tabu の最大利得 + aspiration(tabu でも best 更新なら可)
        best_v = -1
        best_g = -1.0e18
        asp_v = -1
        asp_g = -1.0e18
        for v in range(n):
            g = gains[v]
            if it >= tabu_until[v]:
                if g > best_g:
                    best_g = g
                    best_v = v
            else:
                # tabu。best 更新を満たすなら aspiration 候補
                if cut + g > best_cut and g > asp_g:
                    asp_g = g
                    asp_v = v

        # aspiration が非 tabu 最良を上回るなら採用
        if asp_v >= 0 and asp_g > best_g:
            mv = asp_v
            mg = asp_g
        else:
            mv = best_v
            mg = best_g

        if mv < 0:
            break  # 全頂点 tabu(理論上ほぼ起きない)

        _apply_flip(mv, s, gains, indptr, indices, data)
        cut += mg
        tabu_until[mv] = it + tenure

        if cut > best_cut:
            best_cut = cut
            best_s = s.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cr:
                # 摂動: gamma_pert 個ランダムにフリップして現在解を撹乱
                for _ in range(gamma_pert):
                    rv = np.random.randint(0, n)
                    _apply_flip(rv, s, gains, indptr, indices, data)
                    cut += gains[rv] * 0.0  # gain は更新済み; cut は再計算が安全
                # 摂動後は cut を厳密再計算(増分の積み重ね誤差回避)
                _, cut = _init_gains(n, indptr, indices, data, s)
                # gains も作り直し
                gains2, _ = _init_gains(n, indptr, indices, data, s)
                for i in range(n):
                    gains[i] = gains2[i]
                no_improve = 0

    return best_s, best_cut


@njit(cache=True, parallel=True)
def _tabu_search_batch(
    n, indptr, indices, data, s_batch, ts_iters, cr, gamma_pert,
    alpha_tenure, seeds
):
    """初期解バッチ s_batch (num, n) を各々 TS で磨く(trial 並列)。"""
    num = s_batch.shape[0]
    out_signs = np.zeros((num, n), dtype=np.int8)
    out_cuts = np.zeros(num, dtype=np.float64)
    for t in prange(num):
        np.random.seed(seeds[t])
        s = s_batch[t].copy()
        bs, bc = _tabu_search_one(
            n, indptr, indices, data, s, ts_iters, cr, gamma_pert, alpha_tenure
        )
        for i in range(n):
            out_signs[t, i] = bs[i]
        out_cuts[t] = bc
    return out_signs, out_cuts


@njit(cache=True)
def _cut_of(n, indptr, indices, data, s):
    """解 s のカット値(検証・スコア用)。"""
    cut = 0.0
    for v in range(n):
        sv = s[v]
        for p in range(indptr[v], indptr[v + 1]):
            if s[indices[p]] != sv:
                cut += data[p]
    return cut * 0.5


@njit(cache=True)
def _aligned_overlap(s1, s2, n):
    """ラベル対称性込みの一致サイズ s(I1,I2)=max(同ラベル一致, 反転一致)。"""
    same = 0
    for i in range(n):
        if s1[i] == s2[i]:
            same += 1
    diff = n - same
    return same if same >= diff else diff


@njit(cache=True)
def _pool_min_distances(pop, n, p):
    """各個体の集団内最小ハミング距離 D_i = min_{j≠i} (n − aligned_overlap)。"""
    D = np.full(p, 1.0e18)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            d = n - _aligned_overlap(pop[i], pop[j], n)
            if d < D[i]:
                D[i] = d
    return D


# ============================================================
#  グルーピング交叉(Python; 1 世代に 1 回 / run なので軽量)
# ============================================================
def _grouping_crossover(sa: np.ndarray, sb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """共通分割継承(grouping)交叉。両親の合意した側を継承し、残りを貪欲割当て。

    sa, sb: int8 (n,) スピン(0/1)。返り値も int8 (n,)。
    ラベル対称性を考慮して sb を整列(0/1 反転して一致を最大化)。
    """
    n = sa.shape[0]
    agree = int(np.sum(sa == sb))
    if agree < n - agree:
        sb = 1 - sb  # ラベル反転で一致を最大化

    child = np.full(n, -1, dtype=np.int8)
    same = sa == sb
    child[same] = sa[same]  # 合意した核を継承
    unassigned = np.where(~same)[0]
    if unassigned.size > 0:
        # 残りはランダム順に、交互に側を割り当て(両親から等距離 + 多様性)
        rng.shuffle(unassigned)
        side = 0
        for v in unassigned:
            child[v] = side
            side ^= 1
    return child


# ============================================================
#  DisQual プール更新(Python)
# ============================================================
def _pool_update(pop: np.ndarray, fits: np.ndarray, child: np.ndarray,
                 child_fit: float, n: int, beta: float) -> bool:
    """子を仮に加え、score 最小の個体を捨てる。置換したら True。

    score(i) = beta * norm(fit_i) + (1-beta) * norm(D_i)
    norm(y) = (y - ymin) / (ymax - ymin + 1)
    """
    p = pop.shape[0]
    ext = np.empty((p + 1, n), dtype=np.int8)
    ext[:p] = pop
    ext[p] = child
    ext_fit = np.empty(p + 1)
    ext_fit[:p] = fits
    ext_fit[p] = child_fit

    D = _pool_min_distances(ext, n, p + 1)
    fmin, fmax = ext_fit.min(), ext_fit.max()
    dmin, dmax = D.min(), D.max()
    nf = (ext_fit - fmin) / (fmax - fmin + 1.0)
    nd = (D - dmin) / (dmax - dmin + 1.0)
    score = beta * nf + (1.0 - beta) * nd

    worst = int(np.argmin(score))
    if worst == p:
        return False  # 子が最弱 → 捨てる
    pop[worst] = child
    fits[worst] = child_fit
    return True


# ============================================================
#  公開 API: メメティック実行(num_trials 独立)
# ============================================================
def simulate_ga_batch(
    n: int,
    edges,
    weights: Optional[list],
    num_trials: int,
    *,
    pop_size: int = 10,
    max_generations: int = 200,
    ts_iters: int = 20000,
    cr: int = 3000,
    gamma_pert: Optional[int] = None,
    alpha_tenure: int = 15,
    beta_quality: float = 0.6,
    time_budget: Optional[float] = None,
    init_signs: Optional[np.ndarray] = None,
    seeds: Optional[np.ndarray] = None,
):
    """メメティックアルゴリズムを num_trials 個、独立に実行。

    各 trial は pop_size の集団を持ち、毎世代 1 子を交叉→TS→DisQual で更新。
    num_trials 個の子は 1 回の njit prange でまとめて磨く(並列)。

    Args:
        max_generations : 世代数(主要な計算量ノブ。anytime sweep で振る)
        ts_iters        : TS 1 回あたりの反復数
        gamma_pert      : 摂動フリップ数。None なら max(50, 0.05*n)
        time_budget     : 秒。指定時はこの実時間で世代ループを打ち切る
        init_signs      : (num_trials, n) or (n,) の warm-start 初期解(±1 or 0/1)。
                          各 run の集団 1 個体目に投入。ハイブリッド用。
        seeds           : (num_trials,) 乱数シード

    Returns:
        best_cuts  : (num_trials,)
        best_signs : (num_trials, n) bool
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    indptr, indices, data = _build_csr(n, edge_a, edge_b, edge_w)

    if seeds is None:
        seeds = np.arange(num_trials, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    if gamma_pert is None:
        gamma_pert = max(50, int(0.05 * n))

    # warm-start 初期解の整形(0/1 へ)
    warm = None
    if init_signs is not None:
        warm = np.asarray(init_signs)
        if warm.ndim == 1:
            warm = np.tile(warm, (num_trials, 1))
        warm = (warm > 0).astype(np.int8)

    rngs = [np.random.default_rng(int(s)) for s in seeds]

    # --- 初期集団生成 + TS 磨き(全 trial×pop をまとめて prange) ---
    init_batch = np.zeros((num_trials * pop_size, n), dtype=np.int8)
    for t in range(num_trials):
        for j in range(pop_size):
            idx = t * pop_size + j
            if warm is not None and j == 0:
                init_batch[idx] = warm[t]
            else:
                init_batch[idx] = (rngs[t].random(n) < 0.5).astype(np.int8)
    init_seeds = (seeds.repeat(pop_size) + np.arange(num_trials * pop_size)) % (2**31)
    signs0, cuts0 = _tabu_search_batch(
        n, indptr, indices, data, init_batch, ts_iters, cr, gamma_pert,
        alpha_tenure, init_seeds.astype(np.int64)
    )
    pops = signs0.reshape(num_trials, pop_size, n).copy()
    fits = cuts0.reshape(num_trials, pop_size).copy()

    # 各 run の best
    best_cuts = fits.max(axis=1).copy()
    best_signs = np.zeros((num_trials, n), dtype=np.int8)
    for t in range(num_trials):
        best_signs[t] = pops[t, int(np.argmax(fits[t]))]

    t0 = time.perf_counter()
    gen = 0
    while gen < max_generations:
        # 各 run で交叉して子を 1 個作る
        children = np.zeros((num_trials, n), dtype=np.int8)
        for t in range(num_trials):
            ia, ib = rngs[t].choice(pop_size, size=2, replace=False)
            children[t] = _grouping_crossover(pops[t, ia], pops[t, ib], rngs[t])
        child_seeds = ((seeds + 7919 * (gen + 1)) % (2**31)).astype(np.int64)
        # 子をまとめて TS 磨き
        cs, cc = _tabu_search_batch(
            n, indptr, indices, data, children, ts_iters, cr, gamma_pert,
            alpha_tenure, child_seeds
        )
        # DisQual で各 run のプール更新
        for t in range(num_trials):
            _pool_update(pops[t], fits[t], cs[t], cc[t], n, beta_quality)
            if cc[t] > best_cuts[t]:
                best_cuts[t] = cc[t]
                best_signs[t] = cs[t]
        gen += 1
        if time_budget is not None and (time.perf_counter() - t0) >= time_budget:
            break

    return best_cuts, best_signs.astype(bool)


# ============================================================
#  公開 API: TS-only 磨き(warm-start ハイブリッド / アブレーション)
# ============================================================
def tabu_refine_batch(
    n: int,
    edges,
    weights: Optional[list],
    init_signs: np.ndarray,
    *,
    ts_iters: int = 50000,
    cr: int = 3000,
    gamma_pert: Optional[int] = None,
    alpha_tenure: int = 15,
    seeds: Optional[np.ndarray] = None,
):
    """与えた初期解 init_signs (num,n) を Tabu Search のみで磨く。

    ハイブリッド(CIM/CAC → TS)や、GA の「局所探索のみ」アブレーション用。
    init_signs は ±1 でも 0/1 でも可。
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    indptr, indices, data = _build_csr(n, edge_a, edge_b, edge_w)

    warm = np.asarray(init_signs)
    if warm.ndim == 1:
        warm = warm[None, :]
    s_batch = (warm > 0).astype(np.int8).copy()
    num = s_batch.shape[0]
    if seeds is None:
        seeds = np.arange(num, dtype=np.int64)
    seeds = np.asarray(seeds, dtype=np.int64)
    if gamma_pert is None:
        gamma_pert = max(50, int(0.05 * n))

    signs, cuts = _tabu_search_batch(
        n, indptr, indices, data, s_batch, ts_iters, cr, gamma_pert,
        alpha_tenure, seeds
    )
    return cuts, signs.astype(bool)
