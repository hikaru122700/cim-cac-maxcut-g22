"""バケット方式の Tabu Search(GA.py の _tabu_search_one の高速版)。

GA.py の走査版は毎反復で全頂点を走査し、同点の手ごとに乱数を引く。利得最大の
手が多数ある疎なグラフでは、これが遅さの原因になる(2026-10-16 の修正後で約 8 倍)。

ここでは利得の値ごとに非 tabu の頂点を配列(バケット)で管理する。
  - 非 tabu の最大利得の手:最上位の空でないバケットから等確率に 1 つ選ぶ(乱数 1 回)
  - tabu の頂点:バケットから外し、tabu リストで管理する。期限が来たら戻す。
    aspiration(tabu でも best を更新する手)は tabu リストだけを見る。
  - best 解のコピー:best から離れる手の直前にだけ取る(結果は走査版と同じ)

選ぶ規則は走査版の random_ties=True と同じ(非 tabu の最大利得の手から等確率、
aspiration は利得最大・同点なら番号最小)。乱数の引き方が違うので、同じ seed でも
走査版とは軌跡が一致しない。利得が整数になるグラフ(重みが整数)専用。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# GA.py と同じ動的 tenure のパターン
_TENURE_PATTERN = np.array([1, 2, 1, 4, 1, 2, 1, 8, 1, 2, 1, 4, 1, 2, 1], dtype=np.int64)


@njit(cache=True)
def max_weighted_degree(n, indptr, data):
    """Σ_u |w_vu| の最大値(利得の絶対値の上限)。"""
    out = 0.0
    for v in range(n):
        acc = 0.0
        for p in range(indptr[v], indptr[v + 1]):
            acc += abs(data[p])
        if acc > out:
            out = acc
    return int(out + 0.5)


@njit(cache=True)
def _init_gains(n, indptr, indices, data, s):
    gains = np.zeros(n, dtype=np.float64)
    cut = 0.0
    for v in range(n):
        g = 0.0
        sv = s[v]
        for p in range(indptr[v], indptr[v + 1]):
            if s[indices[p]] == sv:
                g += data[p]
            else:
                g -= data[p]
                cut += data[p] * 0.5
        gains[v] = g
    return gains, cut


@njit(cache=True, inline="always")
def _insert(v, b, bucket, size, pos, where):
    k = size[b]
    bucket[b, k] = v
    pos[v] = k
    size[b] = k + 1
    where[v] = b


@njit(cache=True, inline="always")
def _remove(v, bucket, size, pos, where):
    b = where[v]
    k = pos[v]
    last = size[b] - 1
    u = bucket[b, last]
    bucket[b, k] = u
    pos[u] = k
    size[b] = last
    where[v] = -1


@njit(cache=True)
def _rebuild(n, gains, offset, in_tabu, bucket, size, pos, where):
    """非 tabu の全頂点をバケットに入れ直し、最上位バケットの番号を返す。"""
    size[:] = 0
    top = -1
    for v in range(n):
        where[v] = -1
        if not in_tabu[v]:
            b = int(gains[v] + offset + 0.5)
            _insert(v, b, bucket, size, pos, where)
            if b > top:
                top = b
    return top


@njit(cache=True)
def _count_violations(n, it, gains, offset, tabu_until, in_tabu, where, size, top,
                      best_g, cut, best_cut, asp_v):
    """検査用:バケットの状態を全走査の結果と照合し、食い違いの数を返す。"""
    bad = 0
    scan_max = -1.0e18
    n_ties = 0
    scan_asp_v = -1
    scan_asp_g = -1.0e18
    for v in range(n):
        g = gains[v]
        tabu = it < tabu_until[v]
        if tabu != in_tabu[v]:
            bad += 1
        if tabu:
            if where[v] != -1:
                bad += 1
            if cut + g > best_cut and g > scan_asp_g:
                scan_asp_g = g
                scan_asp_v = v
        else:
            if where[v] != int(g + offset + 0.5):
                bad += 1
            if g > scan_max:
                scan_max = g
                n_ties = 1
            elif g == scan_max:
                n_ties += 1
    if n_ties > 0:
        if top < 0 or scan_max != best_g or size[top] != n_ties:
            bad += 1
    elif top != -1:
        bad += 1
    if scan_asp_v != asp_v:
        bad += 1
    return bad


@njit(cache=True)
def tabu_search_one_bucket(n, indptr, indices, data, s, ts_iters, cr, gamma_pert,
                           alpha_tenure, offset, check):
    """単一解 s(0/1、破壊的に更新)に対する TS。(best 解, best cut, 検査の食い違い数)。"""
    gains, cut = _init_gains(n, indptr, indices, data, s)
    nb = 2 * offset + 1
    bucket = np.empty((nb, n), dtype=np.int32)
    size = np.zeros(nb, dtype=np.int64)
    pos = np.empty(n, dtype=np.int64)
    where = np.full(n, -1, dtype=np.int64)
    tabu_until = np.zeros(n, dtype=np.int64)
    in_tabu = np.zeros(n, dtype=np.bool_)
    tabu_list = np.empty(n, dtype=np.int64)
    n_tabu = 0
    top = _rebuild(n, gains, offset, in_tabu, bucket, size, pos, where)

    best_s = s.copy()
    best_cut = cut
    pending_best = False  # s が best で、まだ best_s にコピーしていない
    no_improve = 0
    violations = 0

    for it in range(ts_iters):
        tenure = alpha_tenure * _TENURE_PATTERN[(it // 100) % 15]

        # 期限の来た tabu を戻しつつ、aspiration の候補を探す
        asp_v = -1
        asp_g = -1.0e18
        k = 0
        while k < n_tabu:
            v = tabu_list[k]
            if it >= tabu_until[v]:
                n_tabu -= 1
                tabu_list[k] = tabu_list[n_tabu]
                in_tabu[v] = False
                b = int(gains[v] + offset + 0.5)
                _insert(v, b, bucket, size, pos, where)
                if b > top:
                    top = b
                continue
            g = gains[v]
            if cut + g > best_cut and (g > asp_g or (g == asp_g and v < asp_v)):
                asp_g = g
                asp_v = v
            k += 1

        best_v = -1
        best_g = -1.0e18
        if top >= 0:
            best_g = float(top - offset)
            m = size[top]
            r = np.random.randint(0, m) if m > 1 else 0
            best_v = bucket[top, r]

        if check:
            violations += _count_violations(n, it, gains, offset, tabu_until, in_tabu,
                                            where, size, top, best_g, cut, best_cut, asp_v)

        if asp_v >= 0 and asp_g > best_g:
            mv = asp_v
            mg = asp_g
        else:
            mv = best_v
            mg = best_g
        if mv < 0:
            break

        if pending_best and not (cut + mg > best_cut):
            best_s[:] = s
            pending_best = False

        if where[mv] >= 0:
            _remove(mv, bucket, size, pos, where)
        sv_old = s[mv]
        gains[mv] = -gains[mv]
        for p in range(indptr[mv], indptr[mv + 1]):
            u = indices[p]
            if s[u] == sv_old:
                gains[u] -= 2.0 * data[p]
            else:
                gains[u] += 2.0 * data[p]
            if where[u] >= 0:
                _remove(u, bucket, size, pos, where)
                b = int(gains[u] + offset + 0.5)
                _insert(u, b, bucket, size, pos, where)
                if b > top:
                    top = b
        s[mv] = 1 - sv_old
        cut += mg
        tabu_until[mv] = it + tenure
        if not in_tabu[mv]:
            in_tabu[mv] = True
            tabu_list[n_tabu] = mv
            n_tabu += 1
        while top >= 0 and size[top] == 0:
            top -= 1

        if cut > best_cut:
            best_cut = cut
            pending_best = True
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cr:
                # 摂動:gamma_pert 個をランダムに反転し、利得と cut を厳密に再計算
                for _ in range(gamma_pert):
                    rv = np.random.randint(0, n)
                    s[rv] = 1 - s[rv]
                gains, cut = _init_gains(n, indptr, indices, data, s)
                top = _rebuild(n, gains, offset, in_tabu, bucket, size, pos, where)
                no_improve = 0

    if pending_best:
        best_s[:] = s
    return best_s, best_cut, violations


@njit(cache=True, parallel=True)
def tabu_search_batch_bucket(n, indptr, indices, data, s_batch, ts_iters, cr, gamma_pert,
                             alpha_tenure, seeds, offset, check):
    """初期解バッチ (num, n) を各々 TS で磨く(trial 並列)。GA._tabu_search_batch と同じ seed 規約。"""
    num = s_batch.shape[0]
    out_signs = np.zeros((num, n), dtype=np.int8)
    out_cuts = np.zeros(num, dtype=np.float64)
    out_bad = np.zeros(num, dtype=np.int64)
    for t in prange(num):
        np.random.seed(seeds[t])
        s = s_batch[t].copy()
        bs, bc, bad = tabu_search_one_bucket(n, indptr, indices, data, s, ts_iters, cr,
                                             gamma_pert, alpha_tenure, offset, check)
        for i in range(n):
            out_signs[t, i] = bs[i]
        out_cuts[t] = bc
        out_bad[t] = bad
    return out_signs, out_cuts, out_bad
