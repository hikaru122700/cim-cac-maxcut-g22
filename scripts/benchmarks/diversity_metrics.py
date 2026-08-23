"""diversity_metrics.py — MAX-CUT 解集合の「多様性」指標。

MAX-CUT の解は分割(頂点の 2 彩色)なので、**全体反転 s -> -s は同一解**。
すべての距離はこの対称性を畳んだ形で定義する:

    d(a, b) = min(H(a, b), n - H(a, b)) / n      (H = ハミング距離)

d = 0 なら同一分割、d = 0.5 が最大(無相関)。ランダム 2 分割どうしの期待値は
n が大きいとき約 0.5 - O(1/sqrt(n))(反転対称で min を取るぶん僅かに下がる)。

提供する指標:
  - mean_pairwise / min_pairwise / max_pairwise : ペアワイズ距離統計
  - n_distinct        : 相異なる分割の個数(全体反転を同一視)
  - n_distinct_cuts   : 相異なるカット値の個数
  - frozen_frac       : 全試行でほぼ同じ側に固定される頂点の割合(|m_i| > thr)
  - entropy_bits      : 頂点あたりの平均二値エントロピー [bit](1 = 完全にランダム)
  - near_* : BKS 近傍解(gap <= near_gap_pct)だけに絞った多様性
"""
from __future__ import annotations

import numpy as np


def to_binary(signs) -> np.ndarray:
    """(T, n) の符号/bool/0-1 配列を 0/1 の int8 に正規化する。"""
    a = np.asarray(signs)
    if a.dtype == bool:
        b = a.astype(np.int8)
    else:
        b = (a > 0).astype(np.int8)
    return np.ascontiguousarray(b)


def hamming_matrix(S: np.ndarray) -> np.ndarray:
    """(T, n) 0/1 行列の生ハミング距離行列 (T, T)。"""
    X = S.astype(np.float64)
    Y = 1.0 - X
    H = X @ Y.T + Y @ X.T
    return H


def distance_matrix(S: np.ndarray) -> np.ndarray:
    """反転対称を畳んだ正規化距離行列 (T, T)、値域 [0, 0.5]。"""
    n = S.shape[1]
    H = hamming_matrix(S)
    D = np.minimum(H, n - H) / float(n)
    np.fill_diagonal(D, 0.0)
    return D


def cross_distance_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """異なる集合 A(Ta, n), B(Tb, n) 間の反転対称正規化距離 (Ta, Tb)。"""
    n = A.shape[1]
    Xa, Ya = A.astype(np.float64), 1.0 - A.astype(np.float64)
    Xb, Yb = B.astype(np.float64), 1.0 - B.astype(np.float64)
    H = Xa @ Yb.T + Ya @ Xb.T
    return np.minimum(H, n - H) / float(n)


def canonicalize(S: np.ndarray) -> np.ndarray:
    """全体反転の自由度を潰す(先頭要素が 0 になるように揃える)。"""
    flip = S[:, 0] == 1
    C = S.copy()
    C[flip] = 1 - C[flip]
    return C


def align_to_reference(S: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """各行を、参照解 ref に近い方の反転に揃える。"""
    n = S.shape[1]
    h = (S != ref[None, :]).sum(axis=1)
    flip = h > (n - h)
    A = S.copy()
    A[flip] = 1 - A[flip]
    return A


def count_distinct(S: np.ndarray) -> int:
    C = canonicalize(S)
    return int(np.unique(C, axis=0).shape[0])


def summarize(signs, cuts, bks: float, *,
              frozen_thr: float = 0.9,
              near_gap_pct: float = 0.5) -> dict:
    """1 手法 × 1 インスタンスの解集合をまとめて要約する。

    Parameters
    ----------
    signs : (T, n) 解(bool / ±1 / 0-1 いずれでも可)
    cuts  : (T,)   統一採点済みカット値
    bks   : 既知ベスト
    frozen_thr : |磁化| がこれ以上なら「固定頂点」とみなす
    near_gap_pct : BKS からの相対 gap[%] がこれ以下を「近傍最適解」とする
    """
    S = to_binary(signs)
    c = np.asarray(cuts, dtype=float)
    T, n = S.shape

    D = distance_matrix(S)
    iu = np.triu_indices(T, k=1)
    dv = D[iu]

    ref = S[int(np.argmax(c))]
    A = align_to_reference(S, ref)
    p = A.mean(axis=0)                      # 各頂点が「1 側」に来る割合
    m = 2.0 * p - 1.0                       # 磁化 [-1, 1]
    eps = 1e-12
    ent = -(p * np.log2(p + eps) + (1 - p) * np.log2(1 - p + eps))

    gap_pct = (bks - c) / bks * 100.0
    near = gap_pct <= near_gap_pct
    out = {
        "num_trials": int(T),
        "n": int(n),
        "cut_mean": float(c.mean()),
        "cut_max": float(c.max()),
        "cut_min": float(c.min()),
        "cut_std": float(c.std()),
        "gap_pct_mean": float(gap_pct.mean()),
        "gap_pct_best": float(gap_pct.min()),
        "mean_pairwise": float(dv.mean()),
        "median_pairwise": float(np.median(dv)),
        "min_pairwise": float(dv.min()),
        "max_pairwise": float(dv.max()),
        "n_distinct": count_distinct(S),
        "n_distinct_cuts": int(np.unique(c).size),
        "frozen_frac": float((np.abs(m) >= frozen_thr).mean()),
        "entropy_bits": float(ent.mean()),
        "near_gap_pct": float(near_gap_pct),
        "n_near": int(near.sum()),
    }
    if near.sum() >= 2:
        Dn = D[np.ix_(near, near)]
        iun = np.triu_indices(int(near.sum()), k=1)
        out["near_mean_pairwise"] = float(Dn[iun].mean())
        out["near_n_distinct"] = count_distinct(S[near])
    else:
        out["near_mean_pairwise"] = float("nan")
        out["near_n_distinct"] = int(near.sum())
    return out


def classical_mds(D: np.ndarray, dim: int = 2) -> np.ndarray:
    """距離行列 D (N, N) の古典的 MDS 埋め込み (N, dim)。"""
    N = D.shape[0]
    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J
    B = (B + B.T) / 2.0
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:dim]
    w = np.clip(w[idx], 0.0, None)
    return V[:, idx] * np.sqrt(w)[None, :]
