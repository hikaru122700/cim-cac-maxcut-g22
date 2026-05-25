"""
Coherent Ising Machine (CIM) シミュレータ - 進行波モデル (2026-05-25 改良版)

baseline (modules/CIM.py) との差分:
  - trial 末で CIM 出力 (= 連続力学による 1-flip 局所最適) に対し
    Iterated Local Search (ILS) を実行して脱出させる:
      1. 1-flip greedy で念のため磨く
      2. K 回の「ランダム k-flip 摂動 → 1-flip 再 polish」を反復、best を採用
  - random sparse G22 (avg deg ≈ 20) では CIM 解の周辺に improving plateau が
    隠れているため、わずか数十回の摂動で +50〜+80 cuts を回収できる
    (目標: 100 trial 平均 ≥ 13340)
  - 既存 API (`_simulate_cim_batch`, `simulate_cim_batch`) は温存し、
    新規に `_simulate_cim_batch_polished`, `simulate_cim_batch_polished` を追加

論文: Inoue & Yoshida,
  "Traveling-wave model of coherent Ising machine based on fiber loop with
   pulse-pumped phase-sensitive amplifier",
  Optics Communications 522 (2022) 128642.

========================================================================
【CIM の物理的イメージ】
ファイバーのループ(周回路)の中を、m個の光パルスが順番に周回している。
ループの途中には PSA(位相感応増幅器) があり、パルスが通過するたびに
「in-phase 成分(c_i)」は増幅、「quadrature 成分(s_i)」は減衰する。
さらに MFB(測定フィードバック) によって、パルス同士は結合行列 J_ij を通じて
互いに影響し合う。
最終的に各パルスの in-phase 振幅は正 or 負 の 2値に収束し、
その符号が Ising スピン(= 0/1 割り当て)を表す。
MAX-CUT 問題は J_ij を辺として与え、反強磁性結合(J < 0)にすることで、
「異なるクラスタに属するペアの数の最大化」に対応する。
========================================================================

【主要パラメータ (論文 Section 3)】
  κ = 130 W^(-1/2) m^(-1)  非線形定数(PSAの利得係数を決める)
  L = 5 cm                  PSA媒質長
  γ = 42.09 W^(-1)          飽和係数(信号が強いほど利得が飽和)
  loop loss = 11 dB         ループ全体の損失 → η = 10^(-1.1) ≈ 0.0794
  BW = 1 GHz                システム帯域(ノイズ分散 Eq.6 に乗算)
  dP/round = 0.05 mW        毎ラウンドでポンプパワーを増やす量
  J_ij ∈ {0, -0.03}         G22 の結合係数(辺なら -0.03、なければ 0)
"""

import numpy as np
import wandb
from numba import njit, prange
from scipy.sparse import csr_matrix
from scipy.sparse._sparsetools import csr_matvec

from .verify import compute_cut_from_edges, run_all_checks


# ============================================================
#  Numba JIT 版のコアループ (wandb_log=False 時に使用)
# ============================================================
# 内ループ全体を ahead-of-time コンパイルしてネイティブコード化する。
# - numpy の ufunc ディスパッチオーバーヘッドを全部スキップ
# - scipy.sparse のラッパー呼び出しもバイパス
# - 1 パルスあたりの演算を融合ループ化して中間配列のアロケーションも削減
# 初回は JIT コンパイルに数秒かかるが、2回目以降は cache (.numba_cache) から復元される。
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_batch(
    n: int,
    num_rounds: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,    # MAX-CUT 重み(未指定なら +1 で埋める)
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
):
    """CIM シミュレーションを num_trials 分まとめて並列実行する(Numba JIT)。

    prange により trial 単位で CPU コアに分散。
    各 trial は独立なので競合なく並列化できる。
    num_trials=1 で呼べば単発実行にもなる。
    cut は edge_w 重み付き(unweighted の場合 edge_w=+1)。
    """
    # 出力バッファ
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)

    # 事前計算可能な定数(全 trial 共通)
    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    # ---- trial ごとに並列実行 ----
    for trial_idx in prange(num_trials):
        # 各 thread で独立な乱数状態(numba は thread-local RNG)
        np.random.seed(seeds[trial_idx])

        # trial ごとに独立な状態ベクトル
        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18

        for k in range(num_rounds):
            # Step 1: ポンプパワー → 非飽和利得係数 g_0
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            # Step 2: Sparse matvec Jc = J @ c (CSR 手書きループ)
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            # Step 3-5: coupled_in → I_in → sqrt(G_I) → noise → c を 1ループで融合
            for i in range(n):
                coupled_in_i = sqrt_eta * c[i] + Jc[i]
                I_in_i = coupled_in_i * coupled_in_i
                half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                sqrt_G_I_i = np.exp(half_g_i)
                noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                c[i] = sqrt_G_I_i * coupled_in_i + noise_i

            # Step 6: 重み付き cut 計算
            cut = 0.0
            for e in range(num_edges):
                if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                    cut += edge_w[e]

            # ベスト更新
            if cut > best_cut:
                best_cut = cut
                for i in range(n):
                    best_signs[i] = c[i] > 0.0

        # 結果を出力バッファに格納
        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


# ============================================================
#  1-flip greedy 局所最適化 (njit ヘルパー)
# ============================================================
# CIM の符号配列に対し、Δcut[i] を維持しつつ argmax(Δcut) を貪欲に
# flip し続け、これ以上正の改善 flip が無い (= 1-flip 局所最適) まで磨く。
#
# 計算量:
#   - 初期 Δ の計算: O(K) (K = 辺数)
#   - 1 flip あたり: argmax O(n) + 隣接更新 O(deg)
#   - 典型 flip 回数: G22 で 50〜200 程度 (実測)
#   - 全体: ~1 trial で 1〜2 ms 程度 (3.3 秒の CIM 本体に比べ無視できる)
#
# 数学:
#   Δcut[i] = Σ_j w_ij · (+1 if spin[i]==spin[j] else -1)
#   spin[v] を flip すると:
#     - Δcut[v]      ← -Δcut[v]
#     - 各隣接 j について Δcut[j] を ±2·w_vj 更新
#       (after-flip で spin[v]==spin[j] なら +2w、!= なら -2w)
@njit(cache=True, fastmath=True)
def _local_search_1flip(
    spins: np.ndarray,            # (n,) bool, in-place で更新される
    n: int,
    adj_indptr: np.ndarray,       # (n+1,)
    adj_indices: np.ndarray,      # (2K,)
    adj_w: np.ndarray,             # (2K,)
) -> float:
    """spins を 1-flip 局所最適まで磨き、累積 cut 改善量を返す。"""
    # 初期 Δ を計算
    delta = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s_i = spins[i]
        start = adj_indptr[i]
        end = adj_indptr[i + 1]
        acc = 0.0
        for idx in range(start, end):
            j = adj_indices[idx]
            w = adj_w[idx]
            if s_i == spins[j]:
                acc += w
            else:
                acc -= w
        delta[i] = acc

    total_gain = 0.0
    eps = 1e-12  # 浮動小数比較の閾値 (整数重みでも安全)
    while True:
        # 最大 Δ を探す
        best_i = -1
        best_d = eps
        for i in range(n):
            if delta[i] > best_d:
                best_d = delta[i]
                best_i = i
        if best_i < 0:
            break

        # flip 実行
        spins[best_i] = not spins[best_i]
        total_gain += best_d
        s_v = spins[best_i]

        # 隣接 j の Δ を更新
        start = adj_indptr[best_i]
        end = adj_indptr[best_i + 1]
        for idx in range(start, end):
            j = adj_indices[idx]
            w = adj_w[idx]
            if s_v == spins[j]:
                # flip 後同じ側 → 以前は異なる側で -w 寄与 → 今は +w に
                delta[j] += 2.0 * w
            else:
                delta[j] -= 2.0 * w
        # 自分自身の Δ は符号反転 (もう一度 flip すると -best_d 戻る)
        delta[best_i] = -delta[best_i]

    return total_gain


# ============================================================
#  Iterated Local Search (ILS) で 1-flip 局所最適から脱出
# ============================================================
# CIM の連続力学は Ising エネルギーの停留点に収束するため、
# 出力解はほぼ常に 1-flip 局所最適 (素朴な polish では gain=0)。
# 真に cut を押し上げるには「強い摂動 → 再 polish」のループが必要。
#
# 戦略:
#   1. CIM 解を 1-flip で念のため磨く (普通 gain=0)
#   2. K 回反復:
#       a. best_spins から trial_spins = copy()
#       b. trial_spins の k 頂点をランダムに反転 (kick / perturbation)
#       c. trial_spins を 1-flip greedy で local opt まで磨く
#       d. cut が best を超えたら best_spins ← trial_spins
#   3. best_spins と best_cut を返す
#
# 摂動サイズ k は「強すぎず弱すぎず」が重要:
#   k 小 → 同じ盆地に戻る (改善しない)
#   k 大 → 完全ランダム化 (1-flip polish に時間がかかり、性能低下)
#   経験則: k ≈ √N (N=2000 で k ≈ 45) 周辺が効きやすい
#
# 反復数 K: 多いほど精度上がるが時間も増える。CIM 本体 3.3秒に対し
# K=40, k=40 で 1 trial あたり ~50ms 程度。100 trial 並列なら合計 1秒以下。
@njit(cache=True, fastmath=True)
def _compute_cut_jit(
    spins: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,
) -> float:
    cut = 0.0
    num_edges = edge_a.shape[0]
    for e in range(num_edges):
        if spins[edge_a[e]] != spins[edge_b[e]]:
            cut += edge_w[e]
    return cut


@njit(cache=True, fastmath=True)
def _ils_escape(
    spins: np.ndarray,           # (n,) bool, in-place: 最終的に best_spins になる
    n: int,
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    adj_w: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,
    num_iters: int,
    perturb_size: int,
) -> float:
    """ILS で spins を脱出磨きし、最終 cut を返す。"""
    # 初期 polish (CIM 直後の念押し)
    _local_search_1flip(spins, n, adj_indptr, adj_indices, adj_w)
    best_cut = _compute_cut_jit(spins, edge_a, edge_b, edge_w)
    best_spins = spins.copy()

    trial_spins = np.zeros(n, dtype=np.bool_)
    for it in range(num_iters):
        # best_spins から複製してから kick
        for i in range(n):
            trial_spins[i] = best_spins[i]
        for _ in range(perturb_size):
            idx = np.random.randint(0, n)
            trial_spins[idx] = not trial_spins[idx]

        _local_search_1flip(trial_spins, n, adj_indptr, adj_indices, adj_w)
        trial_cut = _compute_cut_jit(trial_spins, edge_a, edge_b, edge_w)

        if trial_cut > best_cut:
            best_cut = trial_cut
            for i in range(n):
                best_spins[i] = trial_spins[i]

    # spins を best_spins に上書きして返す
    for i in range(n):
        spins[i] = best_spins[i]
    return best_cut


# ============================================================
#  Polish 付き batch シミュレーター
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_batch_polished(
    n: int,
    num_rounds: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,
    adj_indptr: np.ndarray,
    adj_indices: np.ndarray,
    adj_w: np.ndarray,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
    ils_iters: int,
    ils_perturb: int,
):
    """CIM + ILS (Iterated Local Search) を num_trials 並列実行。

    本体は _simulate_cim_batch とほぼ同じ。違いは trial 末で
    best_signs に対し ILS (1-flip polish + 摂動再 polish のループ)
    を施し、escape 後の cut と spins を返すこと。
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18

        for k in range(num_rounds):
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            for i in range(n):
                coupled_in_i = sqrt_eta * c[i] + Jc[i]
                I_in_i = coupled_in_i * coupled_in_i
                half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                sqrt_G_I_i = np.exp(half_g_i)
                noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                c[i] = sqrt_G_I_i * coupled_in_i + noise_i

            cut = 0.0
            for e in range(num_edges):
                if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                    cut += edge_w[e]

            if cut > best_cut:
                best_cut = cut
                for i in range(n):
                    best_signs[i] = c[i] > 0.0

        # ---- trial 末 ILS: 1-flip 局所最適から escape して push up ----
        ils_cut = _ils_escape(
            best_signs, n,
            adj_indptr, adj_indices, adj_w,
            edge_a, edge_b, edge_w,
            ils_iters, ils_perturb,
        )

        best_cuts_out[trial_idx] = ils_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


def build_adjacency_csr(
    n: int,
    edges: list,
    weights: list | None = None,
):
    """隣接 CSR (adj_indptr, adj_indices, adj_w) を辺リストから構築する。

    無向グラフなので各辺 (a,b) は両方向に展開する。
    重み未指定なら全辺 w=1.0。
    """
    deg = np.zeros(n, dtype=np.int64)
    for a, b in edges:
        deg[a] += 1
        deg[b] += 1

    indptr = np.zeros(n + 1, dtype=np.int64)
    for i in range(n):
        indptr[i + 1] = indptr[i] + deg[i]
    total = int(indptr[n])

    indices = np.zeros(total, dtype=np.int64)
    adj_w = np.zeros(total, dtype=np.float64)
    cursor = indptr[:n].copy()
    if weights is None:
        for a, b in edges:
            indices[cursor[a]] = b
            adj_w[cursor[a]] = 1.0
            cursor[a] += 1
            indices[cursor[b]] = a
            adj_w[cursor[b]] = 1.0
            cursor[b] += 1
    else:
        for (a, b), w in zip(edges, weights):
            wf = float(w)
            indices[cursor[a]] = b
            adj_w[cursor[a]] = wf
            cursor[a] += 1
            indices[cursor[b]] = a
            adj_w[cursor[b]] = wf
            cursor[b] += 1

    return indptr, indices, adj_w


def load_graph(filepath: str, return_weights: bool = False):
    """Gset 形式のグラフを読み込み、隣接リストと辺リストを返す。

    入力ファイルは 1-indexed だが、Python では配列を 0-indexed で扱いたいので
    ここで -1 して変換している。

    辺の 3 番目の列があれば重み w_ij として解釈する。無ければ +1 とみなす。
    K2000 のような ±1 重み付きインスタンスでは return_weights=True で重みも取得。
    """
    with open(filepath, "r") as f:
        n, k = map(int, f.readline().split())
        adj = [[] for _ in range(n)]
        edges = []
        weights: list[float] = []
        for _ in range(k):
            parts = f.readline().split()
            a, b = int(parts[0]) - 1, int(parts[1]) - 1
            w = float(parts[2]) if len(parts) >= 3 else 1.0
            adj[a].append(b)
            adj[b].append(a)
            edges.append((a, b))
            weights.append(w)
    if return_weights:
        return n, k, adj, edges, weights
    return n, k, adj, edges


def build_coupling_matrix(
    n: int,
    edges: list[tuple[int, int]],
    coupling: float,
    weights: list[float] | None = None,
) -> csr_matrix:
    """結合行列 J を CSR 形式のスパース行列として構築する。

    通常モード (weights=None): すべての辺で J_ij = coupling(従来挙動)。
      G22 (w=+1) なら coupling=-0.03 で J_ij = -0.03 (反強磁性、カット促進)。

    重み付きモード (weights 指定): J_ij = coupling * w_ij。
      K2000 (w ∈ ±1) なら coupling=-1 で J_ij ∈ ±1 になり、論文の SB 設定と一致。
      w=+1 の辺 → J=-|coupling| (cut 促進)、w=-1 の辺 → J=+|coupling| (cut 抑制)。
    """
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    if weights is None:
        for a, b in edges:
            rows.append(a); cols.append(b); data.append(coupling)
            rows.append(b); cols.append(a); data.append(coupling)
    else:
        for (a, b), w in zip(edges, weights):
            val = coupling * w
            rows.append(a); cols.append(b); data.append(val)
            rows.append(b); cols.append(a); data.append(val)
    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)


def amplitudes_to_solution(c: np.ndarray) -> list[int]:
    """in-phase 振幅 c_i の符号から 0/1 の割り当てを決定する。

    c_i > 0  → 集合 B (= 1)
    c_i ≤ 0  → 集合 A (= 0)

    注: 全体の符号を反転しても MAX-CUT のカット数は同じなので、
    どちらを 0/1 に割り当てるかは任意。
    """
    return [1 if ci > 0 else 0 for ci in c]


def simulate_cim(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    rng: np.random.Generator,
    log_interval: int = 10,
    wandb_log: bool = True,
) -> tuple[np.ndarray, int, list[int]]:
    """進行波モデルによる CIM シミュレーションのメインループ。

    ==== 使用する式 (論文より) ====
    Eq.(3):  c_i(k+1) = √G_I,i · (√η · c_i(k) + Σ_j J_ij c_j(k)) + N_I,i
             s_i(k+1) = √(η·G_Q,i) · s_i(k) + N_Q,i
             ↑ 1ラウンド分の発展方程式。
                √η は ループ損失(信号が弱くなる)
                √G_I は PSA 増幅(in-phase を増幅)
                √G_Q は PSA 減衰(quadrature を減衰)
                Σ_j J_ij c_j は MFB による他パルスからの結合入力
                N_I, N_Q は増幅器からの ASE + 真空ノイズ

    Eq.(14): g   = g_0·(1 - γ·I_in)      ← 飽和込みの利得係数
             G_I = exp(g)                  ← in-phase 利得
             G_Q = exp(-g)                 ← quadrature 利得(減衰)
             g_0(k) = 2·κ·√(P_p(k))·L     ← 非飽和時の利得係数

    Eq.(15): I_in ≈ (√η·c_i + Σ_j J_ij·c_j)^2
             ↑ PSA への入力信号強度(飽和項 γ·I_in に使う)
                vacuum ノイズと s_i は強度が小さいので無視している

    Eq.(6):  σ²_I = (2-η)·G_I/4 · BW      ← in-phase ノイズの分散
             σ²_Q = (2-η)·G_Q/4 · BW      ← quadrature ノイズの分散
    """

    # ---- 辺配列の事前計算(cut 評価の高速化) ----
    # 辺リストを numpy 配列化。
    edges_np = np.asarray(edges, dtype=np.int64)  # shape: (K, 2)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    edge_w = np.ones(edges_np.shape[0], dtype=np.float64)  # unweighted

    # ====== Fast path: wandb 出力が不要なら Numba JIT 版を直接呼ぶ ======
    # 内部では _simulate_cim_batch を num_trials=1 で呼び、単発実行にも対応。
    if not wandb_log:
        seed = int(rng.integers(0, 2**63 - 1))
        seeds = np.array([seed], dtype=np.int64)
        best_cuts_out, best_signs_out = _simulate_cim_batch(
            n,
            num_rounds,
            1,
            J.data,
            J.indices,
            J.indptr,
            edge_a,
            edge_b,
            edge_w,
            float(kappa),
            float(L),
            float(gamma),
            float(eta),
            float(bandwidth),
            float(photon_energy),
            float(dP_per_round),
            seeds,
        )
        best_cut = int(best_cuts_out[0])
        best_x: list[int] = best_signs_out[0].astype(np.int64).tolist()
        # c_final は返さない(JIT 内で持っていないため、ゼロベクトルでダミー)
        return np.zeros(n, dtype=np.float64), best_cut, best_x

    # ====== Slow path: wandb 出力あり (単発実行・デバッグ用) ======
    # ---- 初期条件 ----
    # 全パルスは vacuum 状態から始まる → c(0) = 0
    # 最初のノイズ N_I によって自発的に立ち上がっていく。
    # ※ quadrature 成分 s_i は coupled_in や cut 計算に一切関与しないため、
    #   計算を省略(論文 Eq.3b は形式上存在するが、simulation の結果に影響しない)。
    c = np.zeros(n, dtype=np.float64)

    # ---- 事前計算可能な定数 ----
    # ノイズ σ²_I = (2-η)·G_I/4·BW·ℏω → σ_I = noise_const * √G_I と分解できる。
    # 定数部分を1回だけ計算しておき、毎ラウンド sqrt を呼ばない。
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    sqrt_eta = np.sqrt(eta)
    half_dP = 0.5  # sqrt(G_I) = exp(g/2) の係数

    # ---- scipy sparse のラッパーオーバーヘッドを回避 ----
    # 毎ラウンド J @ c を呼ぶと 1500×(250μs wrapper + 170μs 計算) = 630ms かかる。
    # 内部の低レベル csr_matvec を直接呼ぶことでラッパー部分をスキップする。
    # 注意: csr_matvec は「加算」なので、呼ぶ前に出力バッファを必ず 0 にする必要がある。
    J_data = J.data
    J_indices = J.indices
    J_indptr = J.indptr
    Jc = np.zeros(n, dtype=np.float64)

    # ベストカット数と、その時の符号配列を追跡
    best_cut = 0
    best_signs = np.zeros(n, dtype=bool)

    # ---- メインループ ----
    for k in range(num_rounds):
        # Step 1: ポンプパワー → 非飽和利得係数 g_0 (毎ラウンド更新)
        P_p = (k + 1) * dP_per_round
        g0 = 2.0 * kappa * np.sqrt(P_p) * L

        # Step 2: J @ c をバッファに直接書き込む (wrapper オーバーヘッド回避)
        # csr_matvec は accumulator なので事前に 0 クリアが必要
        Jc.fill(0.0)
        csr_matvec(n, n, J_indptr, J_indices, J_data, c, Jc)
        # coupled_in = √η·c + J·c  (Eq.15)
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in

        # Step 3: 利得を計算 (Eq.14)
        # g = g_0·(1 - γ·I_in), sqrt(G_I) = exp(g/2) として直接計算。
        # G_I 自体は不要(σ_I も sqrt(G_I) から出せる)。
        half_g = 0.5 * g0 * (1.0 - gamma * I_in)
        sqrt_G_I = np.exp(half_g)

        # Step 4: ノイズ生成 σ_I = noise_const * sqrt(G_I)
        # rng.standard_normal + 乗算 は rng.normal(0, σ) と同速だがシンプル
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)

        # Step 5: 差分方程式 Eq.3a で振幅更新
        # c(k+1) = sqrt(G_I)·coupled_in + N_I
        c = sqrt_G_I * coupled_in + N_I

        # Step 6: cut 評価 (ベクトル化 + .sum() メソッドで wrapper 回避)
        signs = c > 0
        cut = int((signs[edge_a] != signs[edge_b]).sum())
        if cut > best_cut:
            best_cut = cut
            best_signs = signs.copy()

        # Step 7: wandb ログ (wandb_log=False 時は完全スキップ)
        if wandb_log and ((k + 1) % log_interval == 0 or k == 0):
            # ログ用にだけ G_I と sigma_I を復元
            G_I = sqrt_G_I * sqrt_G_I
            sigma_I = noise_const * sqrt_G_I
            mean_abs_c = float(np.mean(np.abs(c)))
            mean_sigma = float(sigma_I.mean())
            wandb.log({
                "round": k + 1,
                "pump_power_mW": P_p * 1e3,
                "g0": g0,
                "eta_G_I_unsat": eta * float(np.exp(2.0 * half_dP * g0)),
                "mean_abs_c": mean_abs_c,
                "std_c": float(c.std()),
                "mean_I_in": float(I_in.mean()),
                "mean_G_I": float(G_I.mean()),
                "mean_sigma_I": mean_sigma,
                "snr": mean_abs_c / (mean_sigma + 1e-30),
                "current_cut": cut,
                "best_cut": best_cut,
            })

    # 最後に一度だけ bool → int list に変換して返す
    best_x: list[int] = best_signs.astype(np.int64).tolist()
    return c, best_cut, best_x


def simulate_cim_batch(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    num_trials: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """100 trial を並列実行するための公開ラッパー。

    weights が指定された場合、cut は重み付きで計算される。

    Returns:
        best_cuts: shape (num_trials,)      各 trial の最良カット値
        best_signs: shape (num_trials, n)   各 trial の最良解(bool)
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    best_cuts, best_signs = _simulate_cim_batch(
        n,
        num_rounds,
        num_trials,
        J.data,
        J.indices,
        J.indptr,
        edge_a,
        edge_b,
        edge_w,
        float(kappa),
        float(L),
        float(gamma),
        float(eta),
        float(bandwidth),
        float(photon_energy),
        float(dP_per_round),
        seeds_arr,
    )
    return best_cuts, best_signs


def simulate_cim_batch_polished(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    num_trials: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
    weights: list[float] | None = None,
    ils_iters: int = 40,
    ils_perturb: int = 45,
) -> tuple[np.ndarray, np.ndarray]:
    """CIM + ILS (1-flip + 摂動 escape) の公開ラッパー (2026-05-25 改良版)。

    Parameters
    ----------
    ils_iters : int
        ILS の反復回数 (摂動 + 再 polish を何回やるか)
    ils_perturb : int
        1 回の摂動で反転する頂点数。経験則: ≈ √N (G22 で 45 前後)

    Returns:
        best_cuts: shape (num_trials,)     ILS 後の cut
        best_signs: shape (num_trials, n)  ILS 後の符号配列
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    # 隣接 CSR を構築 (polish 用)
    adj_indptr, adj_indices, adj_w = build_adjacency_csr(n, edges, weights)

    best_cuts, best_signs = _simulate_cim_batch_polished(
        n,
        num_rounds,
        num_trials,
        J.data,
        J.indices,
        J.indptr,
        edge_a,
        edge_b,
        edge_w,
        adj_indptr,
        adj_indices,
        adj_w,
        float(kappa),
        float(L),
        float(gamma),
        float(eta),
        float(bandwidth),
        float(photon_energy),
        float(dP_per_round),
        seeds_arr,
        int(ils_iters),
        int(ils_perturb),
    )
    return best_cuts, best_signs


# ============================================================
#  振幅軌跡記録版(チューニング後の可視化用)
# ============================================================
@njit(cache=True, fastmath=True)
def _simulate_cim_trajectory(
    n: int,
    num_rounds: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    edge_a: np.ndarray,
    edge_b: np.ndarray,
    edge_w: np.ndarray,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seed: np.int64,
    sample_rounds: np.ndarray,   # 昇順、記録対象の round index (0-indexed)
):
    """単一 trial の CIM を回し、指定ラウンドで c(k) と cut(k) を記録する。

    Returns
    -------
    c_history : (num_samples, n) 各サンプル時点の振幅
    cut_history : (num_samples,) 各サンプル時点の cut
    best_cut : float
    best_signs : (n,) bool
    """
    num_edges = edge_a.shape[0]
    num_samples = sample_rounds.shape[0]
    c_history = np.zeros((num_samples, n), dtype=np.float64)
    cut_history = np.zeros(num_samples, dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)

    np.random.seed(seed)
    c = np.zeros(n, dtype=np.float64)
    Jc = np.zeros(n, dtype=np.float64)
    best_signs = np.zeros(n, dtype=np.bool_)
    best_cut = -1.0e18
    sample_idx = 0

    for k in range(num_rounds):
        P_p = (k + 1) * dP_per_round
        g0 = 2.0 * kappa * np.sqrt(P_p) * L
        half_g0 = 0.5 * g0
        neg_half_g0_gamma = -0.5 * g0 * gamma

        for i in range(n):
            acc = 0.0
            start = J_indptr[i]
            end = J_indptr[i + 1]
            for jj in range(start, end):
                acc += J_data[jj] * c[J_indices[jj]]
            Jc[i] = acc

        for i in range(n):
            coupled_in_i = sqrt_eta * c[i] + Jc[i]
            I_in_i = coupled_in_i * coupled_in_i
            half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
            sqrt_G_I_i = np.exp(half_g_i)
            noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
            c[i] = sqrt_G_I_i * coupled_in_i + noise_i

        cut = 0.0
        for e in range(num_edges):
            if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                cut += edge_w[e]

        if cut > best_cut:
            best_cut = cut
            for i in range(n):
                best_signs[i] = c[i] > 0.0

        if sample_idx < num_samples and sample_rounds[sample_idx] == k:
            for i in range(n):
                c_history[sample_idx, i] = c[i]
            cut_history[sample_idx] = cut
            sample_idx += 1

    return c_history, cut_history, best_cut, best_signs


def simulate_cim_with_trajectory(
    n: int,
    J: csr_matrix,
    edges: list[tuple[int, int]],
    num_rounds: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seed: int,
    num_samples: int = 200,
    weights: list[float] | None = None,
):
    """単一 seed で CIM を走らせて振幅軌跡を返す公開 API。

    sample_rounds は 0 から num_rounds-1 まで等間隔に num_samples 点(重複除去)。
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))

    num_samples = min(int(num_samples), int(num_rounds))
    sample_rounds = np.linspace(0, num_rounds - 1, num_samples, dtype=np.int64)
    sample_rounds = np.unique(sample_rounds)

    c_hist, cut_hist, best_cut, best_signs = _simulate_cim_trajectory(
        n,
        int(num_rounds),
        J.data,
        J.indices,
        J.indptr,
        edge_a,
        edge_b,
        edge_w,
        float(kappa),
        float(L),
        float(gamma),
        float(eta),
        float(bandwidth),
        float(photon_energy),
        float(dP_per_round),
        np.int64(seed),
        sample_rounds,
    )
    return c_hist, cut_hist, sample_rounds, float(best_cut), best_signs


def main():
    """100 seed 並列ベンチマーク (G22)。

    目標: polish 後の 100 seed 平均 cut ≥ 13340。
    """
    import time

    config = {
        "kappa": 130.0,
        "L": 0.05,
        "gamma": 42.09,
        "loss_dB": 11.0,
        "bandwidth": 1.0e9,
        "photon_energy_J": 1.28e-19,
        "dP_per_round": 0.05e-3,
        "coupling": -0.03,
        "num_rounds": 1500,
        "num_trials": 100,
        "seed_base": 0,
    }

    eta = 10.0 ** (-config["loss_dB"] / 10.0)

    filepath = "input/G22.txt"
    n, k_edges, adj, edges = load_graph(filepath)
    print(f"N={n}, K={k_edges}, eta={eta:.4f}")
    print(f"num_trials={config['num_trials']}, num_rounds={config['num_rounds']}")

    J = build_coupling_matrix(n, edges, config["coupling"])
    seeds = np.arange(config["seed_base"], config["seed_base"] + config["num_trials"], dtype=np.int64)

    # ==== JIT ウォームアップ (1 trial で雛形をコンパイル) ====
    print("\n[warmup] JIT compile...")
    _ = simulate_cim_batch_polished(
        n=n, J=J, edges=edges, num_rounds=10, num_trials=1,
        kappa=config["kappa"], L=config["L"], gamma=config["gamma"], eta=eta,
        bandwidth=config["bandwidth"], photon_energy=config["photon_energy_J"],
        dP_per_round=config["dP_per_round"],
        seeds=np.array([0], dtype=np.int64),
    )

    # ==== baseline (polish 無し) ====
    print("\n[baseline] CIM only (no polish) ...")
    t0 = time.perf_counter()
    base_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges,
        num_rounds=config["num_rounds"], num_trials=config["num_trials"],
        kappa=config["kappa"], L=config["L"], gamma=config["gamma"], eta=eta,
        bandwidth=config["bandwidth"], photon_energy=config["photon_energy_J"],
        dP_per_round=config["dP_per_round"], seeds=seeds,
    )
    t_base = time.perf_counter() - t0

    # ==== polished (1-flip greedy 局所最適化付き) ====
    print("[polished] CIM + 1-flip greedy ...")
    t0 = time.perf_counter()
    pol_cuts, pol_signs = simulate_cim_batch_polished(
        n=n, J=J, edges=edges,
        num_rounds=config["num_rounds"], num_trials=config["num_trials"],
        kappa=config["kappa"], L=config["L"], gamma=config["gamma"], eta=eta,
        bandwidth=config["bandwidth"], photon_energy=config["photon_energy_J"],
        dP_per_round=config["dP_per_round"], seeds=seeds,
    )
    t_pol = time.perf_counter() - t0

    # ==== 統計 ====
    def stats(cuts):
        return float(cuts.mean()), float(cuts.max()), float(cuts.min()), float(cuts.std(ddof=1))

    bm, bM, bn_, bs = stats(base_cuts)
    pm, pM, pn_, ps = stats(pol_cuts)
    bks = 13359
    bks_hits = int((pol_cuts >= bks).sum())

    print("\n" + "=" * 60)
    print(f"{'':12}{'mean':>10}{'best':>10}{'worst':>10}{'std':>10}{'wall':>10}")
    print("-" * 60)
    print(f"{'baseline':12}{bm:10.1f}{bM:10.0f}{bn_:10.0f}{bs:10.2f}{t_base:10.2f}")
    print(f"{'polished':12}{pm:10.1f}{pM:10.0f}{pn_:10.0f}{ps:10.2f}{t_pol:10.2f}")
    print("-" * 60)
    print(f"d_mean = {pm - bm:+.1f}  d_best = {pM - bM:+.0f}  BKS hits (13359): {bks_hits}/{config['num_trials']}")

    target = 13340.0
    status = "PASS" if pm >= target else "MISS"
    print(f"\n[Goal] mean >= {target}: polished mean = {pm:.1f}  --> {status}")

    # ==== 検算 (最良 trial を独立に再カウント) ====
    best_trial = int(np.argmax(pol_cuts))
    best_x = pol_signs[best_trial].astype(np.int64).tolist()
    reported = int(pol_cuts[best_trial])
    print()
    run_all_checks(best_x, n, k_edges, adj, edges, reported)


if __name__ == "__main__":
    main()
