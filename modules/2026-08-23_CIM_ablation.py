"""2026-08-23_CIM_ablation.py — CIM の「多様性が出ない理由」を切り分けるアブレーション版。

`modules/CIM.py` の進行波モデル(Inoue & Yoshida 2022)をベースに、
仮説検証用のノブを 4 つ追加しただけのもの。**既定値では baseline と
ビット一致する**(乱数の消費順序も変えていない)ので、差分がそのまま
ノブの効果になる。

追加ノブ(対応する仮説):
  (1) init_scale  … 初期振幅を一様乱数 U(-init_scale, +init_scale) にする。
                     baseline は c(0) = 0(全試行で同一の初期状態)。
                     → 「初期条件の自由度が無いこと」が多様性欠如の原因か?
  (2) dP_per_round… ポンプ ramp の速さ。既存引数だが、掃引対象として明示。
                     → 「しきい値近傍の線形増幅段が長いこと」が原因か?
  (3) noise_mult  … 真空雑音 n_0 の振幅を定数倍する。
                     → 「飽和後にノイズが効かないこと」が原因か?
  (4) async_frac / seq_update … 全頂点同時更新をやめる。
                     async_frac: 各 round で更新する頂点の割合(Bernoulli)。
                     seq_update: 選ばれた頂点をランダム順に逐次更新し、
                                 J c を差分更新する(Gauss-Seidel 型の真の非同期)。
                     → 「更新順序という対称性の破れ源が無いこと」が原因か?

  (5) pump_offset … ポンプを round 0 ではなく round `pump_offset` の水準から始める
                     (第1段階=しきい値下のパタパタ期間を丸ごと飛ばす)。
                     → 「第1段階は解の形成に寄与しているのか?」

baseline 相当: init_scale=0.0, noise_mult=1.0, async_frac=1.0, seq_update=False,
pump_offset=0
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange
from scipy.sparse import csr_matrix


@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_ablation_batch(
    n: int,
    num_rounds: int,
    num_trials: int,
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
    seeds: np.ndarray,
    init_scale: float,
    noise_mult: float,
    async_frac: float,
    seq_update: bool,
    pump_offset: int,
):
    """CIM アブレーション版のコアループ(trial 単位で prange 並列)。

    既定ノブ (init_scale=0, noise_mult=1, async_frac=1, seq_update=False) では
    modules/CIM.py の _simulate_cim_batch と完全に同一の計算・同一の乱数消費。
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy) * noise_mult
    num_edges = edge_a.shape[0]
    do_async = async_frac < 1.0

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18

        # ---- (1) 初期振幅。init_scale=0 なら乱数を一切引かない(baseline 互換) ----
        if init_scale > 0.0:
            for i in range(n):
                c[i] = (np.random.random() - 0.5) * 2.0 * init_scale

        # 非同期更新で使う作業配列(同期時は未使用)
        sel = np.empty(n, dtype=np.int64)

        for k in range(num_rounds):
            # Step 1: ポンプパワー → 非飽和利得係数 g_0
            P_p = (k + 1 + pump_offset) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            # Step 2: Jc = J @ c(round 開始時点の c から)
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            if not do_async:
                # ---- 同期更新(baseline) ----
                for i in range(n):
                    coupled_in_i = sqrt_eta * c[i] + Jc[i]
                    I_in_i = coupled_in_i * coupled_in_i
                    half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                    sqrt_G_I_i = np.exp(half_g_i)
                    noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                    c[i] = sqrt_G_I_i * coupled_in_i + noise_i
            else:
                # ---- (4) 非同期更新: 割合 async_frac の頂点だけ進める ----
                m = 0
                for i in range(n):
                    if np.random.random() < async_frac:
                        sel[m] = i
                        m += 1

                if seq_update and m > 1:
                    # Fisher-Yates で更新順序をランダム化
                    for a in range(m - 1, 0, -1):
                        b = np.int64(np.random.random() * (a + 1))
                        if b > a:
                            b = a
                        tmp = sel[a]
                        sel[a] = sel[b]
                        sel[b] = tmp

                for t in range(m):
                    i = sel[t]
                    coupled_in_i = sqrt_eta * c[i] + Jc[i]
                    I_in_i = coupled_in_i * coupled_in_i
                    half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                    sqrt_G_I_i = np.exp(half_g_i)
                    noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                    c_new = sqrt_G_I_i * coupled_in_i + noise_i
                    if seq_update:
                        # 真の非同期: 更新を即座に隣接頂点の局所場へ反映(差分更新)
                        delta = c_new - c[i]
                        start = J_indptr[i]
                        end = J_indptr[i + 1]
                        for jj in range(start, end):
                            Jc[J_indices[jj]] += J_data[jj] * delta
                    c[i] = c_new

            # Step 6: 重み付き cut(baseline と同一)
            cut = 0.0
            for e in range(num_edges):
                if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                    cut += edge_w[e]

            if cut > best_cut:
                best_cut = cut
                for i in range(n):
                    best_signs[i] = c[i] > 0.0

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


def simulate_cim_ablation_batch(
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
    *,
    init_scale: float = 0.0,
    noise_mult: float = 1.0,
    async_frac: float = 1.0,
    seq_update: bool = False,
    pump_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """公開ラッパー。既定引数で modules.CIM.simulate_cim_batch と一致する。

    Returns:
        best_cuts:  (num_trials,)
        best_signs: (num_trials, n)  bool
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    return _simulate_cim_ablation_batch(
        n,
        int(num_rounds),
        int(num_trials),
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
        float(init_scale),
        float(noise_mult),
        float(async_frac),
        bool(seq_update),
        int(pump_offset),
    )


def threshold_round(kappa, L, eta, dP_per_round) -> float:
    """発振しきい値に達する round 数の解析推定。

    1 周あたりの正味利得が 1 を超える条件 sqrt(eta) * exp(g0/2) = 1 より
        g0 = 2 * kappa * sqrt(P_p) * L = -ln(eta),  P_p = k * dP_per_round
    を k について解く。線形増幅段(= 固有モード競合が起きる区間)の長さの目安。
    """
    g0_th = -np.log(eta)
    p_th = (g0_th / (2.0 * kappa * L)) ** 2
    return p_th / dP_per_round


# ============================================================
#  診断: 初期振幅の記憶がどこまで残るか
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _probe_cim_amplitude(
    n: int,
    num_rounds: int,
    num_trials: int,
    J_data: np.ndarray,
    J_indices: np.ndarray,
    J_indptr: np.ndarray,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    dP_per_round: float,
    seeds: np.ndarray,
    init_scale: float,
    stride: int,
):
    """同期更新の CIM を回しつつ、stride round ごとに 2 つの量を記録する。

      norms[t, m]    … その時点の振幅ノルム ||c||_2
      overlaps[t, m] … 初期符号 sign(c(0)) との重なり |1 - 2*hamming/n|(反転対称を畳む)

    init_scale=0 のときは c(0)=0 で符号が定義できないため、overlaps は
    「round 1 直後の符号」を基準にする(= ノイズが作った最初の符号パターン)。
    """
    K = num_rounds // stride
    norms = np.zeros((num_trials, K), dtype=np.float64)
    overlaps = np.zeros((num_trials, K), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])
        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        ref = np.zeros(n, dtype=np.bool_)

        if init_scale > 0.0:
            for i in range(n):
                c[i] = (np.random.random() - 0.5) * 2.0 * init_scale
            for i in range(n):
                ref[i] = c[i] > 0.0

        m = 0
        for k in range(num_rounds):
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            for i in range(n):
                acc = 0.0
                for jj in range(J_indptr[i], J_indptr[i + 1]):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            for i in range(n):
                coupled_in_i = sqrt_eta * c[i] + Jc[i]
                I_in_i = coupled_in_i * coupled_in_i
                sqrt_G_I_i = np.exp(half_g0 + neg_half_g0_gamma * I_in_i)
                c[i] = (sqrt_G_I_i * coupled_in_i
                        + np.random.standard_normal() * (noise_const * sqrt_G_I_i))

            if k == 0 and init_scale <= 0.0:
                for i in range(n):
                    ref[i] = c[i] > 0.0

            if (k + 1) % stride == 0 and m < K:
                nrm = 0.0
                diff = 0
                for i in range(n):
                    nrm += c[i] * c[i]
                    if (c[i] > 0.0) != ref[i]:
                        diff += 1
                norms[trial_idx, m] = np.sqrt(nrm)
                overlaps[trial_idx, m] = abs(1.0 - 2.0 * diff / n)
                m += 1

    return norms, overlaps


def probe_cim_amplitude(
    n, J, num_rounds, num_trials, kappa, L, gamma, eta, bandwidth,
    photon_energy, dP_per_round, seeds, *, init_scale=0.0, stride=20,
):
    """_probe_cim_amplitude の公開ラッパー。(norms, overlaps, rounds) を返す。"""
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    norms, overlaps = _probe_cim_amplitude(
        int(n), int(num_rounds), int(num_trials),
        J.data, J.indices, J.indptr,
        float(kappa), float(L), float(gamma), float(eta), float(bandwidth),
        float(photon_energy), float(dP_per_round), seeds_arr,
        float(init_scale), int(stride),
    )
    rounds = np.arange(1, norms.shape[1] + 1) * stride
    return norms, overlaps, rounds


# ============================================================
#  診断: 3 段階(パタパタ期 / 形成期 / 飽和期)の境界を測る
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _probe_cim_trajectory(
    n: int,
    num_rounds: int,
    num_trials: int,
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
    seeds: np.ndarray,
    stride: int,
):
    """同期更新の CIM を回し、stride round ごとに 4 つの量を記録する。

      norms[t, m]     … 振幅ノルム ||c||_2
      cuts[t, m]      … その時点の符号が与えるカット
      best_cuts[t, m] … round 1..k の累積最良カット(= k round で打ち切ったときの出力)
      flips[t, m]     … 直前の記録時点から符号が変わった頂点の割合(パタパタ度)

    best_cuts が平坦になった round 以降は、走らせても出力が変わらない。
    """
    K = num_rounds // stride
    norms = np.zeros((num_trials, K), dtype=np.float64)
    cuts = np.zeros((num_trials, K), dtype=np.float64)
    best_cuts = np.zeros((num_trials, K), dtype=np.float64)
    flips = np.zeros((num_trials, K), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])
        c = np.zeros(n, dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        prev = np.zeros(n, dtype=np.bool_)
        best = -1.0e18
        m = 0

        for k in range(num_rounds):
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            for i in range(n):
                acc = 0.0
                for jj in range(J_indptr[i], J_indptr[i + 1]):
                    acc += J_data[jj] * c[J_indices[jj]]
                Jc[i] = acc

            for i in range(n):
                coupled_in_i = sqrt_eta * c[i] + Jc[i]
                I_in_i = coupled_in_i * coupled_in_i
                sqrt_G_I_i = np.exp(half_g0 + neg_half_g0_gamma * I_in_i)
                c[i] = (sqrt_G_I_i * coupled_in_i
                        + np.random.standard_normal() * (noise_const * sqrt_G_I_i))

            cut = 0.0
            for e in range(num_edges):
                if (c[edge_a[e]] > 0.0) != (c[edge_b[e]] > 0.0):
                    cut += edge_w[e]
            if cut > best:
                best = cut

            if (k + 1) % stride == 0 and m < K:
                nrm = 0.0
                nflip = 0
                for i in range(n):
                    nrm += c[i] * c[i]
                    sgn = c[i] > 0.0
                    if k + 1 > stride and sgn != prev[i]:
                        nflip += 1
                    prev[i] = sgn
                norms[trial_idx, m] = np.sqrt(nrm)
                cuts[trial_idx, m] = cut
                best_cuts[trial_idx, m] = best
                flips[trial_idx, m] = nflip / n
                m += 1

    return norms, cuts, best_cuts, flips


def probe_cim_trajectory(
    n, J, edges, num_rounds, num_trials, kappa, L, gamma, eta, bandwidth,
    photon_energy, dP_per_round, seeds, weights=None, *, stride=10,
):
    """_probe_cim_trajectory の公開ラッパー。(norms, cuts, best_cuts, flips, rounds)。"""
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    edge_w = (np.ones(edges_np.shape[0]) if weights is None
              else np.ascontiguousarray(np.asarray(weights, dtype=np.float64)))
    out = _probe_cim_trajectory(
        int(n), int(num_rounds), int(num_trials), J.data, J.indices, J.indptr,
        edge_a, edge_b, edge_w, float(kappa), float(L), float(gamma), float(eta),
        float(bandwidth), float(photon_energy), float(dP_per_round),
        np.ascontiguousarray(np.asarray(seeds, dtype=np.int64)), int(stride))
    rounds = np.arange(1, out[0].shape[1] + 1) * stride
    return (*out, rounds)
