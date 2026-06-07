"""
CIM + PT v3 swap ablation : swap 有/無を同一構成で切り分ける対照実験 (2026-06-08)

【この版で追加した点】
  従来 (2026-06-06_CIM_PT_v3.py) の比較は
      ランプCIM(1レプリカ) / v2(swap有) / v3(swap有)
  だったため、「v3 がランプCIMに勝った」効果に
      (a) 1→3 レプリカ化(異なるランプ速度の多スタート集団)の寄与
      (b) その3本の間で swap した寄与(= PT の本体)
  が混在し、swap 自体の純粋な効果を分離できていなかった。
  本版では 3本ランプ構成のまま **swap 有/無だけを切り替えた対照**
  (v3 swap有 vs v3 swap無)を正式な比較対象として報告・作図する。
  swap無の best_cuts は元々 β較正用に計算済みだったものを採用する
  (pump_mults・ランプ・seed・trial数すべて swap有と同一)。

----------------------------------------------------------------------
CIM + PT v3 : 各レプリカにも時間発展(ポンプ・ランプ)を持たせた版 (2026-06-06)

----------------------------------------------------------------------
背景 (docs/CIM_PT_why_failed.md の続き):
  v1/v2 は各レプリカのポンプを **固定** していた。v2 で β ラダーの向きを正しても
  ランプ CIM に届かなかった原因は、固定ポンプでは「探索する場所(低ポンプ)」と
  「凍結する場所(高ポンプ)」が分離して両立せず、焼きなまし(連続的な凍結)を
  再現できないことにあった。

v3 の変更点:
  各レプリカに **ポンプ・ランプ(時間発展)** を持たせる。各レプリカのポンプを
      P_r(k) = mult_r * P_ramp(k),   P_ramp(k) = (k+1) * dP_per_round
  とする(P_ramp は通常ランプ CIM と同一の掃引)。
    - replica1 (mult=1.0) は通常ランプ CIM そのもの。
    - replica0 (mult<1) はゆっくり昇って探索を長く保ち、
      replica2 (mult>1) は速く昇って早く凍結する(= 焼きなまし速度の異なる集団)。
  これで全レプリカが「凍結」でき、PT スワップで良い配置を集団間で共有する
  (population-annealing 的な構成)。
  β は swap 無効ランの定常カットから較正し、v2 の教訓どおり
  **最良カットのレプリカを cold(β 最大)** に割り当てる。

比較 (等計算量):
  ランプ CIM (baseline) / CIM+PT v2(固定ポンプ反転) / CIM+PT v3(ランプ各レプリカ)。
  物理パラメータ・pump_mults・swap 間隔・seed は揃える。
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from numba import njit, prange

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

NR = 3


# ============================================================
#  純粋ヘルパー (v1 と同一; importlib の numba キャッシュ衝突を避け自己完結化)
# ============================================================
def compute_threshold_pump(kappa: float, L: float, eta: float) -> float:
    """CIM 発振しきい値ポンプ P_th = (ln(1/η) / (2κL))^2 [W]。"""
    return (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2


def calibrate_betas(cut_tail, kappa_target: float = 1.0, eps: float = 1.0):
    """swap 無効ランの定常カットから β ラダー(昇順, 低温ほど大)を構成する。"""
    betas = np.zeros(NR, dtype=np.float64)
    for r in range(NR - 1):
        gap = abs(float(cut_tail[r + 1]) - float(cut_tail[r]))
        betas[r + 1] = betas[r] + kappa_target / max(gap, eps)
    return betas


# ============================================================
#  固定ポンプ 3 レプリカ PT カーネル (v1 と同一; v2 比較用に自己完結で保持)
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_pt_fixed_batch(
    n, num_rounds, num_trials,
    J_data, J_indices, J_indptr, edge_a, edge_b, edge_w,
    kappa, L, gamma, eta, bandwidth, photon_energy,
    pump_levels, betas, swap_interval, do_swap, sample_interval, num_samples, seeds,
):
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)
    traj_best = np.zeros((num_trials, num_samples), dtype=np.float64)
    traj_amp = np.zeros((num_trials, num_samples, NR), dtype=np.float64)
    traj_cut = np.zeros((num_trials, num_samples, NR), dtype=np.float64)
    swap_acc_out = np.zeros((num_trials, NR - 1), dtype=np.float64)
    swap_att_out = np.zeros((num_trials, NR - 1), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    half_g0 = np.empty(NR, dtype=np.float64)
    neg_half_g0_gamma = np.empty(NR, dtype=np.float64)
    for r in range(NR):
        g0_r = 2.0 * kappa * np.sqrt(pump_levels[r]) * L
        half_g0[r] = 0.5 * g0_r
        neg_half_g0_gamma[r] = -0.5 * g0_r * gamma

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])
        c = np.zeros((NR, n), dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        cut_r = np.zeros(NR, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18
        swap_acc = np.zeros(NR - 1, dtype=np.float64)
        swap_att = np.zeros(NR - 1, dtype=np.float64)

        for k in range(num_rounds):
            for r in range(NR):
                for i in range(n):
                    acc = 0.0
                    start = J_indptr[i]
                    end = J_indptr[i + 1]
                    for jj in range(start, end):
                        acc += J_data[jj] * c[r, J_indices[jj]]
                    Jc[i] = acc
                hg0 = half_g0[r]
                nhg = neg_half_g0_gamma[r]
                for i in range(n):
                    coupled_in_i = sqrt_eta * c[r, i] + Jc[i]
                    I_in_i = coupled_in_i * coupled_in_i
                    half_g_i = hg0 + nhg * I_in_i
                    sqrt_G_I_i = np.exp(half_g_i)
                    noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                    c[r, i] = sqrt_G_I_i * coupled_in_i + noise_i

            for r in range(NR):
                cr = 0.0
                for e in range(num_edges):
                    if (c[r, edge_a[e]] > 0.0) != (c[r, edge_b[e]] > 0.0):
                        cr += edge_w[e]
                cut_r[r] = cr
                if cr > best_cut:
                    best_cut = cr
                    for i in range(n):
                        best_signs[i] = c[r, i] > 0.0

            if do_swap == 1 and (k + 1) % swap_interval == 0:
                for r in range(NR - 1):
                    swap_att[r] += 1.0
                    dbeta = betas[r] - betas[r + 1]
                    dE = cut_r[r + 1] - cut_r[r]
                    arg = dbeta * dE
                    accept = False
                    if arg >= 0.0:
                        accept = True
                    elif np.random.random() < np.exp(arg):
                        accept = True
                    if accept:
                        swap_acc[r] += 1.0
                        for i in range(n):
                            tmp = c[r, i]
                            c[r, i] = c[r + 1, i]
                            c[r + 1, i] = tmp
                        tc = cut_r[r]
                        cut_r[r] = cut_r[r + 1]
                        cut_r[r + 1] = tc

            if (k + 1) % sample_interval == 0:
                s_idx = (k + 1) // sample_interval - 1
                if 0 <= s_idx < num_samples:
                    traj_best[trial_idx, s_idx] = best_cut
                    for r in range(NR):
                        acc = 0.0
                        for i in range(n):
                            acc += abs(c[r, i])
                        traj_amp[trial_idx, s_idx, r] = acc / n
                        traj_cut[trial_idx, s_idx, r] = cut_r[r]

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]
        for r in range(NR - 1):
            swap_acc_out[trial_idx, r] = swap_acc[r]
            swap_att_out[trial_idx, r] = swap_att[r]

    return (best_cuts_out, best_signs_out, traj_best, traj_amp, traj_cut,
            swap_acc_out, swap_att_out)


def simulate_cim_pt_fixed(
    n, J, edges, num_rounds, num_trials,
    kappa, L, gamma, eta, bandwidth, photon_energy, seeds,
    *, pump_levels, betas=None, swap_interval=10, do_swap=True,
    sample_interval=None, weights=None,
):
    """固定ポンプ 3 レプリカ + Metropolis PT (v1 相当) の公開 API。"""
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    num_edges = edges_np.shape[0]
    edge_w = (np.ones(num_edges, dtype=np.float64) if weights is None
              else np.ascontiguousarray(np.asarray(weights, dtype=np.float64)))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    pump_levels = np.ascontiguousarray(np.sort(np.asarray(pump_levels, dtype=np.float64)))
    if betas is None:
        betas = np.zeros(NR, dtype=np.float64)
    betas = np.ascontiguousarray(np.asarray(betas, dtype=np.float64))

    if sample_interval is None or sample_interval <= 0:
        sample_interval = int(num_rounds)
    sample_interval = int(sample_interval)
    num_samples = max(1, int(num_rounds) // sample_interval)

    (best_cuts, best_signs, traj_best, traj_amp, traj_cut,
     swap_acc, swap_att) = _simulate_cim_pt_fixed_batch(
        n, int(num_rounds), int(num_trials),
        J.data, J.indices, J.indptr, edge_a, edge_b, edge_w,
        float(kappa), float(L), float(gamma), float(eta),
        float(bandwidth), float(photon_energy),
        pump_levels, betas, int(swap_interval), int(1 if do_swap else 0),
        sample_interval, num_samples, seeds_arr)
    att_tot = swap_att.sum(axis=0)
    acc_tot = swap_acc.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        swap_rate = np.where(att_tot > 0, acc_tot / att_tot, 0.0)
    return {
        "best_cuts": best_cuts, "best_signs": best_signs,
        "traj_best": traj_best, "traj_amp": traj_amp, "traj_cut": traj_cut,
        "sample_rounds": np.arange(1, num_samples + 1) * sample_interval,
        "swap_rate": swap_rate, "swap_accepts": acc_tot, "swap_attempts": att_tot,
    }


# ============================================================
#  ランプ付き 3 レプリカ CIM + Metropolis 受理 PT カーネル
#  (上の固定ポンプ版を「ポンプを毎ラウンド更新」に拡張)
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_pt_ramp_batch(
    n, num_rounds, num_trials,
    J_data, J_indices, J_indptr,
    edge_a, edge_b, edge_w,
    kappa, L, gamma, eta, bandwidth, photon_energy,
    pump_mults,        # (NR,) 各レプリカのランプ倍率
    dP_per_round,      # ランプ 1 ラウンドあたりの増分 [W]
    betas,             # (NR,) スワップ受理用 実効逆温度
    swap_interval, do_swap, sample_interval, num_samples,
    seeds,
):
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)
    traj_best = np.zeros((num_trials, num_samples), dtype=np.float64)
    traj_amp = np.zeros((num_trials, num_samples, NR), dtype=np.float64)
    traj_cut = np.zeros((num_trials, num_samples, NR), dtype=np.float64)
    swap_acc_out = np.zeros((num_trials, NR - 1), dtype=np.float64)
    swap_att_out = np.zeros((num_trials, NR - 1), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]
    two_kappa_L = 2.0 * kappa * L

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        c = np.zeros((NR, n), dtype=np.float64)
        Jc = np.zeros(n, dtype=np.float64)
        cut_r = np.zeros(NR, dtype=np.float64)
        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18
        swap_acc = np.zeros(NR - 1, dtype=np.float64)
        swap_att = np.zeros(NR - 1, dtype=np.float64)

        for k in range(num_rounds):
            P_ramp = (k + 1) * dP_per_round
            # ---- 各レプリカを 1 ラウンド発展(ポンプは毎ラウンド更新) ----
            for r in range(NR):
                P_r = pump_mults[r] * P_ramp
                g0_r = two_kappa_L * np.sqrt(P_r)
                hg0 = 0.5 * g0_r
                nhg = -0.5 * g0_r * gamma
                for i in range(n):
                    acc = 0.0
                    start = J_indptr[i]
                    end = J_indptr[i + 1]
                    for jj in range(start, end):
                        acc += J_data[jj] * c[r, J_indices[jj]]
                    Jc[i] = acc
                for i in range(n):
                    coupled_in_i = sqrt_eta * c[r, i] + Jc[i]
                    I_in_i = coupled_in_i * coupled_in_i
                    half_g_i = hg0 + nhg * I_in_i
                    sqrt_G_I_i = np.exp(half_g_i)
                    noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                    c[r, i] = sqrt_G_I_i * coupled_in_i + noise_i

            # ---- 各レプリカ cut 評価 + best 更新 ----
            for r in range(NR):
                cr = 0.0
                for e in range(num_edges):
                    if (c[r, edge_a[e]] > 0.0) != (c[r, edge_b[e]] > 0.0):
                        cr += edge_w[e]
                cut_r[r] = cr
                if cr > best_cut:
                    best_cut = cr
                    for i in range(n):
                        best_signs[i] = c[r, i] > 0.0

            # ---- PT スワップ(隣接ペア Metropolis 受理) ----
            if do_swap == 1 and (k + 1) % swap_interval == 0:
                for r in range(NR - 1):
                    swap_att[r] += 1.0
                    dbeta = betas[r] - betas[r + 1]
                    dE = cut_r[r + 1] - cut_r[r]
                    arg = dbeta * dE
                    accept = False
                    if arg >= 0.0:
                        accept = True
                    elif np.random.random() < np.exp(arg):
                        accept = True
                    if accept:
                        swap_acc[r] += 1.0
                        for i in range(n):
                            tmp = c[r, i]
                            c[r, i] = c[r + 1, i]
                            c[r + 1, i] = tmp
                        tc = cut_r[r]
                        cut_r[r] = cut_r[r + 1]
                        cut_r[r + 1] = tc

            # ---- サンプリング ----
            if (k + 1) % sample_interval == 0:
                s_idx = (k + 1) // sample_interval - 1
                if 0 <= s_idx < num_samples:
                    traj_best[trial_idx, s_idx] = best_cut
                    for r in range(NR):
                        acc = 0.0
                        for i in range(n):
                            acc += abs(c[r, i])
                        traj_amp[trial_idx, s_idx, r] = acc / n
                        traj_cut[trial_idx, s_idx, r] = cut_r[r]

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]
        for r in range(NR - 1):
            swap_acc_out[trial_idx, r] = swap_acc[r]
            swap_att_out[trial_idx, r] = swap_att[r]

    return (best_cuts_out, best_signs_out, traj_best, traj_amp, traj_cut,
            swap_acc_out, swap_att_out)


def simulate_cim_pt_ramp_batch(
    n, J, edges, num_rounds, num_trials,
    kappa, L, gamma, eta, bandwidth, photon_energy, dP_per_round,
    seeds, *, pump_mults, betas, swap_interval=10, do_swap=True,
    sample_interval=None, weights=None,
):
    """ランプ付き 3 レプリカ CIM + Metropolis PT の公開 API。"""
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    num_edges = edges_np.shape[0]
    edge_w = (np.ones(num_edges, dtype=np.float64) if weights is None
              else np.ascontiguousarray(np.asarray(weights, dtype=np.float64)))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    pump_mults = np.ascontiguousarray(np.asarray(pump_mults, dtype=np.float64))
    betas = np.ascontiguousarray(np.asarray(betas, dtype=np.float64))

    if sample_interval is None or sample_interval <= 0:
        sample_interval = int(num_rounds)
    sample_interval = int(sample_interval)
    num_samples = max(1, int(num_rounds) // sample_interval)

    (best_cuts, best_signs, traj_best, traj_amp, traj_cut,
     swap_acc, swap_att) = _simulate_cim_pt_ramp_batch(
        n, int(num_rounds), int(num_trials),
        J.data, J.indices, J.indptr, edge_a, edge_b, edge_w,
        float(kappa), float(L), float(gamma), float(eta),
        float(bandwidth), float(photon_energy),
        pump_mults, float(dP_per_round), betas,
        int(swap_interval), int(1 if do_swap else 0),
        sample_interval, num_samples, seeds_arr)
    sample_rounds = np.arange(1, num_samples + 1) * sample_interval
    att_tot = swap_att.sum(axis=0)
    acc_tot = swap_acc.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        swap_rate = np.where(att_tot > 0, acc_tot / att_tot, 0.0)
    return {
        "best_cuts": best_cuts, "best_signs": best_signs,
        "traj_best": traj_best, "traj_amp": traj_amp, "traj_cut": traj_cut,
        "sample_rounds": sample_rounds, "swap_rate": swap_rate,
        "swap_accepts": acc_tot, "swap_attempts": att_tot,
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
    from modules.verify import compute_cut_from_edges

    EXPERIMENT_KIND = "cim_pt_v3_swap_ablation"
    KNOWN_BEST = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}

    parser = argparse.ArgumentParser(
        description="CIM+PT v3(各レプリカにランプ) を v2/ランプCIM と等計算量比較")
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--cim-rounds", type=int, default=1500)
    parser.add_argument("--cim-coupling", type=float, default=-0.03)
    parser.add_argument("--pump-mults", type=float, nargs=3, default=[0.8, 1.0, 1.3])
    parser.add_argument("--swap-interval", type=int, default=10)
    parser.add_argument("--kappa-target", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--known-best", type=int, default=None)
    parser.add_argument("--tag", type=str, default="perreplica_ramp")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 2 * plt.rcParams["font.size"]

    def ticks_in(ax):
        ax.tick_params(direction="in", which="both", top=True, right=True)

    graph_path = Path(args.graph)
    graph_name = graph_path.stem
    n, k_edges, _adj, edges, weights = load_graph(str(graph_path), return_weights=True)
    use_weights = any(w != 1.0 for w in weights)
    w_arg = weights if use_weights else None
    print(f"Graph: {graph_path} N={n} K={k_edges} weighted={use_weights}")

    known_best = args.known_best if args.known_best is not None else KNOWN_BEST.get(graph_name)
    if known_best is not None:
        print(f"Known best: {known_best}")

    NT = args.num_trials
    pt_seeds = np.arange(args.seed_base, args.seed_base + NT, dtype=np.int64)
    cim_seeds = np.arange(args.seed_base, args.seed_base + NR * NT, dtype=np.int64)
    pump_mults = np.sort(np.asarray(args.pump_mults, dtype=np.float64))

    J = build_coupling_matrix(n, edges, args.cim_coupling, weights=w_arg)
    cim_params = dict(
        kappa=130.0, L=0.05, gamma=42.09, eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19, dP_per_round=0.05e-3)
    pt_phys = {k: v for k, v in cim_params.items() if k != "dP_per_round"}
    dP = cim_params["dP_per_round"]
    p_th = compute_threshold_pump(cim_params["kappa"], cim_params["L"], cim_params["eta"])

    # 各レプリカが P_th を横切るラウンド(参考表示)
    cross = [int(np.ceil(p_th / (m * dP))) for m in pump_mults]
    print(f"P_th={p_th*1e3:.2f}mW  pump_mults={pump_mults.tolist()}  "
          f"P_th 到達round(低→高mult)={cross}")

    # ==== baseline ランプ CIM (NR*NT trial) ====
    print(f"\n[ランプCIM] {NR*NT} trials  rounds={args.cim_rounds}")
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=NR * NT, seeds=cim_seeds, weights=w_arg, **cim_params)
    cim_time = time.time() - t0
    print(f"  time={cim_time:.2f}s  mean={cim_cuts.mean():.1f}  best={cim_cuts.max():.0f}")

    # ==== v2 参考: 固定ポンプ反転 PT (同一 seed) ====
    pump_levels_fixed = np.sort(np.array([m * p_th for m in pump_mults]))
    res_v2_noswap = simulate_cim_pt_fixed(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds, num_trials=NT,
        seeds=pt_seeds, weights=w_arg, pump_levels=pump_levels_fixed, do_swap=False,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    cut_tail_fixed = res_v2_noswap["traj_cut"][:, -max(1, res_v2_noswap["sample_rounds"].size // 3):, :].mean(axis=(0, 1))
    betas_fixed_norm = calibrate_betas(cut_tail_fixed, kappa_target=args.kappa_target)
    betas_fixed_rev = betas_fixed_norm.max() - betas_fixed_norm   # v2: 低ポンプ=cold
    print(f"\n[CIM+PT v2(固定ポンプ反転)] {NT} trials")
    t0 = time.time()
    res_v2 = simulate_cim_pt_fixed(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds, num_trials=NT,
        seeds=pt_seeds, weights=w_arg, pump_levels=pump_levels_fixed,
        betas=betas_fixed_rev, do_swap=True,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    v2_time = time.time() - t0
    v2_cuts = res_v2["best_cuts"]
    print(f"  time={v2_time:.2f}s  mean={v2_cuts.mean():.1f}  best={v2_cuts.max():.0f}  "
          f"受理率={[round(x,3) for x in res_v2['swap_rate'].tolist()]}")

    # ==== v3 swap 無効ランプ参照 (β較正 & 向き判定) ====
    print(f"\n[v3 ランプ各レプリカ/swap無] {NT} trials  (β較正)")
    t0 = time.time()
    res_noswap = simulate_cim_pt_ramp_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds, num_trials=NT,
        seeds=pt_seeds, weights=w_arg, pump_mults=pump_mults,
        betas=np.zeros(NR), do_swap=False, dP_per_round=dP,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    noswap_time = time.time() - t0
    sample_rounds = res_noswap["sample_rounds"]
    tail = max(1, sample_rounds.size // 3)
    cut_tail = res_noswap["traj_cut"][:, -tail:, :].mean(axis=(0, 1))
    amp_tail = res_noswap["traj_amp"][:, -tail:, :].mean(axis=(0, 1))
    # swap無 の trial 別ベスト分布(= 3本ランプ多スタートのみ・swap無しの対照群)。
    # swap有(res_v3)と pump_mults/ランプ/seed/trial数が完全一致し、do_swap だけが違う。
    v3_noswap_cuts = res_noswap["best_cuts"]
    print(f"  time={noswap_time:.2f}s  定常カット (mult低→高) = "
          f"[{cut_tail[0]:.1f}, {cut_tail[1]:.1f}, {cut_tail[2]:.1f}]")
    print(f"  [対照群] v3 swap無 best: mean={v3_noswap_cuts.mean():.1f}  "
          f"best={v3_noswap_cuts.max():.0f}  std={v3_noswap_cuts.std():.1f}")

    betas_norm = calibrate_betas(cut_tail, kappa_target=args.kappa_target)
    # v2 の教訓: 最良カットのレプリカを cold(β最大) に。
    if cut_tail[0] >= cut_tail[-1]:
        betas_v3 = betas_norm.max() - betas_norm     # 低 mult 側を cold
        direction = "低mult(ゆっくり昇温)=cold"
    else:
        betas_v3 = betas_norm                        # 高 mult 側を cold
        direction = "高mult(速く昇温)=cold"
    best_rep = int(np.argmax(cut_tail))
    print(f"  最良レプリカ index={best_rep}  → β割当: {direction}")
    print(f"  β v3 = [{betas_v3[0]:.3e}, {betas_v3[1]:.3e}, {betas_v3[2]:.3e}]")

    # ==== v3 PT (swap 有効) ====
    print(f"\n[CIM+PT v3(ランプ各レプリカ)] {NT} trials  swap/{args.swap_interval}")
    t0 = time.time()
    res_v3 = simulate_cim_pt_ramp_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds, num_trials=NT,
        seeds=pt_seeds, weights=w_arg, pump_mults=pump_mults,
        betas=betas_v3, do_swap=True, dP_per_round=dP,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    v3_time = time.time() - t0
    v3_cuts = res_v3["best_cuts"]
    v3_rate = res_v3["swap_rate"]
    print(f"  time={v3_time:.2f}s  mean={v3_cuts.mean():.1f}  best={v3_cuts.max():.0f}  "
          f"受理率={[round(x,3) for x in v3_rate.tolist()]}")

    # ==== 検証 ====
    def verify(res, cuts, name):
        bt = int(np.argmax(cuts))
        x = res["best_signs"][bt].astype(np.int64).tolist()
        rc = (sum(weights[i] for i, (a, b) in enumerate(edges) if x[a] != x[b])
              if use_weights else compute_cut_from_edges(x, edges))
        ok = abs(rc - cuts[bt]) < 1e-6
        print(f"[verify] {name}: kernel={cuts[bt]:.0f} 独立再計算={rc:.0f} 一致={ok}")
        return ok
    ok = verify(res_v2, v2_cuts, "v2固定反転") and verify(res_v3, v3_cuts, "v3ランプ")
    if not ok:
        raise SystemExit("検証失敗")

    # ==== サマリ ====
    # 主眼は「v3 swap無」vs「v3 swap有」の対照(swap の純粋効果)。
    results = {"ランプCIM": cim_cuts,
               "CIM+PT v2(固定ポンプ反転)": v2_cuts,
               "CIM+PT v3(swap無/3本ランプのみ)": v3_noswap_cuts,
               "CIM+PT v3(swap有)": v3_cuts}
    times = {"ランプCIM": cim_time,
             "CIM+PT v2(固定ポンプ反転)": v2_time,
             "CIM+PT v3(swap無/3本ランプのみ)": noswap_time,
             "CIM+PT v3(swap有)": v3_time}
    order = list(results.keys())
    print("\n" + "=" * 96)
    print(f"{'Method':<28} {'Ntrial':>7} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Time[s]':>9}")
    print("-" * 96)
    for name in order:
        c = results[name]
        line = (f"{name:<28} {c.size:>7d} {c.mean():>10.1f} {c.max():>10.1f} {c.min():>10.1f} "
                f"{c.std():>8.1f} {times[name]:>9.2f}")
        if known_best is not None:
            line += f"  ratio={c.max()/known_best:.4f}"
        print(line)
    print("=" * 96)

    # ==== 出力 ====
    kind_root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name.split("_", 1)[0][1:].isdigit():
            max_v = max(max_v, int(p.name.split("_", 1)[0][1:]))
    desc = [f"rounds{args.cim_rounds}", f"swap{args.swap_interval}"]
    if NT != 100:
        desc.append(f"trials{NT}")
    if args.tag:
        desc.append(args.tag)
    out_dir = kind_root / f"v{max_v + 1}_{'_'.join(desc)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    colors = {"ランプCIM": "#1f77b4", "CIM+PT v2(固定ポンプ反転)": "#2ca02c",
              "CIM+PT v3(swap無/3本ランプのみ)": "#9467bd",
              "CIM+PT v3(swap有)": "#d62728"}

    # --- Fig1: running best ---
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x_cim = np.arange(1, cim_cuts.size + 1)
    x_pt = np.arange(1, NT + 1) * NR
    ax.plot(x_cim, np.maximum.accumulate(cim_cuts), color=colors["ランプCIM"],
            linewidth=2.0, label=f"ランプCIM ({cim_time:.1f}s)")
    ax.plot(x_pt, np.maximum.accumulate(v2_cuts), color=colors["CIM+PT v2(固定ポンプ反転)"],
            linewidth=2.0, label=f"CIM+PT v2 固定ポンプ反転 ({v2_time:.1f}s)")
    ax.plot(x_pt, np.maximum.accumulate(v3_noswap_cuts), color=colors["CIM+PT v3(swap無/3本ランプのみ)"],
            linewidth=2.0, linestyle="--", label=f"CIM+PT v3 swap無 ({noswap_time:.1f}s)")
    ax.plot(x_pt, np.maximum.accumulate(v3_cuts), color=colors["CIM+PT v3(swap有)"],
            linewidth=2.4, label=f"CIM+PT v3 swap有 ({v3_time:.1f}s)")
    if known_best is not None:
        ax.axhline(known_best, color="red", linestyle="--", linewidth=1.2, label=f"既知ベスト {known_best}")
    ax.set_xlabel("計算量(レプリカ実行数 換算)", fontsize=LABEL_FS)
    ax.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax.set_title(f"等計算量での累積最良カット — swap 有/無の対照 ({graph_name})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "running_best.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'running_best.png'}")

    # --- Fig2: ヒストグラム ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    all_cuts = np.concatenate([cim_cuts, v2_cuts, v3_cuts])
    x_min = float(all_cuts.min()) - max(20, abs(all_cuts.min()) * 0.005)
    x_max = max(float(all_cuts.max()) + 20, (known_best + 10) if known_best else 0)
    bins = np.linspace(x_min, x_max, 30)
    for ax, name in zip(axes, order):
        c = results[name]
        ax.hist(c, bins=bins, color=colors[name], alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(c.mean(), color="black", linestyle=":", linewidth=1.2, label=f"平均 {c.mean():.0f}")
        if known_best is not None:
            ax.axvline(known_best, color="red", linestyle="--", linewidth=1.2, label=f"既知ベスト {known_best}")
        ax.set_title(f"{name}\n平均:{c.mean():.0f} 最良:{c.max():.0f}", fontsize=10)
        ax.set_xlabel("カット値", fontsize=LABEL_FS)
        ax.set_ylabel("頻度", fontsize=LABEL_FS)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ticks_in(ax)
    fig.suptitle(f"ランプCIM vs v2固定反転 vs v3各レプリカランプ — {graph_name} (等計算量)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'hist.png'}")

    # --- Fig3: レプリカ別ポンプ・ランプ軌跡 + 振幅(v3 swap無) ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))
    rep_colors = ["#d62728", "#ff7f0e", "#1f77b4"]
    kk = np.arange(1, args.cim_rounds + 1)
    for r in range(NR):
        axL.plot(kk, pump_mults[r] * kk * dP * 1e3, color=rep_colors[r], linewidth=2.0,
                 label=f"replica{r} mult={pump_mults[r]:.1f}")
    axL.axhline(p_th * 1e3, color="black", linestyle="--", linewidth=1.2, label=f"発振しきい値 {p_th*1e3:.0f}mW")
    axL.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axL.set_ylabel("ポンプ電力 P [mW]", fontsize=LABEL_FS)
    axL.set_title("各レプリカのポンプ・ランプ(時間発展)")
    axL.legend(loc="upper left", fontsize=9)
    axL.grid(alpha=0.3)
    ticks_in(axL)
    amp_mean = res_noswap["traj_amp"].mean(axis=0)
    for r in range(NR):
        axR.plot(sample_rounds, amp_mean[:, r], color=rep_colors[r], linewidth=2.0,
                 label=f"replica{r} mult={pump_mults[r]:.1f}")
    axR.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axR.set_ylabel("mean|c| (trial平均)", fontsize=LABEL_FS)
    axR.set_title("レプリカ別 平均振幅 — 順に凍結 (swap無効)")
    axR.legend(loc="upper left", fontsize=9)
    axR.grid(alpha=0.3)
    ticks_in(axR)
    fig.suptitle(f"v3: 各レプリカのランプと凍結の様子 ({graph_name})", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "ramps_amplitude.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'ramps_amplitude.png'}")

    # ==== summary.json ====
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges,
        "num_trials_pt": NT, "num_trials_cim": int(NR * NT),
        "equal_compute": True, "n_replicas": NR, "cim_rounds": args.cim_rounds,
        "pump_mults": pump_mults.tolist(), "p_th_mW": p_th * 1e3,
        "p_th_cross_round": cross, "dP_per_round_mW": dP * 1e3,
        "swap_interval": args.swap_interval, "kappa_target": args.kappa_target,
        "v3_tail_cut": cut_tail.tolist(), "v3_tail_mean_abs_c": amp_tail.tolist(),
        "v3_best_replica_index": best_rep, "v3_beta_direction": direction,
        "betas_v3": betas_v3.tolist(), "swap_rate_v3": v3_rate.tolist(),
        "betas_v2_fixed_reversed": betas_fixed_rev.tolist(),
        "swap_rate_v2": res_v2["swap_rate"].tolist(),
        "known_best": known_best,
        "ramp_CIM": {"mean": float(cim_cuts.mean()), "best": float(cim_cuts.max()),
                     "worst": float(cim_cuts.min()), "std": float(cim_cuts.std()),
                     "n_trial": int(cim_cuts.size), "time_s": cim_time},
        "CIM_PT_v2_fixed_reversed": {"mean": float(v2_cuts.mean()), "best": float(v2_cuts.max()),
                                     "worst": float(v2_cuts.min()), "std": float(v2_cuts.std()),
                                     "n_trial": int(v2_cuts.size), "time_s": v2_time},
        "CIM_PT_v3_perreplica_ramp": {"mean": float(v3_cuts.mean()), "best": float(v3_cuts.max()),
                                      "worst": float(v3_cuts.min()), "std": float(v3_cuts.std()),
                                      "n_trial": int(v3_cuts.size), "time_s": v3_time},
        "verify_ok": bool(ok),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "data.npz", cim=cim_cuts, v2=v2_cuts, v3=v3_cuts,
             v3_traj_best=res_v3["traj_best"], v3_traj_amp=res_noswap["traj_amp"],
             v3_traj_cut=res_noswap["traj_cut"], sample_rounds=sample_rounds,
             pump_mults=pump_mults, betas_v3=betas_v3)
    print(f"  saved: {out_dir / 'summary.json'}")
    print("\n[結論メモ]")
    print(f"  平均カット差 (対 ランプCIM): v2固定反転={v2_cuts.mean()-cim_cuts.mean():+.1f}, "
          f"v3ランプ={v3_cuts.mean()-cim_cuts.mean():+.1f}")
    print(f"  最良カット差 (対 ランプCIM): v2固定反転={v2_cuts.max()-cim_cuts.max():+.1f}, "
          f"v3ランプ={v3_cuts.max()-cim_cuts.max():+.1f}")


if __name__ == "__main__":
    main()
