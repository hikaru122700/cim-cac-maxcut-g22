"""
Coherent Ising Machine (CIM) + PT (Parallel Tempering) ハイブリッド (2026-05-29 実装)

baseline (modules/CIM.py) との差分:
  - ポンプをランプさせず、**固定ポンプ利得の 3 レプリカ**を同時に回す。
    各レプリカは CIM の発振しきい値 P_th を基準にした 3 領域に対応する:
      replica 0 (ノイズ支配 = 高温) : P < P_th  → 信号は減衰し符号が揺らぐ
      replica 1 (急増 = 臨界)       : P ≈ P_th  → 符号形成の途中
      replica 2 (高止まり = 低温)   : P > P_th  → 振幅が飽和してロック
  - 一定ラウンドごとに **PT スワップ**: 隣接レプリカ (0-1, 1-2) の振幅ベクトル
    c を確率 p_swap で**無条件交換**する(確率固定スワップ)。
  - annealing(温度を下げて凍結)に相当する役割は PT スワップが担う:
    高温レプリカが探索した配置を低温レプリカへ流して凍結・評価する。

発振しきい値の物理:
  ループ 1 周の正味利得が損失を上回ると発振する。非飽和では η·exp(g0) = 1、
  すなわち g0 = ln(1/η)。利得係数は g0 = 2·κ·√(P_p)·L なので
    P_th = ( ln(1/η) / (2·κ·L) )^2
  これを基準に pump_mults = [<1, ≈1, >1] を掛けて 3 段のポンプを作る。

物理モデル本体は baseline (Inoue & Yoshida, Optics Comm. 522 (2022) 128642) と同一。
ICM は入れない(PT 機能のみ)。modules/CIM.py は一切変更しない。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


NR = 3  # レプリカ数(3 領域に対応、固定)


@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_pt_batch(
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
    pump_levels: np.ndarray,    # (NR,) 各レプリカの固定ポンプパワー [W] 昇順
    swap_interval: int,         # この round 間隔ごとにスワップ試行 (>=1)
    p_swap: float,              # 隣接ペアの交換確率 (0..1)
    sample_interval: int,       # 記録の round 間隔 (>=1)
    num_samples: int,           # = num_rounds // sample_interval
    seeds: np.ndarray,
):
    """固定ポンプ 3 レプリカ CIM + 確率固定 PT を num_trials 並列実行。

    Returns
    -------
    best_cuts  : (num_trials,)                  各 trial の最終最良カット
    best_signs : (num_trials, n) bool           最良解の符号 (c>0)
    traj_best  : (num_trials, num_samples)       best-so-far 軌跡
    traj_amp   : (num_trials, num_samples, NR)   レプリカ別 mean|c| 軌跡
    traj_cut   : (num_trials, num_samples, NR)   レプリカ別 現在カット 軌跡
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)
    traj_best = np.zeros((num_trials, num_samples), dtype=np.float64)
    traj_amp = np.zeros((num_trials, num_samples, NR), dtype=np.float64)
    traj_cut = np.zeros((num_trials, num_samples, NR), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    # 各レプリカのポンプは固定 → 利得係数も全 round 共通で 1 回だけ計算
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

        for k in range(num_rounds):
            # ---- 各レプリカを 1 ラウンド発展(固定ポンプ) ----
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

            # ---- 各レプリカの cut 評価 + best 更新 ----
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

            # ---- PT スワップ(隣接ペアを確率 p_swap で無条件交換) ----
            if (k + 1) % swap_interval == 0:
                for r in range(NR - 1):
                    if np.random.random() < p_swap:
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

    return best_cuts_out, best_signs_out, traj_best, traj_amp, traj_cut


def compute_threshold_pump(kappa: float, L: float, eta: float) -> float:
    """CIM 発振しきい値ポンプ P_th = (ln(1/η) / (2κL))^2 [W]。"""
    return (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2


def simulate_cim_pt_batch(
    n: int,
    J,                                  # scipy.sparse.csr_matrix
    edges: list[tuple[int, int]],
    num_rounds: int,
    num_trials: int,
    kappa: float,
    L: float,
    gamma: float,
    eta: float,
    bandwidth: float,
    photon_energy: float,
    seeds: np.ndarray,
    *,
    pump_levels: np.ndarray | None = None,
    pump_mults: tuple[float, float, float] = (0.5, 1.0, 2.5),
    swap_interval: int = 10,
    p_swap: float = 0.5,
    sample_interval: int | None = None,
    weights: list[float] | None = None,
) -> dict:
    """固定ポンプ 3 段 CIM + 確率固定 PT を num_trials 並列実行する公開 API。

    pump_levels を直接渡すか、未指定なら P_th × pump_mults で 3 段を生成。
    返り値は dict(best_cuts, best_signs, traj_best, traj_amp, traj_cut,
    pump_levels, p_th, sample_rounds)。
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    p_th = compute_threshold_pump(kappa, L, eta)
    if pump_levels is None:
        pump_levels = np.array([m * p_th for m in pump_mults], dtype=np.float64)
    pump_levels = np.ascontiguousarray(np.sort(np.asarray(pump_levels, dtype=np.float64)))

    if sample_interval is None or sample_interval <= 0:
        sample_interval = int(num_rounds)
    sample_interval = int(sample_interval)
    num_samples = int(num_rounds) // sample_interval
    if num_samples < 1:
        num_samples = 1
        sample_interval = int(num_rounds)

    best_cuts, best_signs, traj_best, traj_amp, traj_cut = _simulate_cim_pt_batch(
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
        pump_levels,
        int(swap_interval),
        float(p_swap),
        sample_interval,
        num_samples,
        seeds_arr,
    )
    sample_rounds = np.arange(1, num_samples + 1) * sample_interval
    return {
        "best_cuts": best_cuts,
        "best_signs": best_signs,
        "traj_best": traj_best,
        "traj_amp": traj_amp,
        "traj_cut": traj_cut,
        "pump_levels": pump_levels,
        "p_th": p_th,
        "sample_rounds": sample_rounds,
    }


# ============================================================
#  検証 + 比較ベンチ (main)
# ============================================================
def main() -> None:
    import argparse
    import json
    import sys
    import time
    from datetime import date
    from pathlib import Path

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
    from modules.verify import compute_cut_from_edges

    EXPERIMENT_KIND = "cim_pt"
    KNOWN_BEST: dict[str, int] = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}

    parser = argparse.ArgumentParser(description="CIM vs CIM+PT(固定ポンプ3段) 比較ベンチ")
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--cim-rounds", type=int, default=1500)
    parser.add_argument("--cim-coupling", type=float, default=-0.03)
    parser.add_argument("--pump-mults", type=float, nargs=3, default=[0.5, 1.0, 2.5],
                        help="P_th に対する 3 レプリカのポンプ倍率 (高温→低温)")
    parser.add_argument("--swap-interval", type=int, default=10)
    parser.add_argument("--p-swap", type=float, default=0.5)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--known-best", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 2 * plt.rcParams["font.size"]   # 軸ラベルは既定の 2 倍

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

    seeds = np.arange(args.seed_base, args.seed_base + args.num_trials, dtype=np.int64)
    J = build_coupling_matrix(n, edges, args.cim_coupling, weights=w_arg)
    cim_params = dict(
        kappa=130.0, L=0.05, gamma=42.09, eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    )
    pt_phys = {k: v for k, v in cim_params.items() if k != "dP_per_round"}

    p_th = compute_threshold_pump(cim_params["kappa"], cim_params["L"], cim_params["eta"])
    pump_levels = np.sort(np.array([m * p_th for m in args.pump_mults]))
    print(f"P_th = {p_th * 1e3:.3f} mW  → pump_levels (mW) = "
          f"{[round(p * 1e3, 3) for p in pump_levels]}  (mults={sorted(args.pump_mults)})")

    # ==== baseline CIM (ランプあり) ====
    print(f"\n[CIM] {args.num_trials} trials  rounds={args.cim_rounds}")
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=args.num_trials, seeds=seeds, weights=w_arg, **cim_params,
    )
    cim_time = time.time() - t0
    print(f"  time={cim_time:.2f}s  mean={cim_cuts.mean():.1f}  best={cim_cuts.max():.0f}")

    # ==== CIM + PT (固定ポンプ3段) ====
    print(f"\n[CIM+PT] {args.num_trials} trials  rounds={args.cim_rounds}  "
          f"swap/{args.swap_interval}  p_swap={args.p_swap}")
    t0 = time.time()
    res = simulate_cim_pt_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=args.num_trials, seeds=seeds, weights=w_arg,
        pump_levels=pump_levels, swap_interval=args.swap_interval,
        p_swap=args.p_swap, sample_interval=args.sample_interval, **pt_phys,
    )
    pt_time = time.time() - t0
    pt_cuts = res["best_cuts"]
    sample_rounds = res["sample_rounds"]
    print(f"  time={pt_time:.2f}s  mean={pt_cuts.mean():.1f}  best={pt_cuts.max():.0f}")

    # ==== 領域characterization (swap無効) ====
    # スワップを入れると各スロットに様々な config が流入し、スロット別の
    # mean|c| 時間平均が均されて 3 領域が見えなくなる。固定ポンプが作る
    # 3 領域(ノイズ支配/臨界/飽和)は swap を切った素の発展で最も明瞭に
    # 現れるので、領域の同定と可視化はこの参照 runで行う。最適化比較は
    # 上の swap 有効 run(res)で行う。
    reg_trials = min(args.num_trials, 8)
    print(f"\n[regime] {reg_trials} trials  swap無効  (3領域の同定用)")
    res_reg = simulate_cim_pt_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=reg_trials, seeds=seeds[:reg_trials], weights=w_arg,
        pump_levels=pump_levels, swap_interval=args.swap_interval,
        p_swap=0.0, sample_interval=args.sample_interval, **pt_phys,
    )

    # ==== 検証1: 最良解の符号からカットを独立再計算 ====
    best_trial = int(np.argmax(pt_cuts))
    x_best = res["best_signs"][best_trial].astype(np.int64).tolist()
    if use_weights:
        recut = sum(weights[i] for i, (a, b) in enumerate(edges) if x_best[a] != x_best[b])
    else:
        recut = compute_cut_from_edges(x_best, edges)
    ok = abs(recut - pt_cuts[best_trial]) < 1e-6
    print(f"\n[verify-1] best trial={best_trial}  kernel cut={pt_cuts[best_trial]:.0f}  "
          f"独立再計算={recut:.0f}  一致={ok}")
    if not ok:
        raise SystemExit("検証失敗: カーネルのカット値が独立計算と一致しません")

    # ==== 検証2: 3レプリカが 3 領域(mean|c|)に分離しているか ====
    # 領域同定 run(swap無効)の後半(定常)での mean|c| の trial 平均。
    tail = max(1, sample_rounds.size // 3)
    amp_tail = res_reg["traj_amp"][:, -tail:, :].mean(axis=(0, 1))  # (NR,)
    cut_tail = res_reg["traj_cut"][:, -tail:, :].mean(axis=(0, 1))  # (NR,)
    print(f"[verify-2] 定常 mean|c| (replica 0→2) = "
          f"[{amp_tail[0]:.4f}, {amp_tail[1]:.4f}, {amp_tail[2]:.4f}]")
    print(f"           定常 cut    (replica 0→2) = "
          f"[{cut_tail[0]:.1f}, {cut_tail[1]:.1f}, {cut_tail[2]:.1f}]")
    amp_monotone = amp_tail[0] < amp_tail[1] < amp_tail[2]
    print(f"           振幅が高温<臨界<低温の順に増加: {amp_monotone}")

    # ==== サマリ ====
    results = {"CIM": cim_cuts, "CIM+PT": pt_cuts}
    times = {"CIM": cim_time, "CIM+PT": pt_time}
    print("\n" + "=" * 78)
    print(f"{'Method':<10} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Time[s]':>10}")
    print("-" * 78)
    for name in ["CIM", "CIM+PT"]:
        c = results[name]
        line = (f"{name:<10} {c.mean():>10.1f} {c.max():>10.1f} {c.min():>10.1f} "
                f"{c.std():>8.1f} {times[name]:>10.2f}")
        if known_best is not None:
            line += f"  ratio={c.max() / known_best:.4f}"
        print(line)
    print("=" * 78)

    # ==== 出力ディレクトリ (results 規約) ====
    kind_root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    desc_parts = [f"rounds{args.cim_rounds}", f"swap{args.swap_interval}"]
    if args.num_trials != 100:
        desc_parts.append(f"trials{args.num_trials}")
    if args.tag:
        desc_parts.append(args.tag)
    out_dir = kind_root / f"v{max_v + 1}_{'_'.join(desc_parts)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    colors = {"CIM": "#1f77b4", "CIM+PT": "#9467bd"}
    rep_colors = ["#d62728", "#ff7f0e", "#1f77b4"]  # 高温→臨界→低温
    rep_labels = ["replica0 ノイズ支配(高温)", "replica1 臨界(中温)", "replica2 飽和(低温)"]

    # --- Fig1: ヒストグラム ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    all_cuts = np.concatenate([cim_cuts, pt_cuts])
    x_min = float(all_cuts.min()) - max(20, abs(all_cuts.min()) * 0.005)
    x_max = float(all_cuts.max()) + max(20, abs(all_cuts.max()) * 0.005)
    if known_best is not None:
        x_max = max(x_max, known_best + 10)
    bins = np.linspace(x_min, x_max, 30)
    for ax, name in zip(axes, ["CIM", "CIM+PT"]):
        c = results[name]
        ax.hist(c, bins=bins, color=colors[name], alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(c.mean(), color="black", linestyle=":", linewidth=1.2, label=f"平均 {c.mean():.0f}")
        if known_best is not None:
            ax.axvline(known_best, color="red", linestyle="--", linewidth=1.2,
                       label=f"既知ベスト {known_best}")
        ax.set_title(f"{name}  時間:{times[name]:.1f}s  平均:{c.mean():.0f}  最良:{c.max():.0f}", fontsize=11)
        ax.set_xlabel("カット値", fontsize=LABEL_FS)
        ax.set_ylabel("頻度", fontsize=LABEL_FS)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")
        ticks_in(ax)
    fig1.suptitle(f"CIM vs CIM+PT — {graph_name} (各 {args.num_trials} trial)", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig1)
    print(f"  saved: {out_dir / 'hist.png'}")

    # --- Fig2: running best ---
    fig2, ax2 = plt.subplots(figsize=(10, 5.4))
    for name in ["CIM", "CIM+PT"]:
        running = np.maximum.accumulate(results[name])
        ax2.plot(np.arange(1, args.num_trials + 1), running, color=colors[name],
                 linewidth=2.0, label=f"{name} ({times[name]:.1f}s)")
    if known_best is not None:
        ax2.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax2.set_xlabel("trial 数", fontsize=LABEL_FS)
    ax2.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax2.set_title(f"trial 数に対する累積最良カット ({graph_name})")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    ticks_in(ax2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "running_best.png", dpi=150)
    plt.close(fig2)
    print(f"  saved: {out_dir / 'running_best.png'}")

    # --- Fig3: best-so-far 収束軌跡 ---
    fig3, ax3 = plt.subplots(figsize=(10, 5.4))
    tb_mean = res["traj_best"].mean(axis=0)
    tb_best = res["traj_best"].max(axis=0)
    tb_p10 = np.percentile(res["traj_best"], 10, axis=0)
    tb_p90 = np.percentile(res["traj_best"], 90, axis=0)
    ax3.fill_between(sample_rounds, tb_p10, tb_p90, color=colors["CIM+PT"], alpha=0.18,
                     label="CIM+PT 10–90%ile")
    ax3.plot(sample_rounds, tb_mean, color=colors["CIM+PT"], linewidth=2.2,
             label=f"CIM+PT 平均 (最終 {tb_mean[-1]:.0f})")
    ax3.plot(sample_rounds, tb_best, color=colors["CIM+PT"], linewidth=1.5, linestyle="--",
             label=f"CIM+PT 最良 (最終 {tb_best[-1]:.0f})")
    ax3.axhline(cim_cuts.mean(), color=colors["CIM"], linestyle=":", linewidth=1.6,
                label=f"CIM 平均 {cim_cuts.mean():.0f}")
    ax3.axhline(cim_cuts.max(), color=colors["CIM"], linestyle="-.", linewidth=1.4,
                label=f"CIM 最良 {cim_cuts.max():.0f}")
    if known_best is not None:
        ax3.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax3.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    ax3.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax3.set_title(f"CIM+PT の収束軌跡 ({graph_name}, {args.num_trials} trial)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(alpha=0.3)
    ticks_in(ax3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "trajectory.png", dpi=150)
    plt.close(fig3)
    print(f"  saved: {out_dir / 'trajectory.png'}")

    # --- Fig4: 3レプリカの振幅領域 + カット (検証2 の可視化, swap無効 run) ---
    amp_mean = res_reg["traj_amp"].mean(axis=0)  # (num_samples, NR)
    cut_mean = res_reg["traj_cut"].mean(axis=0)  # (num_samples, NR)
    fig4, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0))
    for r in range(NR):
        axL.plot(sample_rounds, amp_mean[:, r], color=rep_colors[r], linewidth=2.0,
                 label=f"{rep_labels[r]}  P={pump_levels[r] * 1e3:.2f}mW")
    axL.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axL.set_ylabel("mean|c| (trial平均)", fontsize=LABEL_FS)
    axL.set_title("レプリカ別 平均振幅 — 3領域への分離 (swap無効)")
    axL.legend(loc="upper left", fontsize=9)
    axL.grid(alpha=0.3)
    ticks_in(axL)
    for r in range(NR):
        axR.plot(sample_rounds, cut_mean[:, r], color=rep_colors[r], linewidth=2.0,
                 label=f"{rep_labels[r]}")
    if known_best is not None:
        axR.axhline(known_best, color="red", linestyle="--", linewidth=1.2, label=f"既知ベスト {known_best}")
    axR.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axR.set_ylabel("現在カット (trial平均)", fontsize=LABEL_FS)
    axR.set_title("レプリカ別 現在カット (swap無効)")
    axR.legend(loc="lower right", fontsize=9)
    axR.grid(alpha=0.3)
    ticks_in(axR)
    fig4.suptitle(f"PT 3 レプリカの動作領域 ({graph_name}, swap無効 {reg_trials}trial)", fontsize=13)
    fig4.tight_layout()
    fig4.savefig(out_dir / "amplitude_regimes.png", dpi=150)
    plt.close(fig4)
    print(f"  saved: {out_dir / 'amplitude_regimes.png'}")

    # --- Fig5: swap有効時の振幅推移 (混合の様子) ---
    # res = swap有効 run。trial平均(左)は各スロットへ様々なconfigが流入し
    # 均されて 3 本が重なる。代表 1 trial の生トレース(右)では swap_interval
    # ごとにスロット間で振幅が飛び移る(= 隣接ペア交換)様子が見える。
    amp_on_mean = res["traj_amp"].mean(axis=0)  # (num_samples, NR)
    rep_trial = int(np.argmax(pt_cuts))         # 最良 trial を代表に
    amp_on_one = res["traj_amp"][rep_trial]      # (num_samples, NR)
    fig5, (axL5, axR5) = plt.subplots(1, 2, figsize=(13, 5.0))
    for r in range(NR):
        axL5.plot(sample_rounds, amp_on_mean[:, r], color=rep_colors[r], linewidth=2.0,
                  label=f"{rep_labels[r]}  P={pump_levels[r] * 1e3:.2f}mW")
    axL5.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axL5.set_ylabel("mean|c| (trial平均)", fontsize=LABEL_FS)
    axL5.set_title(f"swap有効 — 全{args.num_trials}trial平均(均されて重なる)")
    axL5.legend(loc="upper left", fontsize=9)
    axL5.grid(alpha=0.3)
    ticks_in(axL5)
    for r in range(NR):
        axR5.plot(sample_rounds, amp_on_one[:, r], color=rep_colors[r], linewidth=1.4,
                  label=f"{rep_labels[r]}")
    axR5.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    axR5.set_ylabel("mean|c|", fontsize=LABEL_FS)
    axR5.set_title(f"swap有効 — 代表1trial(#{rep_trial})の生トレース")
    axR5.legend(loc="upper left", fontsize=9)
    axR5.grid(alpha=0.3)
    ticks_in(axR5)
    fig5.suptitle(f"PT スワップ有効時の振幅推移 ({graph_name}, swap/{args.swap_interval} "
                  f"p={args.p_swap})", fontsize=13)
    fig5.tight_layout()
    fig5.savefig(out_dir / "amplitude_swapon.png", dpi=150)
    plt.close(fig5)
    print(f"  saved: {out_dir / 'amplitude_swapon.png'}")

    # ==== サマリ JSON + 生データ ====
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges,
        "num_trials": args.num_trials, "cim_rounds": args.cim_rounds,
        "pump_mults": sorted(args.pump_mults), "p_th_mW": p_th * 1e3,
        "pump_levels_mW": [p * 1e3 for p in pump_levels.tolist()],
        "swap_interval": args.swap_interval, "p_swap": args.p_swap,
        "known_best": known_best,
        "CIM": {"mean": float(cim_cuts.mean()), "best": float(cim_cuts.max()),
                "worst": float(cim_cuts.min()), "std": float(cim_cuts.std()), "time_s": cim_time},
        "CIM+PT": {"mean": float(pt_cuts.mean()), "best": float(pt_cuts.max()),
                   "worst": float(pt_cuts.min()), "std": float(pt_cuts.std()), "time_s": pt_time},
        "regime_trials_noswap": reg_trials,
        "tail_mean_abs_c": amp_tail.tolist(),
        "tail_cut": cut_tail.tolist(),
        "amp_monotone": bool(amp_monotone),
        "verify_ok": bool(ok),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "data.npz", cim=cim_cuts, pt=pt_cuts,
             traj_best=res["traj_best"], traj_amp=res["traj_amp"], traj_cut=res["traj_cut"],
             sample_rounds=sample_rounds, pump_levels=pump_levels)
    print(f"  saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
