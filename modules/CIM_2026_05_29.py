"""
Coherent Ising Machine (CIM) + ICM ハイブリッド (2026-05-29 実装)

baseline (modules/CIM.py) との差分:
  - 1 trial を **2 レプリカ (A, B)** の CIM で同時に走らせる
  - 一定ラウンドごとに **ICM (Houdayer クラスター移動)** を挿入し、
    A/B 間で「符号が食い違う連結クラスター」の連続振幅 c_i を丸ごと交換する
  - これにより baseline CIM に欠けている **非局所ジャンプ** を注入し、
    連続力学の局所トラップ(振幅飽和後の停滞)から脱出させる

ICM の根拠 (PT_ICM.py と同じ Zhu-Ochoa-Katzgraber 2015 の Houdayer move):
  q_i = sign(c_A_i) * sign(c_B_i) が -1 のサイト(= 符号食い違い)を、
  グラフ上の連結成分(クラスター)として取り出す。クラスター境界は必ず
  q=+1 サイトに限られるため、クラスター内のスピンを A↔B で入れ替えると
  joint Ising エネルギー H_A + H_B が保存される (= 符号レベルで rejection-free)。
  CIM では ±1 ではなく連続振幅 c_i を入れ替えることで、符号反転に加えて
  振幅の大きさも持ち込み、停滞した片方のレプリカへ非局所な揺さぶりを与える。

物理モデル本体は baseline (Inoue & Yoshida, Optics Comm. 522 (2022) 128642) と同一。

既存 API (modules/CIM.py の simulate_cim_batch) は一切触らず、本ファイルは独立。
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


# ============================================================
#  Numba JIT コア: 2 レプリカ CIM + ICM を num_trials 並列実行
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_icm_batch(
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
    icm_interval: int,        # この round 間隔ごとに ICM を試行 (>=1)
    icm_start: int,           # この round 以降のみ ICM を行う (振幅形成待ち)
    num_clusters: int,        # 1 回の ICM イベントで動かすクラスター数
    sample_interval: int,     # best-so-far 記録の round 間隔 (>=1)
    num_samples: int,         # = num_rounds // sample_interval
    seeds: np.ndarray,
):
    """2 レプリカ CIM + ICM を num_trials 並列実行する内部ルーチン。

    Returns
    -------
    best_cuts  : (num_trials,)            各 trial の最終最良カット
    best_signs : (num_trials, n) bool     最良解の符号 (c>0)
    trajectory : (num_trials, num_samples) 各サンプル時点での best-so-far
    """
    best_cuts_out = np.zeros(num_trials, dtype=np.float64)
    best_signs_out = np.zeros((num_trials, n), dtype=np.bool_)
    trajectory = np.zeros((num_trials, num_samples), dtype=np.float64)

    sqrt_eta = np.sqrt(eta)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    num_edges = edge_a.shape[0]

    for trial_idx in prange(num_trials):
        np.random.seed(seeds[trial_idx])

        # 2 レプリカの状態
        c_A = np.zeros(n, dtype=np.float64)
        c_B = np.zeros(n, dtype=np.float64)
        Jc_A = np.zeros(n, dtype=np.float64)
        Jc_B = np.zeros(n, dtype=np.float64)

        best_signs = np.zeros(n, dtype=np.bool_)
        best_cut = -1.0e18

        # ICM 用 BFS バッファ
        visited = np.zeros(n, dtype=np.int8)
        queue = np.empty(n, dtype=np.int64)
        cluster = np.empty(n, dtype=np.int64)
        minus_sites = np.empty(n, dtype=np.int64)

        for k in range(num_rounds):
            # ---- ポンプパワー → 非飽和利得 ----
            P_p = (k + 1) * dP_per_round
            g0 = 2.0 * kappa * np.sqrt(P_p) * L
            half_g0 = 0.5 * g0
            neg_half_g0_gamma = -0.5 * g0 * gamma

            # ---- レプリカ A: matvec + 振幅更新 ----
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c_A[J_indices[jj]]
                Jc_A[i] = acc
            for i in range(n):
                coupled_in_i = sqrt_eta * c_A[i] + Jc_A[i]
                I_in_i = coupled_in_i * coupled_in_i
                half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                sqrt_G_I_i = np.exp(half_g_i)
                noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                c_A[i] = sqrt_G_I_i * coupled_in_i + noise_i

            # ---- レプリカ B ----
            for i in range(n):
                acc = 0.0
                start = J_indptr[i]
                end = J_indptr[i + 1]
                for jj in range(start, end):
                    acc += J_data[jj] * c_B[J_indices[jj]]
                Jc_B[i] = acc
            for i in range(n):
                coupled_in_i = sqrt_eta * c_B[i] + Jc_B[i]
                I_in_i = coupled_in_i * coupled_in_i
                half_g_i = half_g0 + neg_half_g0_gamma * I_in_i
                sqrt_G_I_i = np.exp(half_g_i)
                noise_i = np.random.standard_normal() * (noise_const * sqrt_G_I_i)
                c_B[i] = sqrt_G_I_i * coupled_in_i + noise_i

            # ---- 両レプリカの cut を評価し best 更新 ----
            cut_A = 0.0
            cut_B = 0.0
            for e in range(num_edges):
                a = edge_a[e]
                b = edge_b[e]
                w = edge_w[e]
                if (c_A[a] > 0.0) != (c_A[b] > 0.0):
                    cut_A += w
                if (c_B[a] > 0.0) != (c_B[b] > 0.0):
                    cut_B += w
            if cut_A > best_cut:
                best_cut = cut_A
                for i in range(n):
                    best_signs[i] = c_A[i] > 0.0
            if cut_B > best_cut:
                best_cut = cut_B
                for i in range(n):
                    best_signs[i] = c_B[i] > 0.0

            # ---- ICM (Houdayer クラスター移動) ----
            if (k + 1) >= icm_start and (k + 1) % icm_interval == 0:
                for _cl in range(num_clusters):
                    # 符号が食い違うサイト q_i = -1 を列挙
                    n_minus = 0
                    for i in range(n):
                        if (c_A[i] > 0.0) != (c_B[i] > 0.0):
                            minus_sites[n_minus] = i
                            n_minus += 1
                    if n_minus < 2:
                        break  # クラスターを作るほどの差異がない

                    # ランダムな食い違いサイトを種に連結成分を BFS
                    seed_node = minus_sites[np.random.randint(0, n_minus)]
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
                        start = J_indptr[v]
                        end = J_indptr[v + 1]
                        for jj in range(start, end):
                            u = J_indices[jj]
                            if visited[u] == 0 and ((c_A[u] > 0.0) != (c_B[u] > 0.0)):
                                visited[u] = 1
                                queue[qtail] = u
                                qtail += 1

                    # クラスター内サイトの連続振幅を A↔B でスワップ
                    # (符号食い違いサイトなので両レプリカの符号が同時に反転する)
                    for cc in range(n_cluster):
                        v = cluster[cc]
                        tmp = c_A[v]
                        c_A[v] = c_B[v]
                        c_B[v] = tmp

                # ICM 後に cut 再評価して best 更新
                cut_A = 0.0
                cut_B = 0.0
                for e in range(num_edges):
                    a = edge_a[e]
                    b = edge_b[e]
                    w = edge_w[e]
                    if (c_A[a] > 0.0) != (c_A[b] > 0.0):
                        cut_A += w
                    if (c_B[a] > 0.0) != (c_B[b] > 0.0):
                        cut_B += w
                if cut_A > best_cut:
                    best_cut = cut_A
                    for i in range(n):
                        best_signs[i] = c_A[i] > 0.0
                if cut_B > best_cut:
                    best_cut = cut_B
                    for i in range(n):
                        best_signs[i] = c_B[i] > 0.0

            # ---- trajectory サンプリング ----
            if (k + 1) % sample_interval == 0:
                sample_idx = (k + 1) // sample_interval - 1
                if 0 <= sample_idx < num_samples:
                    trajectory[trial_idx, sample_idx] = best_cut

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out, trajectory


def simulate_cim_icm_batch(
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
    dP_per_round: float,
    seeds: np.ndarray,
    *,
    icm_interval: int = 50,
    icm_start_frac: float = 0.3,
    num_clusters: int = 1,
    sample_interval: int | None = None,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CIM+ICM を num_trials 並列実行する公開 API。

    Parameters
    ----------
    n, J, edges, num_rounds, 物理パラメータ : baseline CIM と同じ
    num_trials : 並列 trial 数 (各 trial は内部で 2 レプリカを持つ)
    icm_interval : ICM を試行する round 間隔
    icm_start_frac : 全 round のうち最初のこの割合は ICM をスキップ
        (CIM の符号が形成されるまで待つ。0.3 = 最初の 30% は素の CIM)
    num_clusters : 1 ICM イベントあたり動かすクラスター数
    sample_interval : best-so-far を trajectory に記録する round 間隔。
        None なら num_rounds (最終値のみ)
    weights : MAX-CUT 重み (None なら全辺 +1)

    Returns
    -------
    best_cuts  : (num_trials,)
    best_signs : (num_trials, n) bool
    trajectory : (num_trials, num_samples)  サンプル時刻は
        (sample_interval, 2*sample_interval, ..., num_rounds)
    """
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    if weights is None:
        edge_w = np.ones(edges_np.shape[0], dtype=np.float64)
    else:
        edge_w = np.ascontiguousarray(np.asarray(weights, dtype=np.float64))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))

    if sample_interval is None or sample_interval <= 0:
        sample_interval = int(num_rounds)
    sample_interval = int(sample_interval)
    num_samples = int(num_rounds) // sample_interval
    if num_samples < 1:
        num_samples = 1
        sample_interval = int(num_rounds)

    icm_start = int(round(icm_start_frac * num_rounds))
    if icm_start < 1:
        icm_start = 1

    best_cuts, best_signs, trajectory = _simulate_cim_icm_batch(
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
        int(icm_interval),
        int(icm_start),
        int(max(1, num_clusters)),
        sample_interval,
        num_samples,
        seeds_arr,
    )
    return best_cuts, best_signs, trajectory


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

    EXPERIMENT_KIND = "cim_icm"
    KNOWN_BEST: dict[str, int] = {
        "G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591,
    }

    parser = argparse.ArgumentParser(description="CIM vs CIM+ICM 比較ベンチ")
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--cim-rounds", type=int, default=1500)
    parser.add_argument("--cim-coupling", type=float, default=-0.03)
    parser.add_argument("--icm-interval", type=int, default=50)
    parser.add_argument("--icm-start-frac", type=float, default=0.3)
    parser.add_argument("--num-clusters", type=int, default=1)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--known-best", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    # ---- プロットスタイル (プロジェクト共通規約) ----
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

    # ==== baseline CIM ====
    print(f"\n[CIM] {args.num_trials} trials  rounds={args.cim_rounds}")
    t0 = time.time()
    cim_cuts, _cim_signs = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=args.num_trials, seeds=seeds, weights=w_arg, **cim_params,
    )
    cim_time = time.time() - t0
    print(f"  time={cim_time:.2f}s  mean={cim_cuts.mean():.1f}  best={cim_cuts.max():.0f}")

    # ==== CIM + ICM ====
    print(f"\n[CIM+ICM] {args.num_trials} trials  rounds={args.cim_rounds}  "
          f"icm/{args.icm_interval}  start_frac={args.icm_start_frac}  clusters={args.num_clusters}")
    t0 = time.time()
    icm_cuts, icm_signs, icm_traj = simulate_cim_icm_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=args.num_trials, seeds=seeds, weights=w_arg,
        icm_interval=args.icm_interval, icm_start_frac=args.icm_start_frac,
        num_clusters=args.num_clusters, sample_interval=args.sample_interval,
        **cim_params,
    )
    icm_time = time.time() - t0
    sample_rounds = np.arange(1, icm_traj.shape[1] + 1) * args.sample_interval
    print(f"  time={icm_time:.2f}s  mean={icm_cuts.mean():.1f}  best={icm_cuts.max():.0f}")

    # ==== 検証: 最良解の符号からカットを独立再計算して突き合わせ ====
    best_trial = int(np.argmax(icm_cuts))
    x_best = icm_signs[best_trial].astype(np.int64).tolist()
    if use_weights:
        recut = sum(weights[i] for i, (a, b) in enumerate(edges) if x_best[a] != x_best[b])
    else:
        recut = compute_cut_from_edges(x_best, edges)
    ok = abs(recut - icm_cuts[best_trial]) < 1e-6
    print(f"\n[verify] best trial={best_trial}  kernel cut={icm_cuts[best_trial]:.0f}  "
          f"独立再計算={recut:.0f}  一致={ok}")
    if not ok:
        raise SystemExit("検証失敗: カーネルのカット値が独立計算と一致しません")

    # ==== サマリ ====
    results = {"CIM": cim_cuts, "CIM+ICM": icm_cuts}
    times = {"CIM": cim_time, "CIM+ICM": icm_time}
    print("\n" + "=" * 78)
    print(f"{'Method':<10} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Time[s]':>10}")
    print("-" * 78)
    for name in ["CIM", "CIM+ICM"]:
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
    desc_parts = [f"rounds{args.cim_rounds}", f"icm{args.icm_interval}"]
    if args.num_trials != 100:
        desc_parts.append(f"trials{args.num_trials}")
    if args.tag:
        desc_parts.append(args.tag)
    out_dir = kind_root / f"v{max_v + 1}_{'_'.join(desc_parts)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    colors = {"CIM": "#1f77b4", "CIM+ICM": "#2ca02c"}

    # --- Fig1: ヒストグラム ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    all_cuts = np.concatenate([cim_cuts, icm_cuts])
    x_min = float(all_cuts.min()) - max(20, abs(all_cuts.min()) * 0.005)
    x_max = float(all_cuts.max()) + max(20, abs(all_cuts.max()) * 0.005)
    if known_best is not None:
        x_max = max(x_max, known_best + 10)
    bins = np.linspace(x_min, x_max, 30)
    for ax, name in zip(axes, ["CIM", "CIM+ICM"]):
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
    fig1.suptitle(f"CIM vs CIM+ICM — {graph_name} (各 {args.num_trials} trial)", fontsize=13)
    fig1.tight_layout()
    fig1.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig1)
    print(f"  saved: {out_dir / 'hist.png'}")

    # --- Fig2: running best ---
    fig2, ax2 = plt.subplots(figsize=(10, 5.4))
    for name in ["CIM", "CIM+ICM"]:
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

    # --- Fig3: CIM+ICM 収束軌跡 ---
    fig3, ax3 = plt.subplots(figsize=(10, 5.4))
    traj_mean = icm_traj.mean(axis=0)
    traj_best = icm_traj.max(axis=0)
    traj_p10 = np.percentile(icm_traj, 10, axis=0)
    traj_p90 = np.percentile(icm_traj, 90, axis=0)
    ax3.fill_between(sample_rounds, traj_p10, traj_p90, color=colors["CIM+ICM"],
                     alpha=0.18, label="CIM+ICM 10–90%ile")
    ax3.plot(sample_rounds, traj_mean, color=colors["CIM+ICM"], linewidth=2.2,
             label=f"CIM+ICM 平均 (最終 {traj_mean[-1]:.0f})")
    ax3.plot(sample_rounds, traj_best, color=colors["CIM+ICM"], linewidth=1.5,
             linestyle="--", label=f"CIM+ICM 最良 (最終 {traj_best[-1]:.0f})")
    ax3.axhline(cim_cuts.mean(), color=colors["CIM"], linestyle=":", linewidth=1.6,
                label=f"CIM 平均 {cim_cuts.mean():.0f}")
    ax3.axhline(cim_cuts.max(), color=colors["CIM"], linestyle="-.", linewidth=1.4,
                label=f"CIM 最良 {cim_cuts.max():.0f}")
    icm_x = args.icm_start_frac * args.cim_rounds
    ax3.axvline(icm_x, color="gray", linestyle="--", linewidth=1.0, label=f"ICM 開始 (round {icm_x:.0f})")
    if known_best is not None:
        ax3.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax3.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    ax3.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax3.set_title(f"CIM+ICM の収束軌跡 ({graph_name}, {args.num_trials} trial)")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(alpha=0.3)
    ticks_in(ax3)
    fig3.tight_layout()
    fig3.savefig(out_dir / "trajectory.png", dpi=150)
    plt.close(fig3)
    print(f"  saved: {out_dir / 'trajectory.png'}")

    # ==== サマリ JSON + 生カット ====
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges,
        "num_trials": args.num_trials, "cim_rounds": args.cim_rounds,
        "icm_interval": args.icm_interval, "icm_start_frac": args.icm_start_frac,
        "num_clusters": args.num_clusters, "known_best": known_best,
        "CIM": {"mean": float(cim_cuts.mean()), "best": float(cim_cuts.max()),
                "worst": float(cim_cuts.min()), "std": float(cim_cuts.std()), "time_s": cim_time},
        "CIM+ICM": {"mean": float(icm_cuts.mean()), "best": float(icm_cuts.max()),
                    "worst": float(icm_cuts.min()), "std": float(icm_cuts.std()), "time_s": icm_time},
        "verify_ok": bool(ok),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "cuts.npz", cim=cim_cuts, icm=icm_cuts, traj=icm_traj,
             sample_rounds=sample_rounds)
    print(f"  saved: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
