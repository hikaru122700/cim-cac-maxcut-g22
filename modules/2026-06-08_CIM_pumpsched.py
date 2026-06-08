"""
CIM ポンプ関数最適化 — Phase 0(配列対応カーネル) + Phase 1(手設計族スイープ)
(2026-06-08)  計画: docs/20260608/1612_pump_function_optimization_plan.md

目的:
  ポンプを毎ラウンド `(k+1)*dP`(線形電力)でハードコードする代わりに、
  **任意のポンプ電力スケジュール配列 `P_sched[k]`(長さ K)** を受け取る一般化カーネルを作る。
  これで任意のポンプ形状(線形/べき乗/シグモイド/臨界減速…)が配列差分だけで試せる。

設計判断(計画より):
  - 最適化対象は P 軸(P/P_th)と g0 軸(g0/g0_th)の両方を別軸として比較。
    同じ正規化形状 s(τ) を「P/P_th に適用」と「g0/g0_th に適用」の 2 通りで作る。
  - 単一レプリカでポンプ形状の純粋効果を評価(multi-start・ノイズ schedule は後段)。

カーネルは modules/CIM.py の _simulate_cim_batch を正確にミラーし、唯一
  P_p = (k+1)*dP_per_round   →   P_p = P_sched[k]
だけを変える。よって P_sched[k]=(k+1)*dP を渡すと現行結果を seed 一致でビット再現する
(同一の in-kernel 式・同一 RNG 順序)。main() の冒頭でこれを assert で検証する。
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

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


# ============================================================
#  ポンプ電力スケジュール配列を受け取る CIM カーネル
#  (modules/CIM.py の _simulate_cim_batch を正確にミラー; P_p のみ配列化)
# ============================================================
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_sched_batch(
    n, num_rounds, num_trials,
    J_data, J_indices, J_indptr,
    edge_a, edge_b, edge_w,
    kappa, L, gamma, eta, bandwidth, photon_energy,
    P_sched,          # (num_rounds,) 各ラウンドのポンプ電力 [W]
    seeds,
):
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
            # ★ 唯一の変更点: ポンプ電力を配列から読む(現行は (k+1)*dP)
            P_p = P_sched[k]
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

        best_cuts_out[trial_idx] = best_cut
        for i in range(n):
            best_signs_out[trial_idx, i] = best_signs[i]

    return best_cuts_out, best_signs_out


def simulate_cim_sched_batch(
    n, J, edges, P_sched, num_trials,
    kappa, L, gamma, eta, bandwidth, photon_energy, seeds, weights=None,
):
    """ポンプ電力スケジュール P_sched(長さ=num_rounds)で CIM を num_trials 並列実行。"""
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    edge_w = (np.ones(edges_np.shape[0], dtype=np.float64) if weights is None
              else np.ascontiguousarray(np.asarray(weights, dtype=np.float64)))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    P_sched = np.ascontiguousarray(np.asarray(P_sched, dtype=np.float64))
    return _simulate_cim_sched_batch(
        n, int(P_sched.shape[0]), int(num_trials),
        J.data, J.indices, J.indptr, edge_a, edge_b, edge_w,
        float(kappa), float(L), float(gamma), float(eta),
        float(bandwidth), float(photon_energy), P_sched, seeds_arr)


# ============================================================
#  ポンプ関数の族(正規化形状 s(τ)∈[0,1] → P軸/g0軸に適用)
# ============================================================
def shape_s(family, tau, p=1.0, sig_a=12.0, dwell=0.5, band=0.12, s_star=0.5):
    """正規化形状 s(τ)∈[0,1](s(0+)≈0, s(1)=1, 単調増加)。"""
    if family == "linear":
        return tau.copy()
    if family == "power":
        return tau ** p
    if family == "sigmoid":  # 遅-速-遅(中央で加速)
        f = lambda t: 1.0 / (1.0 + np.exp(-sig_a * (t - 0.5)))
        return (f(tau) - f(0.0)) / (f(1.0) - f(0.0))
    if family == "slowdown":  # 高速→しきい近傍(s_star±band)で減速滞在→高速凍結
        s_lo = max(0.0, s_star - band)
        s_hi = min(1.0, s_star + band)
        t1, t2 = (1.0 - dwell) / 2.0, (1.0 + dwell) / 2.0
        s = np.empty_like(tau)
        for i, t in enumerate(tau):
            if t <= t1:
                s[i] = s_lo * (t / t1) if t1 > 0 else s_lo
            elif t <= t2:
                s[i] = s_lo + (s_hi - s_lo) * ((t - t1) / (t2 - t1))
            else:
                s[i] = s_hi + (1.0 - s_hi) * ((t - t2) / (1.0 - t2))
        return s
    raise ValueError(family)


def make_P_sched(K, axis, family, lo, hi, P_th, kappa, L, g0_th, **shape_kw):
    """正規化形状を P 軸 or g0 軸に適用してポンプ電力配列 P_sched[k] を作る。

    axis='P' : lo,hi は P/P_th。  P(k) = (lo + (hi-lo)·s)·P_th
    axis='g0': lo,hi は g0/g0_th。 g0(k) = (lo + (hi-lo)·s)·g0_th, P=(g0/2κL)^2
    s_star(しきい u=1 の位置)は端点から自動算出して slowdown の減速帯中心に使う。
    """
    tau = (np.arange(K) + 1) / K
    s_star = (1.0 - lo) / (hi - lo)  # 正規化値が 1(=しきい)になる s 位置
    s = shape_s(family, tau, s_star=s_star, **shape_kw)
    val = lo + (hi - lo) * s
    if axis == "P":
        return val * P_th
    elif axis == "g0":
        g0 = val * g0_th
        return (g0 / (2.0 * kappa * L)) ** 2
    raise ValueError(axis)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
    from modules.verify import compute_cut_from_edges

    KNOWN_BEST = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}
    EXPERIMENT_KIND = "cim_pumpsched"

    ap = argparse.ArgumentParser(description="CIM ポンプ関数 Phase0(再現サニティ)+Phase1(形状スイープ)")
    ap.add_argument("--graph", default="input/G22.txt")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--coupling", type=float, default=-0.03)
    args = ap.parse_args()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 2 * plt.rcParams["font.size"]

    def ticks_in(ax):
        ax.tick_params(direction="in", which="both", top=True, right=True)

    # 物理パラメータ(論文値; modules/CIM.py main と同一)
    kappa, L, gamma = 130.0, 0.05, 42.09
    eta = 10.0 ** (-1.1)
    bandwidth, photon_energy = 1.0e9, 1.28e-19
    dP = 0.05e-3
    K = args.rounds
    P_th = (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2
    g0_th = np.log(1.0 / eta)
    phys = dict(kappa=kappa, L=L, gamma=gamma, eta=eta,
                bandwidth=bandwidth, photon_energy=photon_energy)

    gp = Path(args.graph)
    graph_name = gp.stem
    n, k_edges, _adj, edges = load_graph(str(gp))
    J = build_coupling_matrix(n, edges, args.coupling)
    NT = args.trials
    seeds = np.arange(args.seed_base, args.seed_base + NT, dtype=np.int64)
    known = KNOWN_BEST.get(graph_name)
    print(f"Graph {graph_name} N={n} K={k_edges}  rounds={K} trials={NT}  P_th={P_th*1e3:.2f}mW")

    # 端点(線形ベースラインに一致させる)
    rP_lo, rP_hi = 1.0 * dP / P_th, K * dP / P_th        # P 軸: 0.0013 → 1.976
    u_lo, u_hi = np.sqrt(rP_lo), np.sqrt(rP_hi)           # g0 軸: 0.036 → 1.406

    # ===== Phase 0: 線形配列が現行カーネルを seed 一致でビット再現するか =====
    print("\n[Phase 0] サニティ: 線形 P_sched が現行 simulate_cim_batch を再現するか")
    P_linear = (np.arange(K) + 1) * dP
    cur_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=K, num_trials=NT, seeds=seeds,
        dP_per_round=dP, **phys)
    sched_cuts, _ = simulate_cim_sched_batch(n, J, edges, P_linear, NT, seeds=seeds, **phys)
    identical = bool(np.array_equal(cur_cuts, sched_cuts))
    print(f"  現行 mean={cur_cuts.mean():.1f}  sched mean={sched_cuts.mean():.1f}  "
          f"ビット一致={identical}")
    if not identical:
        diff = np.abs(cur_cuts - sched_cuts)
        print(f"  [WARN] 不一致 max|Δ|={diff.max()}  (fastmath 差の可能性)")
    assert identical, "Phase0 サニティ失敗: 線形配列が現行を再現しない"
    print("  [OK] 配列対応カーネルは現行ランプを完全再現")

    # ===== Phase 1: 手設計族スイープ(P軸/g0軸 × 各族, 端点一致) =====
    # baseline = P軸 linear (= 現行)。
    SHAPES = [
        ("P", "linear", {}, "P軸 linear (=現行ベースライン)"),
        ("P", "power", {"p": 0.5}, "P軸 power p=0.5 (早上げ)"),
        ("P", "power", {"p": 2.0}, "P軸 power p=2.0 (遅上げ)"),
        ("P", "sigmoid", {}, "P軸 sigmoid (遅速遅)"),
        ("P", "slowdown", {"dwell": 0.5, "band": 0.12}, "P軸 臨界減速"),
        ("g0", "linear", {}, "g0軸 linear"),
        ("g0", "power", {"p": 0.5}, "g0軸 power p=0.5"),
        ("g0", "power", {"p": 2.0}, "g0軸 power p=2.0"),
        ("g0", "sigmoid", {}, "g0軸 sigmoid"),
        ("g0", "slowdown", {"dwell": 0.5, "band": 0.12}, "g0軸 臨界減速"),
    ]
    print(f"\n[Phase 1] 手設計族スイープ {len(SHAPES)} 形状 (端点一致, 同一 seed)")
    rows = []
    for axis, family, kw, label in SHAPES:
        lo, hi = (rP_lo, rP_hi) if axis == "P" else (u_lo, u_hi)
        P_sched = make_P_sched(K, axis, family, lo, hi, P_th, kappa, L, g0_th, **kw)
        t0 = time.time()
        cuts, signs = simulate_cim_sched_batch(n, J, edges, P_sched, NT, seeds=seeds, **phys)
        dt = time.time() - t0
        # 検証: 最良 trial のカットを独立再計算
        bi = int(np.argmax(cuts))
        recut = compute_cut_from_edges(signs[bi].astype(np.int64).tolist(), edges)
        assert abs(recut - cuts[bi]) < 1e-6, f"{label}: カット不一致"
        rows.append(dict(axis=axis, family=family, kw=kw, label=label,
                         P_sched=P_sched, mean=float(cuts.mean()), best=float(cuts.max()),
                         worst=float(cuts.min()), std=float(cuts.std()), time=dt, cuts=cuts))
        print(f"  {label:<28} mean={cuts.mean():8.1f} best={cuts.max():7.0f} "
              f"std={cuts.std():5.1f}  ({dt:.1f}s)")

    base = rows[0]  # P軸 linear
    base_cuts = base["cuts"]
    # ★ ペア比較: 全形状が同一 seed なので、per-seed の差 d=形状−baseline を取ると
    #   seed 起因のばらつきが相殺され、平均差の標準誤差が大幅に縮む(鋭い有意性判定)。
    print(f"\n  baseline(P軸linear) mean={base['mean']:.1f} best={base['best']:.0f}  (N={NT} seeds)")
    print(f"  {'形状':<26} {'Δmean(ペア)':>11} {'SE':>6} {'z=Δ/SE':>8} {'有意|z|≥2':>9}")
    for r in rows[1:]:
        d = r["cuts"] - base_cuts
        dm = float(d.mean())
        dse = float(d.std(ddof=1) / np.sqrt(NT)) if NT > 1 else 0.0
        z = dm / dse if dse > 0 else 0.0
        r["d_mean_paired"] = dm
        r["d_se_paired"] = dse
        r["z"] = z
        r["sig"] = bool(abs(z) >= 2.0)
        print(f"  {r['label']:<26} {dm:+11.2f} {dse:6.2f} {z:+8.2f} {'★' if r['sig'] else '−':>9}")

    # ===== 出力 =====
    kind_root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name.split("_", 1)[0][1:].isdigit():
            max_v = max(max_v, int(p.name.split("_", 1)[0][1:]))
    out_dir = kind_root / f"v{max_v + 1}_rounds{K}_phase1_shapes"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    # Fig1: ポンプ軌跡 g0(k)/g0_th (P軸適用 / g0軸適用 の2パネル)
    tau = (np.arange(K) + 1) / K
    fig, (axP, axG) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    for r in rows:
        g0_traj = 2.0 * kappa * L * np.sqrt(r["P_sched"]) / g0_th
        ax = axP if r["axis"] == "P" else axG
        lw = 2.6 if r is base else 1.8
        ax.plot(np.arange(1, K + 1), g0_traj, linewidth=lw,
                label=r["label"].split(" ", 1)[1] if " " in r["label"] else r["label"])
    for ax, title in [(axP, "P 軸で形状適用"), (axG, "g0 軸で形状適用")]:
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0, label="発振しきい値")
        ax.set_xlabel("ラウンド数", fontsize=LABEL_FS)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
        ticks_in(ax)
    axP.set_ylabel("g0(k) / g0_th (分岐は 1)", fontsize=LABEL_FS)
    fig.suptitle(f"ポンプ関数の族 — 利得軌跡 ({graph_name}, 端点一致)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "pump_trajectories.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'pump_trajectories.png'}")

    # Fig2: 形状別 mean best_cut (±std), baseline 線
    fig, ax = plt.subplots(figsize=(12, 5.4))
    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    colors = ["#1f77b4" if r["axis"] == "P" else "#2ca02c" for r in rows]
    colors[0] = "#d62728"  # baseline
    xs = np.arange(len(rows))
    ax.bar(xs, means, yerr=stds, color=colors, alpha=0.8, edgecolor="black",
           linewidth=0.5, capsize=3)
    ax.axhline(base["mean"], color="#d62728", linestyle="--", linewidth=1.4,
               label=f"baseline 平均 {base['mean']:.0f}")
    if known:
        ax.axhline(known, color="gray", linestyle=":", linewidth=1.2, label=f"既知ベスト {known}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("平均 best_cut (±std)", fontsize=LABEL_FS)
    lo_y = min(means) - max(stds) - 5
    ax.set_ylim(lo_y, (known + 5) if known else max(means) + max(stds) + 5)
    ax.set_title(f"ポンプ形状別の平均カット ({graph_name}, {NT} seed, 等計算量)")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "shape_meancut.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'shape_meancut.png'}")

    # Fig3: ペアΔmean ±2SE (有意性の本命図; CI が 0 を跨げば非有意)
    others = rows[1:]
    ys = np.arange(len(others))
    dms = [r["d_mean_paired"] for r in others]
    errs = [2.0 * r["d_se_paired"] for r in others]
    cols = ["#2ca02c" if r["axis"] == "g0" else "#1f77b4" for r in others]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.6, label="baseline (P軸 linear)")
    for y, dm, er, c in zip(ys, dms, errs, cols):
        ax.errorbar(dm, y, xerr=er, fmt="o", color=c, ecolor="gray",
                    elinewidth=1.5, capsize=4, markersize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in others], fontsize=9)
    ax.set_xlabel("ペア Δmean (形状 − baseline) ±2SE", fontsize=LABEL_FS)
    ax.set_title(f"ポンプ形状のペア比較有意性 ({graph_name}, {NT} seed)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_significance.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'paired_significance.png'}")

    # summary.json
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges,
        "rounds": K, "trials": NT, "seed_base": args.seed_base,
        "P_th_mW": P_th * 1e3, "g0_th": g0_th,
        "endpoints": {"P_axis": [rP_lo, rP_hi], "g0_axis": [u_lo, u_hi]},
        "phase0_linear_reproduces_current": identical,
        "known_best": known,
        "baseline_label": base["label"],
        "results": [
            {"axis": r["axis"], "family": r["family"], "params": r["kw"], "label": r["label"],
             "mean": r["mean"], "best": r["best"], "worst": r["worst"], "std": r["std"],
             "d_mean_vs_baseline": r["mean"] - base["mean"],
             "d_best_vs_baseline": r["best"] - base["best"],
             "d_mean_paired": r.get("d_mean_paired"), "d_se_paired": r.get("d_se_paired"),
             "z": r.get("z"), "significant": r.get("sig"), "time_s": r["time"]}
            for r in rows
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  saved: {out_dir / 'summary.json'}")

    # 結論メモ
    best_shape = max(rows[1:], key=lambda r: r["mean"])
    print(f"\n[結論メモ] 平均最良の非線形形状: {best_shape['label']} "
          f"(Δmean={best_shape['mean']-base['mean']:+.1f}, Δbest={best_shape['best']-base['best']:+.0f})")
    print("  Δmean が std を超えるかで有意性を判断。次フェーズで Optuna 最適化＋held-out検証。")


if __name__ == "__main__":
    main()
