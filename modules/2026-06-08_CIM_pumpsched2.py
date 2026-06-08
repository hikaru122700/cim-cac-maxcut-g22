"""
CIM ポンプ関数最適化 — Phase 2: 端点 u_max / しきい超えタイミング掃引 (2026-06-08)
計画: docs/20260608/1612_pump_function_optimization_plan.md
前段: modules/2026-06-08_CIM_pumpsched.py (Phase 0/1)

Phase 1 の知見:
  端点を揃えると形状効果は2次(±5程度)だが、ペア200seedで有意。勝者は「利得 g0 を線形に
  上げる」ランプ(g0軸 linear ≡ P軸 power p=2)で +5.3(z≈3)。臨界減速・シグモイドは悪化。
  → 未検証の主レバーは端点/しきい超えタイミング(H2)。本フェーズで叩く。

Phase 2(基準形状 = linear-gain, 評価 = ペア200seed vs 現行 P-linear):
  - 2A: 終端利得 u_max を掃引(g0軸 linear, u0 固定)。終端ポンプ準位の効果。
  - 2B: explore 割合 f を掃引(two-phase: しきい u=1 に τ=f で到達 → u_max へ凍結)。
        終端準位を固定して「しきい超えタイミング」だけを動かし、2A と交絡分離。

カーネル/形状生成は Phase 1 モジュールと同一(自己完結コピー; digit 始まりは import 不可かつ
numba cache×importlib が壊れるため、各モジュール自己完結が本リポジトリの方針)。
twophase 形状のみ追加。
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


# ===== Phase 1 と同一カーネル(電力スケジュール配列を受ける; 自己完結コピー) =====
@njit(cache=True, fastmath=True, parallel=True)
def _simulate_cim_sched_batch(
    n, num_rounds, num_trials, J_data, J_indices, J_indptr,
    edge_a, edge_b, edge_w, kappa, L, gamma, eta, bandwidth, photon_energy, P_sched, seeds,
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
            P_p = P_sched[k]
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


def simulate_cim_sched_batch(n, J, edges, P_sched, num_trials, kappa, L, gamma, eta,
                             bandwidth, photon_energy, seeds, weights=None):
    edges_np = np.asarray(edges, dtype=np.int64)
    edge_a = np.ascontiguousarray(edges_np[:, 0])
    edge_b = np.ascontiguousarray(edges_np[:, 1])
    edge_w = (np.ones(edges_np.shape[0], dtype=np.float64) if weights is None
              else np.ascontiguousarray(np.asarray(weights, dtype=np.float64)))
    seeds_arr = np.ascontiguousarray(np.asarray(seeds, dtype=np.int64))
    P_sched = np.ascontiguousarray(np.asarray(P_sched, dtype=np.float64))
    return _simulate_cim_sched_batch(
        n, int(P_sched.shape[0]), int(num_trials), J.data, J.indices, J.indptr,
        edge_a, edge_b, edge_w, float(kappa), float(L), float(gamma), float(eta),
        float(bandwidth), float(photon_energy), P_sched, seeds_arr)


def shape_s(family, tau, p=1.0, f=0.5, s_star=0.5):
    if family == "linear":
        return tau.copy()
    if family == "power":
        return tau ** p
    if family == "twophase":  # しきい(s_star)に τ=f で到達 → 残りで終端へ(線形2区間)
        sk = s_star
        s = np.where(tau < f, sk * (tau / f), sk + (1.0 - sk) * ((tau - f) / (1.0 - f)))
        return s
    raise ValueError(family)


def make_P_sched(K, axis, family, lo, hi, P_th, kappa, L, g0_th, **kw):
    tau = (np.arange(K) + 1) / K
    s_star = (1.0 - lo) / (hi - lo)
    s = shape_s(family, tau, s_star=s_star, **kw)
    val = lo + (hi - lo) * s
    if axis == "P":
        return val * P_th
    g0 = val * g0_th
    return (g0 / (2.0 * kappa * L)) ** 2


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
    from modules.verify import compute_cut_from_edges

    KNOWN_BEST = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}
    EXPERIMENT_KIND = "cim_pumpsched"

    ap = argparse.ArgumentParser(description="Phase 2: u_max / explore割合 掃引 (ペア200seed)")
    ap.add_argument("--graph", default="input/G22.txt")
    ap.add_argument("--rounds", type=int, default=1500)
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--coupling", type=float, default=-0.03)
    ap.add_argument("--u0", type=float, default=0.04, help="始端 g0/g0_th")
    ap.add_argument("--umax-2b", type=float, default=1.4, help="2B の固定終端 g0/g0_th")
    args = ap.parse_args()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 2 * plt.rcParams["font.size"]

    def ticks_in(ax):
        ax.tick_params(direction="in", which="both", top=True, right=True)

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
    u0 = args.u0
    print(f"Graph {graph_name} N={n} K={k_edges}  rounds={K} trials={NT}  "
          f"P_th={P_th*1e3:.2f}mW g0_th={g0_th:.3f}  u0={u0}")

    # baseline = 現行 P-linear (P_p=(k+1)dP)。Phase0 サニティ込み。
    base_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=K, num_trials=NT, seeds=seeds, dP_per_round=dP, **phys)
    P_lin_sched = (np.arange(K) + 1) * dP
    chk_cuts, _ = simulate_cim_sched_batch(n, J, edges, P_lin_sched, NT, seeds=seeds, **phys)
    assert np.array_equal(base_cuts, chk_cuts), "Phase0 再現サニティ失敗"
    print(f"[baseline] P-linear mean={base_cuts.mean():.1f} best={base_cuts.max():.0f}  (再現OK)")

    def paired_eval(P_sched, label):
        cuts, signs = simulate_cim_sched_batch(n, J, edges, P_sched, NT, seeds=seeds, **phys)
        bi = int(np.argmax(cuts))
        recut = compute_cut_from_edges(signs[bi].astype(np.int64).tolist(), edges)
        assert abs(recut - cuts[bi]) < 1e-6, f"{label}: カット不一致"
        d = cuts - base_cuts
        dm = float(d.mean())
        dse = float(d.std(ddof=1) / np.sqrt(NT))
        z = dm / dse if dse > 0 else 0.0
        # しきい超え round (g0/g0_th が 1 を最初に超える round)
        u_traj = 2.0 * kappa * L * np.sqrt(P_sched) / g0_th
        cross = int(np.argmax(u_traj >= 1.0)) + 1 if np.any(u_traj >= 1.0) else K
        return dict(label=label, mean=float(cuts.mean()), best=float(cuts.max()),
                    std=float(cuts.std()), dmean=dm, dse=dse, z=z, sig=bool(abs(z) >= 2),
                    cross=cross, umax=float(u_traj.max()))

    # ===== 2A: u_max 掃引 (g0軸 linear) =====
    U_MAX = [1.1, 1.2, 1.3, 1.4, 1.406, 1.5, 1.6, 1.8, 2.0]
    print(f"\n[2A] u_max 掃引 (g0軸 linear, u0={u0})")
    print(f"  {'u_max':>6} {'cross':>6} {'mean':>9} {'Δmean':>9} {'±2SE':>6} {'z':>6} {'有意':>4}")
    rows_2a = []
    for um in U_MAX:
        P = make_P_sched(K, "g0", "linear", u0, um, P_th, kappa, L, g0_th)
        r = paired_eval(P, f"u_max={um}")
        r["u_max"] = um
        rows_2a.append(r)
        print(f"  {um:>6.3f} {r['cross']:>6d} {r['mean']:>9.1f} {r['dmean']:>+9.2f} "
              f"{2*r['dse']:>6.2f} {r['z']:>+6.2f} {'★' if r['sig'] else '−':>4}")

    # ===== 2B: explore 割合 f 掃引 (two-phase, 終端 u_max 固定) =====
    FRACS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    um2b = args.umax_2b
    print(f"\n[2B] explore割合 f 掃引 (two-phase, 終端 u_max={um2b} 固定, u0={u0})")
    print(f"  {'f':>6} {'cross':>6} {'mean':>9} {'Δmean':>9} {'±2SE':>6} {'z':>6} {'有意':>4}")
    rows_2b = []
    for f in FRACS:
        P = make_P_sched(K, "g0", "twophase", u0, um2b, P_th, kappa, L, g0_th, f=f)
        r = paired_eval(P, f"f={f}")
        r["f"] = f
        rows_2b.append(r)
        print(f"  {f:>6.2f} {r['cross']:>6d} {r['mean']:>9.1f} {r['dmean']:>+9.2f} "
              f"{2*r['dse']:>6.2f} {r['z']:>+6.2f} {'★' if r['sig'] else '−':>4}")

    # ===== 出力 =====
    kind_root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name.split("_", 1)[0][1:].isdigit():
            max_v = max(max_v, int(p.name.split("_", 1)[0][1:]))
    out_dir = kind_root / f"v{max_v + 1}_rounds{K}_phase2_endpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    # Fig: 2パネル (2A: Δmean vs u_max, 2B: Δmean vs f)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.2))
    xa = [r["u_max"] for r in rows_2a]
    ya = [r["dmean"] for r in rows_2a]
    ea = [2 * r["dse"] for r in rows_2a]
    axA.errorbar(xa, ya, yerr=ea, fmt="o-", color="#2ca02c", ecolor="gray", capsize=3)
    axA.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.4, label="baseline (P-linear)")
    axA.set_xlabel("終端利得 u_max = g0_max/g0_th", fontsize=LABEL_FS)
    axA.set_ylabel("ペア Δmean (±2SE)", fontsize=LABEL_FS)
    axA.set_title("2A: 終端ポンプ準位 (g0軸 linear)")
    axA.legend(loc="lower left", fontsize=9)
    axA.grid(alpha=0.3)
    ticks_in(axA)
    xb = [r["f"] for r in rows_2b]
    yb = [r["dmean"] for r in rows_2b]
    eb = [2 * r["dse"] for r in rows_2b]
    axB.errorbar(xb, yb, yerr=eb, fmt="s-", color="#1f77b4", ecolor="gray", capsize=3)
    axB.axhline(0.0, color="#d62728", linestyle="--", linewidth=1.4, label="baseline (P-linear)")
    axB.set_xlabel(f"explore割合 f (しきい到達τ, 終端u_max={um2b})", fontsize=LABEL_FS)
    axB.set_ylabel("ペア Δmean (±2SE)", fontsize=LABEL_FS)
    axB.set_title("2B: しきい超えタイミング (two-phase)")
    axB.legend(loc="lower left", fontsize=9)
    axB.grid(alpha=0.3)
    ticks_in(axB)
    fig.suptitle(f"Phase 2: 端点/タイミング掃引 ({graph_name}, {NT} seed, ペア比較)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "phase2_sweeps.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'phase2_sweeps.png'}")

    best_2a = max(rows_2a, key=lambda r: r["dmean"])
    best_2b = max(rows_2b, key=lambda r: r["dmean"])
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges, "rounds": K, "trials": NT,
        "P_th_mW": P_th * 1e3, "g0_th": g0_th, "u0": u0, "umax_2b": um2b,
        "baseline_P_linear": {"mean": float(base_cuts.mean()), "best": float(base_cuts.max()),
                              "std": float(base_cuts.std())},
        "known_best": known,
        "sweep_2A_umax": [{k: r[k] for k in ("u_max", "cross", "mean", "best", "std",
                                             "dmean", "dse", "z", "sig")} for r in rows_2a],
        "sweep_2B_fraction": [{k: r[k] for k in ("f", "cross", "mean", "best", "std",
                                                 "dmean", "dse", "z", "sig")} for r in rows_2b],
        "best_2A": {k: best_2a[k] for k in ("u_max", "dmean", "dse", "z", "sig", "best")},
        "best_2B": {k: best_2b[k] for k in ("f", "dmean", "dse", "z", "sig", "best")},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print(f"  saved: {out_dir / 'summary.json'}")

    print(f"\n[結論メモ]")
    print(f"  2A 最良: u_max={best_2a['u_max']} Δmean={best_2a['dmean']:+.2f}±{2*best_2a['dse']:.2f} "
          f"z={best_2a['z']:+.2f} {'(有意)' if best_2a['sig'] else '(非有意)'}")
    print(f"  2B 最良: f={best_2b['f']} Δmean={best_2b['dmean']:+.2f}±{2*best_2b['dse']:.2f} "
          f"z={best_2b['z']:+.2f} {'(有意)' if best_2b['sig'] else '(非有意)'}")


if __name__ == "__main__":
    main()
