"""PumpBench 相当の条件比較を【本物のモデル】で G22 再検証する。

おもちゃ正規形モデル(ARC生成コード)ではなく、本リポジトリの検証済み実装を使う:
  - 開ループ各ポンプ形状 → modules/2026-06-08_CIM_pumpsched.py の
    `simulate_cim_sched_batch`(Inoue–Yoshida ファイバーループ進行波 CIM, 物理単位)
  - 閉ループ(CAC)        → modules/CAC.py の `simulate_cac_batch`(Leleu 2021)

各モデルは固有の正しい運用点で動かす(CIM 結合 -0.03 / CAC 結合 -1.0)。
G22・100 trial・seed 揃え(paired)。開ループ条件は同一 rounds で計算量整合、
CAC は標準予算(50000 outer step)で実行し実時間も併記(算法が違うため等round化は不可)。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/runners/run_pumpbench_real_cim_g22.py
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters, simulate_cac_batch  # noqa: E402

# ハイフン入りファイル名の pumpsched モジュールを importlib で読み込む
_spec = importlib.util.spec_from_file_location(
    "cim_pumpsched", ROOT / "modules" / "2026-06-08_CIM_pumpsched.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

EXPERIMENT_KIND = "pumpbench_real_cim_g22"
KNOWN_BEST = 13359
ROUNDS = 1500
TRIALS = 100
CAC_OUTER_STEPS = 50000


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                v = max(v, int(head[1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def summarize(name, label, color, cuts, t_sec, budget):
    cuts = np.asarray(cuts, dtype=float)
    return {
        "name": name, "label": label, "color": color,
        "mean": float(cuts.mean()), "best": float(cuts.max()),
        "worst": float(cuts.min()), "std": float(cuts.std()),
        "p0_hits": int((cuts == KNOWN_BEST).sum()),
        "pct_of_best": float(cuts.max() / KNOWN_BEST),
        "gap_vs_known": float((KNOWN_BEST - cuts.max()) / KNOWN_BEST),
        "seconds": float(t_sec), "budget": budget,
        "_cuts": cuts,
    }


def main() -> None:
    t0 = time.time()
    # ---- 物理パラメータ(modules/CIM.py main と同一) ----
    kappa, L, gamma = 130.0, 0.05, 42.09
    eta = 10.0 ** (-1.1)
    bandwidth, photon_energy = 1.0e9, 1.28e-19
    dP = 0.05e-3
    K = ROUNDS
    P_th = (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2
    g0_th = np.log(1.0 / eta)
    phys = dict(kappa=kappa, L=L, gamma=gamma, eta=eta,
                bandwidth=bandwidth, photon_energy=photon_energy)

    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J_cim = build_coupling_matrix(n, edges, -0.03)   # CIM の正しい結合
    seeds = np.arange(TRIALS, dtype=np.int64)
    print(f"G22 N={n} E={k_edges}  rounds={K} trials={TRIALS}  P_th={P_th*1e3:.2f}mW")

    # 端点(線形ベースラインに一致)
    rP_lo, rP_hi = 1.0 * dP / P_th, K * dP / P_th
    u_lo, u_hi = np.sqrt(rP_lo), np.sqrt(rP_hi)

    # ---- 開ループ条件: ポンプ波形(本物 CIM カーネル) ----
    sched_conditions = [
        ("linear_power", "線形電力ランプ(現行)", "#7f8c8d",
         (np.arange(K) + 1) * dP),
        ("power_early_p05", "べき乗 早上げ p=0.5", "#d35400",
         ps.make_P_sched(K, "P", "power", rP_lo, rP_hi, P_th, kappa, L, g0_th, p=0.5)),
        ("sigmoid", "シグモイド", "#2c5f8a",
         ps.make_P_sched(K, "P", "sigmoid", rP_lo, rP_hi, P_th, kappa, L, g0_th)),
        ("gain_linear", "線形利得ランプ(最良開ループ)", "#16a085",
         ps.make_P_sched(K, "g0", "linear", u_lo, u_hi, P_th, kappa, L, g0_th)),
    ]

    records = []
    for name, label, color, P_sched in sched_conditions:
        t = time.time()
        cuts, _ = ps.simulate_cim_sched_batch(n, J_cim, edges, P_sched, TRIALS,
                                              seeds=seeds, **phys)
        dt = time.time() - t
        rec = summarize(name, label, color, cuts, dt, f"{K} rounds")
        records.append(rec)
        print(f"  {label:<26} mean={rec['mean']:.1f} best={rec['best']:.0f} "
              f"std={rec['std']:.1f} ({rec['pct_of_best']*100:.2f}%) [{dt:.1f}s]")

    # ---- 閉ループ条件: 本物 CAC ----
    J_cac = build_coupling_matrix(n, edges, -1.0)    # CAC の正しい結合
    cac_params = compute_gset_parameters(J_cac, n)
    print(f"  [CAC] p={cac_params['p']:.4f} beta0={cac_params['beta0_error']:.4f} "
          f"tau={cac_params['tau']:.0f} (outer_steps={CAC_OUTER_STEPS})")
    t = time.time()
    cac_cuts, _ = simulate_cac_batch(
        n=n, J=J_cac, edges=edges, num_outer_steps=CAC_OUTER_STEPS, num_trials=TRIALS,
        seeds=seeds, **cac_params)
    dt = time.time() - t
    rec = summarize("cac", "CAC(本物・閉ループ)", "#c0392b", cac_cuts, dt,
                    f"{CAC_OUTER_STEPS} outer×(nx6+ne4)")
    records.append(rec)
    print(f"  {rec['label']:<26} mean={rec['mean']:.1f} best={rec['best']:.0f} "
          f"std={rec['std']:.1f} ({rec['pct_of_best']*100:.2f}%) [{dt:.1f}s]")

    elapsed = time.time() - t0

    # ---- 出力 ----
    out_dir = get_out_dir(f"5cond_{TRIALS}trial_real")
    # 生カット
    np.savez(out_dir / "cuts.npz", **{r["name"]: r["_cuts"] for r in records})
    # JSON
    js = {
        "graph": "G22", "known_best": KNOWN_BEST, "rounds": K, "trials": TRIALS,
        "models": {"open_loop": "modules/CIM.py fiber-loop (coupling -0.03)",
                   "closed_loop": "modules/CAC.py (coupling -1.0)"},
        "records": [{k: v for k, v in r.items() if k != "_cuts"} for r in records],
        "elapsed_seconds": elapsed,
    }
    (out_dir / "results.json").write_text(json.dumps(js, indent=2, default=float),
                                          encoding="utf-8")
    # CSV(ランキング)
    ranked = sorted(records, key=lambda r: -r["mean"])
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "condition", "label", "mean", "best", "worst", "std",
                    "pct_of_best", "p0_hits", "seconds", "budget"])
        for i, r in enumerate(ranked, 1):
            w.writerow([i, r["name"], r["label"], f"{r['mean']:.1f}", f"{r['best']:.0f}",
                        f"{r['worst']:.0f}", f"{r['std']:.1f}", f"{r['pct_of_best']:.4f}",
                        r["p0_hits"], f"{r['seconds']:.1f}", r["budget"]])

    make_figures(records, out_dir)

    print("\n" + "=" * 70)
    print(f"本物モデルでの G22 再検証 完了({elapsed:.1f}s)  既知ベスト={KNOWN_BEST}")
    print(f"{'条件':<28}{'平均':>9}{'最良':>8}{'std':>7}{'%既知':>8}")
    for r in sorted(records, key=lambda r: -r["mean"]):
        print(f"{r['label']:<28}{r['mean']:>9.1f}{r['best']:>8.0f}{r['std']:>7.1f}"
              f"{r['pct_of_best']*100:>7.2f}%")
    print("=" * 70)
    print(f"保存先: {out_dir}")


def make_figures(records, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    LABEL_FS = 13

    order = sorted(records, key=lambda r: r["mean"])  # 平均昇順(左=低い)
    all_cuts = np.concatenate([r["_cuts"] for r in records])
    x_min = float(all_cuts.min()) - 20
    x_max = max(float(all_cuts.max()) + 20, KNOWN_BEST + 10)
    bins = np.linspace(x_min, x_max, 36)

    # --- Fig1: ヒストグラム(参照スタイル) ---
    nc = len(order)
    fig, axes = plt.subplots(1, nc, figsize=(4.6 * nc, 4.6), sharex=True)
    for ax, r in zip(axes, order):
        c = r["_cuts"]
        ax.hist(c, bins=bins, color=r["color"], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axvline(c.mean(), color="black", linestyle=":", linewidth=1.3,
                   label=f"平均 {c.mean():.0f}")
        ax.axvline(KNOWN_BEST, color="red", linestyle="--", linewidth=1.3,
                   label=f"既知ベスト {KNOWN_BEST}")
        ax.set_title(f"{r['label']}\n平均:{c.mean():.0f} 最良:{c.max():.0f} "
                     f"({r['pct_of_best']*100:.2f}%)", fontsize=10)
        ax.set_xlabel("カット値", fontsize=LABEL_FS)
        ax.set_ylabel("頻度", fontsize=LABEL_FS)
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"本物モデルでのカット値分布 — G22 ({TRIALS} trial)  "
                 f"開ループ=ファイバーCIM / 閉ループ=CAC", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig)

    # --- Fig2: 棒グラフ(平均±std + 既知ベスト線) ---
    ranked = sorted(records, key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(9, 5.2))
    labels = [r["label"] for r in ranked]
    means = [r["mean"] for r in ranked]
    errs = [r["std"] for r in ranked]
    colors = [r["color"] for r in ranked]
    ax.bar(range(len(labels)), means, yerr=errs, color=colors, capsize=3)
    ax.axhline(KNOWN_BEST, color="red", ls="--", lw=1.5, label=f"既知ベスト {KNOWN_BEST}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("カット値(平均 ± 標準偏差)", fontsize=LABEL_FS)
    ax.set_ylim(min(means) - 80, KNOWN_BEST + 25)
    ax.set_title("本物モデルでの到達カット値 — G22", fontsize=13)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir/'hist.png'}, {out_dir/'comparison.png'}")


if __name__ == "__main__":
    main()
