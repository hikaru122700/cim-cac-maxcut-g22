"""
CIM + PT v2 : 温度ラダー反転版の検証 (2026-06-06)

----------------------------------------------------------------------
背景 (docs/CIM_PT_why_failed.md の続き):
  v1 (modules/2026-05-29_CIM_PT.py) は β ラダーを「高ポンプ = 低温(cold)」に割り当て、
  PT スワップで良い配置(高カット)を高ポンプ側へ流していた。これは標準 PT の
  「低温レプリカほど良い解」という前提に従ったもの。
  ところが CIM では前提が逆転しており、低ポンプ(ノイズ支配)レプリカが実測で最良カット、
  高ポンプ(飽和)レプリカが最悪カットになる。結果として v1 は「良い配置を悪い高ポンプ側へ
  流して汚す」動きになり、性能が下がった。

v2 の変更点 (これ "だけ"):
  β ラダーを反転し、最良カットを出す **低ポンプ・レプリカに β 最大(cold)** を割り当てる。
  → PT スワップが良い配置を「低ポンプ(実測最良)」側へ集めるようになる
    (= 良い解は低ポンプへ、悪い解は高ポンプへ遷移)。
  物理パラメータ・ポンプ準位 pump_levels・スワップ間隔・seed はすべて v1 と同一。
  スワップ受理カーネル (_simulate_cim_pt_batch) も v1 をそのまま再利用する
  (Metropolis 受理式は β の並び順に依らず正しいので、降順 β を渡すだけで反転になる)。

比較 (等計算量):
  ランプ CIM (baseline) / CIM+PT v1向き(良い解→高ポンプ) / CIM+PT v2反転(良い解→低ポンプ)
  を同一 seed・同一ポンプで並べ、「ラダーを逆にすると効くのか」をデータで判定する。
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))


def _load_v1():
    """digit 始まりでインポート不可な v1 モジュールを importlib でロードする。"""
    spec = importlib.util.spec_from_file_location(
        "cim_pt_v1", HERE / "2026-05-29_CIM_PT.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def calibrate_betas_reversed(calibrate_betas, cut_tail, kappa_target=1.0):
    """v1 の昇順ラダーを「低ポンプ=cold(β最大)」へ反転する。

    通常 (v1): betas_normal = [0, κ/g01, κ/g01+κ/g12]  (replica index で昇順、高ポンプ=cold)
    反転 (v2): betas_rev = betas_normal.max() - betas_normal
              → replica0(低ポンプ) が β 最大、replica(NR-1)(高ポンプ) が β=0。
    隣接ペアの間隔 |Δβ| は各ペアで κ_target に保たれたまま、cold 端だけが逆になる。
    """
    betas_normal = calibrate_betas(cut_tail, kappa_target=kappa_target)
    betas_rev = betas_normal.max() - betas_normal
    return betas_normal, betas_rev


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v1 = _load_v1()
    NR = v1.NR
    from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
    from modules.verify import compute_cut_from_edges

    EXPERIMENT_KIND = "cim_pt_v2"
    KNOWN_BEST = {"G15": 3050, "G22": 13359, "G55": 10299, "G70": 9591}

    parser = argparse.ArgumentParser(
        description="CIM+PT v2(βラダー反転=良い解を低ポンプへ) を v1/ランプCIM と等計算量比較")
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument("--cim-rounds", type=int, default=1500)
    parser.add_argument("--cim-coupling", type=float, default=-0.03)
    parser.add_argument("--pump-mults", type=float, nargs=3, default=[0.8, 1.0, 1.3],
                        help="P_th に対する 3 レプリカのポンプ倍率(低→高)")
    parser.add_argument("--swap-interval", type=int, default=10)
    parser.add_argument("--kappa-target", type=float, default=1.0)
    parser.add_argument("--sample-interval", type=int, default=25)
    parser.add_argument("--known-best", type=int, default=None)
    parser.add_argument("--tag", type=str, default="revladder")
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

    J = build_coupling_matrix(n, edges, args.cim_coupling, weights=w_arg)
    cim_params = dict(
        kappa=130.0, L=0.05, gamma=42.09, eta=10.0 ** (-1.1),
        bandwidth=1.0e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
    )
    pt_phys = {k: v for k, v in cim_params.items() if k != "dP_per_round"}

    p_th = v1.compute_threshold_pump(cim_params["kappa"], cim_params["L"], cim_params["eta"])
    pump_levels = np.sort(np.array([m * p_th for m in args.pump_mults]))
    print(f"P_th = {p_th * 1e3:.3f} mW  → pump_levels (mW) = "
          f"{[round(p * 1e3, 3) for p in pump_levels]}  (mults={sorted(args.pump_mults)})")

    # ==== baseline ランプ CIM (等計算量のため NR*NT trial) ====
    print(f"\n[ランプCIM] {NR * NT} trials  rounds={args.cim_rounds}")
    t0 = time.time()
    cim_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=NR * NT, seeds=cim_seeds, weights=w_arg, **cim_params)
    cim_time = time.time() - t0
    print(f"  time={cim_time:.2f}s  mean={cim_cuts.mean():.1f}  best={cim_cuts.max():.0f}")

    # ==== swap 無効の参照 run (β キャリブレーション & 領域同定) ====
    print(f"\n[CIM-3固定/swap無] {NT} trials  (βキャリブレーション用)")
    t0 = time.time()
    res_noswap = v1.simulate_cim_pt_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=NT, seeds=pt_seeds, weights=w_arg,
        pump_levels=pump_levels, do_swap=False,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    noswap_time = time.time() - t0
    noswap_cuts = res_noswap["best_cuts"]
    sample_rounds = res_noswap["sample_rounds"]

    tail = max(1, sample_rounds.size // 3)
    amp_tail = res_noswap["traj_amp"][:, -tail:, :].mean(axis=(0, 1))
    cut_tail = res_noswap["traj_cut"][:, -tail:, :].mean(axis=(0, 1))
    print(f"  定常カット (replica 0→2, 低→高ポンプ) = "
          f"[{cut_tail[0]:.1f}, {cut_tail[1]:.1f}, {cut_tail[2]:.1f}]")
    best_replica = int(np.argmax(cut_tail))
    print(f"  最良カットのレプリカ index = {best_replica}  "
          f"(0=低ポンプ/ノイズ, {NR-1}=高ポンプ/飽和)")

    betas_normal, betas_rev = calibrate_betas_reversed(
        v1.calibrate_betas, cut_tail, kappa_target=args.kappa_target)
    print(f"  β v1向き(高ポンプ=cold) = [{betas_normal[0]:.3e}, {betas_normal[1]:.3e}, {betas_normal[2]:.3e}]")
    print(f"  β v2反転(低ポンプ=cold) = [{betas_rev[0]:.3e}, {betas_rev[1]:.3e}, {betas_rev[2]:.3e}]")

    # ==== CIM+PT v1 向き (良い解→高ポンプ) ====
    print(f"\n[CIM+PT v1向き] {NT} trials  swap/{args.swap_interval}  (良い解→高ポンプ)")
    t0 = time.time()
    res_v1 = v1.simulate_cim_pt_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=NT, seeds=pt_seeds, weights=w_arg,
        pump_levels=pump_levels, betas=betas_normal, do_swap=True,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    v1_time = time.time() - t0
    v1_cuts = res_v1["best_cuts"]
    v1_rate = res_v1["swap_rate"]
    print(f"  time={v1_time:.2f}s  mean={v1_cuts.mean():.1f}  best={v1_cuts.max():.0f}  "
          f"受理率=[{v1_rate[0]:.3f},{v1_rate[1]:.3f}]")

    # ==== CIM+PT v2 反転 (良い解→低ポンプ) ====
    print(f"\n[CIM+PT v2反転] {NT} trials  swap/{args.swap_interval}  (良い解→低ポンプ)")
    t0 = time.time()
    res_v2 = v1.simulate_cim_pt_batch(
        n=n, J=J, edges=edges, num_rounds=args.cim_rounds,
        num_trials=NT, seeds=pt_seeds, weights=w_arg,
        pump_levels=pump_levels, betas=betas_rev, do_swap=True,
        swap_interval=args.swap_interval, sample_interval=args.sample_interval, **pt_phys)
    v2_time = time.time() - t0
    v2_cuts = res_v2["best_cuts"]
    v2_rate = res_v2["swap_rate"]
    print(f"  time={v2_time:.2f}s  mean={v2_cuts.mean():.1f}  best={v2_cuts.max():.0f}  "
          f"受理率=[{v2_rate[0]:.3f},{v2_rate[1]:.3f}]")

    # ==== 検証: 最良解の符号からカット独立再計算 ====
    def verify(res, cuts, name):
        bt = int(np.argmax(cuts))
        x = res["best_signs"][bt].astype(np.int64).tolist()
        if use_weights:
            rc = sum(weights[i] for i, (a, b) in enumerate(edges) if x[a] != x[b])
        else:
            rc = compute_cut_from_edges(x, edges)
        ok = abs(rc - cuts[bt]) < 1e-6
        print(f"[verify] {name}: kernel={cuts[bt]:.0f} 独立再計算={rc:.0f} 一致={ok}")
        return ok

    ok = verify(res_v1, v1_cuts, "v1向き") and verify(res_v2, v2_cuts, "v2反転")
    if not ok:
        raise SystemExit("検証失敗: カット値が独立計算と不一致")

    # ==== サマリ ====
    results = {"ランプCIM": cim_cuts, "CIM+PT v1向き(→高ポンプ)": v1_cuts,
               "CIM+PT v2反転(→低ポンプ)": v2_cuts}
    times = {"ランプCIM": cim_time, "CIM+PT v1向き(→高ポンプ)": v1_time,
             "CIM+PT v2反転(→低ポンプ)": v2_time}
    order = ["ランプCIM", "CIM+PT v1向き(→高ポンプ)", "CIM+PT v2反転(→低ポンプ)"]
    print("\n" + "=" * 92)
    print(f"{'Method':<26} {'Ntrial':>7} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Time[s]':>9}")
    print("-" * 92)
    for name in order:
        c = results[name]
        line = (f"{name:<26} {c.size:>7d} {c.mean():>10.1f} {c.max():>10.1f} {c.min():>10.1f} "
                f"{c.std():>8.1f} {times[name]:>9.2f}")
        if known_best is not None:
            line += f"  ratio={c.max() / known_best:.4f}"
        print(line)
    print("=" * 92)

    # ==== 出力ディレクトリ (results 規約 v{N}_) ====
    kind_root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    desc = [f"rounds{args.cim_rounds}", f"swap{args.swap_interval}"]
    if NT != 100:
        desc.append(f"trials{NT}")
    if args.tag:
        desc.append(args.tag)
    out_dir = kind_root / f"v{max_v + 1}_{'_'.join(desc)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    colors = {"ランプCIM": "#1f77b4", "CIM+PT v1向き(→高ポンプ)": "#9467bd",
              "CIM+PT v2反転(→低ポンプ)": "#2ca02c"}

    # --- Fig1: running best (等計算量, 横軸=レプリカ実行数換算) ---
    fig, ax = plt.subplots(figsize=(10, 5.4))
    x_cim = np.arange(1, cim_cuts.size + 1)
    x_pt = np.arange(1, NT + 1) * NR
    ax.plot(x_cim, np.maximum.accumulate(cim_cuts), color=colors["ランプCIM"],
            linewidth=2.0, label=f"ランプCIM ({cim_time:.1f}s)")
    ax.plot(x_pt, np.maximum.accumulate(v1_cuts), color=colors["CIM+PT v1向き(→高ポンプ)"],
            linewidth=2.0, label=f"CIM+PT v1向き→高ポンプ ({v1_time:.1f}s)")
    ax.plot(x_pt, np.maximum.accumulate(v2_cuts), color=colors["CIM+PT v2反転(→低ポンプ)"],
            linewidth=2.4, label=f"CIM+PT v2反転→低ポンプ ({v2_time:.1f}s)")
    if known_best is not None:
        ax.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                   label=f"既知ベスト {known_best}")
    ax.set_xlabel("計算量(レプリカ実行数 換算)", fontsize=LABEL_FS)
    ax.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax.set_title(f"等計算量での累積最良カット — βラダー反転の効果 ({graph_name})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "running_best.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'running_best.png'}")

    # --- Fig2: ヒストグラム ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    all_cuts = np.concatenate([cim_cuts, v1_cuts, v2_cuts])
    x_min = float(all_cuts.min()) - max(20, abs(all_cuts.min()) * 0.005)
    x_max = float(all_cuts.max()) + max(20, abs(all_cuts.max()) * 0.005)
    if known_best is not None:
        x_max = max(x_max, known_best + 10)
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
    fig.suptitle(f"ランプCIM vs PT v1向き vs PT v2反転 — {graph_name} (等計算量)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / "hist.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'hist.png'}")

    # --- Fig3: best-so-far 収束軌跡 (v1 vs v2) ---
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for res, name in [(res_v1, "CIM+PT v1向き(→高ポンプ)"), (res_v2, "CIM+PT v2反転(→低ポンプ)")]:
        tb = res["traj_best"]
        ax.fill_between(sample_rounds, np.percentile(tb, 10, axis=0),
                        np.percentile(tb, 90, axis=0), color=colors[name], alpha=0.15)
        ax.plot(sample_rounds, tb.mean(axis=0), color=colors[name], linewidth=2.2,
                label=f"{name} 平均 (最終 {tb.mean(axis=0)[-1]:.0f})")
    ax.axhline(cim_cuts.mean(), color=colors["ランプCIM"], linestyle=":", linewidth=1.6,
               label=f"ランプCIM 平均 {cim_cuts.mean():.0f}")
    if known_best is not None:
        ax.axhline(known_best, color="red", linestyle="--", linewidth=1.2, label=f"既知ベスト {known_best}")
    ax.set_xlabel("ラウンド数", fontsize=LABEL_FS)
    ax.set_ylabel("これまでの最良カット", fontsize=LABEL_FS)
    ax.set_title(f"収束軌跡 — v1向き vs v2反転 ({graph_name}, {NT} trial)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'trajectory.png'}")

    # --- Fig4: β ラダー(v1向き vs v2反転)とポンプ準位の対応 ---
    fig, ax = plt.subplots(figsize=(8, 5.0))
    pl_mW = pump_levels * 1e3
    ax.plot(pl_mW, betas_normal, "o-", color=colors["CIM+PT v1向き(→高ポンプ)"], linewidth=2.0,
            markersize=9, label="v1向き: 高ポンプ=cold(β最大)")
    ax.plot(pl_mW, betas_rev, "s-", color=colors["CIM+PT v2反転(→低ポンプ)"], linewidth=2.0,
            markersize=9, label="v2反転: 低ポンプ=cold(β最大)")
    for x, ct in zip(pl_mW, cut_tail):
        ax.annotate(f"カット{ct:.0f}", (x, max(betas_normal[np.argmin(abs(pl_mW - x))],
                    betas_rev[np.argmin(abs(pl_mW - x))])),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    ax.set_xlabel("ポンプ電力 P [mW]", fontsize=LABEL_FS)
    ax.set_ylabel("実効逆温度 β (大きいほど cold)", fontsize=LABEL_FS)
    ax.set_title("βラダーの向き — どのポンプを cold(良い解の集約先)にするか")
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(alpha=0.3)
    ticks_in(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "beta_ladders.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'beta_ladders.png'}")

    # ==== summary.json + data.npz ====
    summary = {
        "graph": graph_name, "n": n, "k_edges": k_edges,
        "num_trials_pt": NT, "num_trials_cim": int(NR * NT),
        "equal_compute": True, "n_replicas": NR,
        "cim_rounds": args.cim_rounds,
        "pump_mults": sorted(args.pump_mults), "p_th_mW": p_th * 1e3,
        "pump_levels_mW": [p * 1e3 for p in pump_levels.tolist()],
        "swap_interval": args.swap_interval, "kappa_target": args.kappa_target,
        "betas_v1_normal": betas_normal.tolist(),
        "betas_v2_reversed": betas_rev.tolist(),
        "tail_cut": cut_tail.tolist(), "tail_mean_abs_c": amp_tail.tolist(),
        "best_replica_index": best_replica,
        "swap_rate_v1": v1_rate.tolist(), "swap_rate_v2": v2_rate.tolist(),
        "known_best": known_best,
        "ramp_CIM": {"n_trial": int(cim_cuts.size), "mean": float(cim_cuts.mean()),
                     "best": float(cim_cuts.max()), "worst": float(cim_cuts.min()),
                     "std": float(cim_cuts.std()), "time_s": cim_time},
        "CIM_PT_v1_to_highpump": {"n_trial": int(v1_cuts.size), "mean": float(v1_cuts.mean()),
                                  "best": float(v1_cuts.max()), "worst": float(v1_cuts.min()),
                                  "std": float(v1_cuts.std()), "time_s": v1_time},
        "CIM_PT_v2_to_lowpump": {"n_trial": int(v2_cuts.size), "mean": float(v2_cuts.mean()),
                                 "best": float(v2_cuts.max()), "worst": float(v2_cuts.min()),
                                 "std": float(v2_cuts.std()), "time_s": v2_time},
        "verify_ok": bool(ok),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez(out_dir / "data.npz", cim=cim_cuts, v1=v1_cuts, v2=v2_cuts,
             v1_traj_best=res_v1["traj_best"], v2_traj_best=res_v2["traj_best"],
             noswap_traj_amp=res_noswap["traj_amp"], noswap_traj_cut=res_noswap["traj_cut"],
             sample_rounds=sample_rounds, pump_levels=pump_levels,
             betas_normal=betas_normal, betas_rev=betas_rev)
    print(f"  saved: {out_dir / 'summary.json'}")
    print("\n[結論メモ]")
    d_v1 = v1_cuts.mean() - cim_cuts.mean()
    d_v2 = v2_cuts.mean() - cim_cuts.mean()
    print(f"  平均カット差 (対 ランプCIM): v1向き={d_v1:+.1f}, v2反転={d_v2:+.1f}")
    print(f"  v2反転 − v1向き(平均) = {v2_cuts.mean() - v1_cuts.mean():+.1f}")


if __name__ == "__main__":
    main()
