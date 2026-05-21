"""compare_cim_pticm.py — CIM と PT-ICM の解品質を同一インスタンスで比較する。

PT-ICM は Zhu-Ochoa-Katzgraber 2015 の方式で、CIM 系論文(Hamerly+ 2019,
McMahon+ 2016, Leleu+ 2019 ほか)でヒューリスティクスの参照点として頻繁に
用いられる SOTA 級 SA 派生。

使い方:
    python scripts/benchmarks/compare_cim_pticm.py
    python scripts/benchmarks/compare_cim_pticm.py --graph input/G22.txt --num-trials 100
    python scripts/benchmarks/compare_cim_pticm.py --pticm-sweeps 400 --pticm-num-temps 16

出力 (新規約: `results/<今日>/<実験種別>/v{N}_<説明>/<ファイル>`):
    results/<今日>/cim_vs_pticm/v{N}_<desc>/
        hist.png  running_best.png  bar.png  pticm_trajectory.png
        cuts.npz  summary.json
    <desc> は CLI 引数から自動生成(例: sweeps1500_NT16_trajectory)。
    --tag で追加サフィックスを指定可。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch
from modules.PT_ICM import make_geometric_ladder, simulate_pticm_batch


EXPERIMENT_KIND = "cim_vs_pticm"


KNOWN_BEST: dict[str, int] = {
    "G15": 3050,
    "G22": 13359,
    "G55": 10299,
    "G70": 9591,
}


def setup_plot_style() -> None:
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False


def apply_ticks_inward(ax: plt.Axes) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)


def get_kind_root() -> Path:
    """results/<today>/<EXPERIMENT_KIND>/ を返す(必要なら作成)。"""
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    """kind_root 配下の v{N}_* サブディレクトリを見て次の N を返す。"""
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def build_description(args, sample_interval: int) -> str:
    """CLI 引数から run 内容を表す簡潔な説明文字列を生成する。"""
    parts = [f"sweeps{args.pticm_sweeps}", f"NT{args.pticm_num_temps}"]
    if args.num_trials != 100:
        parts.append(f"trials{args.num_trials}")
    if args.pticm_sample_interval > 0 and args.pticm_sample_interval != args.pticm_sweeps:
        parts.append("trajectory")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="CIM vs PT-ICM 比較ベンチ")
    parser.add_argument("--graph", default="input/G22.txt", help="入力グラフファイル")
    parser.add_argument("--num-trials", type=int, default=100, help="各手法の trial 数")
    parser.add_argument("--seed-base", type=int, default=0, help="seed の起点")
    # CIM
    parser.add_argument("--cim-rounds", type=int, default=1500, help="CIM の num_rounds")
    parser.add_argument("--cim-coupling", type=float, default=-0.03, help="CIM の J スケール")
    # PT-ICM
    parser.add_argument("--pticm-sweeps", type=int, default=200, help="PT-ICM 全体スイープ数")
    parser.add_argument("--pticm-sweep-len", type=int, default=0,
                        help="1 スイープでの単スピン試行数 (0 → N)")
    parser.add_argument("--pticm-num-temps", type=int, default=12, help="温度段数")
    parser.add_argument("--pticm-t-min", type=float, default=0.05, help="最低温度")
    parser.add_argument("--pticm-t-max", type=float, default=3.0, help="最高温度")
    parser.add_argument("--pticm-swap-interval", type=int, default=1, help="PT swap 間隔")
    parser.add_argument("--pticm-icm-interval", type=int, default=5, help="ICM 間隔")
    parser.add_argument("--pticm-sample-interval", type=int, default=25,
                        help="trajectory サンプル間隔 (sweeps)。0 で最終値のみ")
    parser.add_argument("--known-best", type=int, default=None, help="既知ベスト(未指定なら自動)")
    parser.add_argument("--tag", type=str, default="",
                        help="出力ディレクトリ名末尾に付ける任意の説明タグ")
    args = parser.parse_args()

    setup_plot_style()

    graph_path = Path(args.graph)
    graph_name = graph_path.stem

    n, k_edges, _adj, edges, weights = load_graph(str(graph_path), return_weights=True)
    use_weights = any(w != 1.0 for w in weights)
    print(f"Graph: {graph_path} N={n} K={k_edges} weighted={use_weights}")

    known_best = args.known_best if args.known_best is not None else KNOWN_BEST.get(graph_name)
    if known_best is not None:
        print(f"Known best: {known_best}")

    seeds = np.arange(args.seed_base, args.seed_base + args.num_trials, dtype=np.int64)

    # ==== CIM 実行 ====
    print(f"\n[CIM] {args.num_trials} trials  num_rounds={args.cim_rounds}  J={args.cim_coupling}")
    J_cim = build_coupling_matrix(
        n, edges, args.cim_coupling, weights=(weights if use_weights else None)
    )
    cim_params = dict(
        kappa=130.0,
        L=0.05,
        gamma=42.09,
        eta=10.0 ** (-1.1),
        bandwidth=1.0e9,
        photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
    )
    t0 = time.time()
    cim_cuts, _cim_signs = simulate_cim_batch(
        n=n,
        J=J_cim,
        edges=edges,
        num_rounds=args.cim_rounds,
        num_trials=args.num_trials,
        seeds=seeds,
        weights=(weights if use_weights else None),
        **cim_params,
    )
    cim_time = time.time() - t0
    print(
        f"  time={cim_time:.2f}s ({cim_time / args.num_trials * 1000:.1f} ms/trial)  "
        f"mean={cim_cuts.mean():.1f}  best={cim_cuts.max():.0f}"
    )

    # ==== PT-ICM 実行 ====
    T_ladder = make_geometric_ladder(args.pticm_t_min, args.pticm_t_max, args.pticm_num_temps)
    sweep_len = args.pticm_sweep_len if args.pticm_sweep_len > 0 else n
    print(
        f"\n[PT-ICM] {args.num_trials} trials  sweeps={args.pticm_sweeps}  sweep_len={sweep_len}\n"
        f"  NT={args.pticm_num_temps}  T:{args.pticm_t_min}→{args.pticm_t_max}  "
        f"swap/{args.pticm_swap_interval}  icm/{args.pticm_icm_interval}"
    )
    sample_interval = args.pticm_sample_interval if args.pticm_sample_interval > 0 else args.pticm_sweeps
    t0 = time.time()
    pticm_cuts, _pticm_signs, pticm_traj = simulate_pticm_batch(
        n=n,
        edges=edges,
        weights=(weights if use_weights else None),
        num_trials=args.num_trials,
        num_sweeps=args.pticm_sweeps,
        sweep_len=sweep_len,
        T_ladder=T_ladder,
        swap_interval=args.pticm_swap_interval,
        icm_interval=args.pticm_icm_interval,
        sample_interval=sample_interval,
        seeds=seeds,
    )
    pticm_time = time.time() - t0
    sample_sweeps = np.arange(1, pticm_traj.shape[1] + 1) * sample_interval
    print(
        f"  time={pticm_time:.2f}s ({pticm_time / args.num_trials * 1000:.1f} ms/trial)  "
        f"mean={pticm_cuts.mean():.1f}  best={pticm_cuts.max():.0f}"
    )

    # ==== 統計サマリ ====
    results = {"CIM": cim_cuts, "PT-ICM": pticm_cuts}
    times = {"CIM": cim_time, "PT-ICM": pticm_time}

    print("\n" + "=" * 82)
    header = f"{'Method':<8} {'Mean':>10} {'Best':>10} {'Worst':>10} {'Std':>8} {'Time[s]':>10}"
    if known_best is not None:
        header += f" {'Ratio':>8}"
    print(header)
    print("-" * 82)
    for name in ["CIM", "PT-ICM"]:
        cuts = results[name]
        line = (
            f"{name:<8} {cuts.mean():>10.1f} {cuts.max():>10.1f} {cuts.min():>10.1f} "
            f"{cuts.std():>8.1f} {times[name]:>10.2f}"
        )
        if known_best is not None:
            line += f" {cuts.max() / known_best:>8.4f}"
        print(line)
    print("=" * 82)

    kind_root = get_kind_root()
    v = next_version(kind_root)
    desc = build_description(args, sample_interval)
    out_dir = kind_root / f"v{v}_{desc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[output] dir={out_dir}")

    colors = {"CIM": "#1f77b4", "PT-ICM": "#d62728"}

    # --- Figure 1: ヒストグラム ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    all_cuts = np.concatenate([cim_cuts, pticm_cuts])
    x_min = float(all_cuts.min()) - max(20, abs(all_cuts.min()) * 0.005)
    x_max = float(all_cuts.max()) + max(20, abs(all_cuts.max()) * 0.005)
    if known_best is not None:
        x_max = max(x_max, known_best + 10)
    bins = np.linspace(x_min, x_max, 30)

    for ax, name in zip(axes, ["CIM", "PT-ICM"]):
        cuts = results[name]
        ax.hist(cuts, bins=bins, color=colors[name], alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(cuts.mean(), color="black", linestyle=":", linewidth=1.2,
                   label=f"平均 {cuts.mean():.0f}")
        if known_best is not None:
            ax.axvline(known_best, color="red", linestyle="--", linewidth=1.2,
                       label=f"既知ベスト {known_best}")
        ax.set_title(
            f"{name}  時間: {times[name]:.1f}s  平均: {cuts.mean():.0f}  最良: {cuts.max():.0f}",
            fontsize=11,
        )
        ax.set_xlabel("カット値")
        ax.set_ylabel("頻度")
        ax.set_xlim(x_min, x_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="upper left")
        apply_ticks_inward(ax)
    fig1.suptitle(
        f"CIM vs PT-ICM — {graph_name} (各 {args.num_trials} trial)",
        fontsize=13,
    )
    fig1.tight_layout()
    hist_path = out_dir / "hist.png"
    fig1.savefig(hist_path, dpi=150)
    plt.close(fig1)
    print(f"  saved: {hist_path}")

    # --- Figure 2: running best ---
    fig2, ax2 = plt.subplots(figsize=(10, 5.4))
    for name in ["CIM", "PT-ICM"]:
        cuts = results[name]
        running = np.maximum.accumulate(cuts)
        ax2.plot(
            np.arange(1, args.num_trials + 1),
            running,
            label=f"{name}  (壁時計 {times[name]:.1f}s)",
            color=colors[name],
            linewidth=2.0,
        )
    if known_best is not None:
        ax2.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax2.set_xlabel("trial 数")
    ax2.set_ylabel("これまでの最良カット")
    ax2.set_title(f"trial 数に対する累積最良カット ({graph_name})")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    apply_ticks_inward(ax2)
    fig2.tight_layout()
    running_path = out_dir / "running_best.png"
    fig2.savefig(running_path, dpi=150)
    plt.close(fig2)
    print(f"  saved: {running_path}")

    # --- Figure 3: bar (mean & best) ---
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    methods = ["CIM", "PT-ICM"]
    means = [float(results[m].mean()) for m in methods]
    bests = [float(results[m].max()) for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax3.bar(x - width / 2, means, width, label="平均",
            color=[colors[m] for m in methods], alpha=0.55, edgecolor="black", linewidth=0.5)
    ax3.bar(x + width / 2, bests, width, label="最良",
            color=[colors[m] for m in methods], alpha=1.0, edgecolor="black", linewidth=0.5)
    if known_best is not None:
        ax3.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods)
    ax3.set_ylabel("カット値")
    ax3.set_title(f"{args.num_trials} trial における平均と最良 ({graph_name})")
    y_min = min(means) - max(50, abs(min(means)) * 0.005)
    y_max = (known_best + 30) if known_best is not None else max(bests) * 1.005
    ax3.set_ylim(y_min, y_max)
    ax3.legend(loc="lower right")
    ax3.grid(axis="y", alpha=0.3)
    for i, _m in enumerate(methods):
        ax3.text(i - width / 2, means[i] + (y_max - y_min) * 0.005, f"{means[i]:.0f}",
                 ha="center", fontsize=9)
        ax3.text(i + width / 2, bests[i] + (y_max - y_min) * 0.005, f"{bests[i]:.0f}",
                 ha="center", fontsize=9)
    apply_ticks_inward(ax3)
    fig3.tight_layout()
    bar_path = out_dir / "bar.png"
    fig3.savefig(bar_path, dpi=150)
    plt.close(fig3)
    print(f"  saved: {bar_path}")

    # --- Figure 4: PT-ICM trajectory (sweep 数 vs cut) ---
    fig4, ax4 = plt.subplots(figsize=(10, 5.4))
    traj_mean = pticm_traj.mean(axis=0)
    traj_best = pticm_traj.max(axis=0)
    traj_p10 = np.percentile(pticm_traj, 10, axis=0)
    traj_p90 = np.percentile(pticm_traj, 90, axis=0)
    ax4.fill_between(sample_sweeps, traj_p10, traj_p90,
                     color=colors["PT-ICM"], alpha=0.18, label="PT-ICM 10–90%ile")
    ax4.plot(sample_sweeps, traj_mean, color=colors["PT-ICM"], linewidth=2.2,
             label=f"PT-ICM 平均 (最終 {traj_mean[-1]:.0f})")
    ax4.plot(sample_sweeps, traj_best, color=colors["PT-ICM"], linewidth=1.5,
             linestyle="--", label=f"PT-ICM 最良 (最終 {traj_best[-1]:.0f})")
    # CIM の最終値を水平線で参照
    ax4.axhline(cim_cuts.mean(), color=colors["CIM"], linestyle=":", linewidth=1.6,
                label=f"CIM 平均 {cim_cuts.mean():.0f}")
    ax4.axhline(cim_cuts.max(), color=colors["CIM"], linestyle="-.", linewidth=1.4,
                label=f"CIM 最良 {cim_cuts.max():.0f}")
    if known_best is not None:
        ax4.axhline(known_best, color="red", linestyle="--", linewidth=1.2,
                    label=f"既知ベスト {known_best}")
    ax4.set_xlabel("PT-ICM スイープ数")
    ax4.set_ylabel("これまでの最良カット")
    ax4.set_title(
        f"PT-ICM の収束軌跡 ({graph_name}, {args.num_trials} trial, sample/{sample_interval} sweeps)"
    )
    ax4.legend(loc="lower right", fontsize=9)
    ax4.grid(alpha=0.3)
    apply_ticks_inward(ax4)
    fig4.tight_layout()
    traj_path = out_dir / "pticm_trajectory.png"
    fig4.savefig(traj_path, dpi=150)
    plt.close(fig4)
    print(f"  saved: {traj_path}")

    # --- 生データ ---
    npz_path = out_dir / "cuts.npz"
    np.savez(
        npz_path,
        cim_cuts=cim_cuts,
        pticm_cuts=pticm_cuts,
        pticm_trajectory=pticm_traj,
        sample_sweeps=sample_sweeps,
        seeds=seeds,
        T_ladder=T_ladder,
    )
    print(f"  saved: {npz_path}")

    # --- summary JSON ---
    summary = {
        "graph": str(graph_path),
        "n": n,
        "k": k_edges,
        "weighted": use_weights,
        "num_trials": args.num_trials,
        "seed_base": args.seed_base,
        "known_best": known_best,
        "cim": {
            "num_rounds": args.cim_rounds,
            "coupling": args.cim_coupling,
            **{kk: float(vv) for kk, vv in cim_params.items()},
            "mean": float(cim_cuts.mean()),
            "best": float(cim_cuts.max()),
            "worst": float(cim_cuts.min()),
            "std": float(cim_cuts.std()),
            "time_sec": cim_time,
        },
        "pticm": {
            "num_sweeps": args.pticm_sweeps,
            "sweep_len": sweep_len,
            "num_temps": args.pticm_num_temps,
            "t_min": args.pticm_t_min,
            "t_max": args.pticm_t_max,
            "T_ladder": T_ladder.tolist(),
            "swap_interval": args.pticm_swap_interval,
            "icm_interval": args.pticm_icm_interval,
            "sample_interval": sample_interval,
            "trajectory_mean": traj_mean.tolist(),
            "trajectory_best": traj_best.tolist(),
            "sample_sweeps": sample_sweeps.tolist(),
            "mean": float(pticm_cuts.mean()),
            "best": float(pticm_cuts.max()),
            "worst": float(pticm_cuts.min()),
            "std": float(pticm_cuts.std()),
            "time_sec": pticm_time,
        },
    }
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved: {json_path}")


if __name__ == "__main__":
    main()
