"""plot_cim_phases.py — cim_phase_ablation の結果を作図する。

使い方(プロジェクトルートから):
    .venv/Scripts/python.exe -u scripts/plotting/plot_cim_phases.py \
        results/<date>/cim_phase_ablation/v{N}_...

出力(同じ run ディレクトリ):
    fig1_phases.png  … 3 段階の境界(ノルム / パタパタ率)と best_cut の頭打ち
    fig2_offset.png  … 第1段階を削ったときの品質と多様性
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 22,
    "legend.fontsize": 18, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "font.size": 14,
})

C_NORM, C_FLIP, C_BEST, C_DIV = "#2471a3", "#c0392b", "#27ae60", "#8e44ad"
PHASE_COLORS = ["#fdebd0", "#d5f5e3", "#eaecee"]
PHASE_NAMES = ["第1段階 パタパタ期", "第2段階 形成期", "第3段階 飽和期"]


def style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)


def phase_bounds(t: dict) -> tuple[int, int]:
    """(第1→2 の境界, 第2→3 の境界)。

    第1→2 は符号の入れ替わり率が初期値の半分に落ちる round、
    第2→3 は best_cut が最終値に達する round(以後の出力は不変)。
    """
    rr = np.array(t["rounds"])[1:]          # 先頭は定義上 0 なので除く
    fm = np.array(t["flip_rate"])[1:]
    f0 = fm[:5].mean()
    k1 = int(rr[int(np.argmax(fm <= 0.5 * f0))]) if (fm <= 0.5 * f0).any() else int(rr[0])
    return k1, int(t["k_final"])


def shade(ax, k1, k2, total):
    for (lo, hi), col in zip([(1, k1), (k1, k2), (k2, total)], PHASE_COLORS):
        ax.axvspan(lo, hi, color=col, zorder=0)


def fig_phases(traj: dict, dss: list[str], out_path: Path):
    fig, axes = plt.subplots(2, len(dss), figsize=(8.5 * len(dss), 11.0))
    axes = np.atleast_2d(axes)
    if len(dss) == 1:
        axes = axes.reshape(2, 1)

    for c, ds in enumerate(dss):
        t = traj[ds]
        rr = np.array(t["rounds"])
        nm, fm, bm = (np.array(t[k]) for k in ("norm", "flip_rate", "best_cut"))
        total, bks = t["total_rounds"], t["bks"]
        k1, k2 = phase_bounds(t)

        # --- 上段: ノルムとパタパタ率 ---
        ax = axes[0, c]
        shade(ax, k1, k2, total)
        ax.plot(rr, nm, color=C_NORM, linewidth=2.6, label="振幅ノルム ||c||")
        ax.set_yscale("log")
        ax.set_ylabel("振幅ノルム ||c||", color=C_NORM)
        ax.tick_params(axis="y", colors=C_NORM)
        ax.set_xlabel("round step")
        ax.set_xscale("log")
        ax.set_title(f"{ds}: 3 段階の境界")
        style(ax)

        ax2 = ax.twinx()
        ax2.plot(rr[1:], fm[1:], color=C_FLIP, linewidth=2.6,
                 label="符号の入れ替わり率")
        ax2.set_ylabel("符号の入れ替わり率", color=C_FLIP)
        ax2.tick_params(axis="y", colors=C_FLIP, direction="in")
        ax2.set_ylim(-0.02, 0.55)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if c == 0:
            ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=16)

        # --- 下段: best_cut(= その round で打ち切ったときの出力) ---
        ax = axes[1, c]
        shade(ax, k1, k2, total)
        gap = (bks - bm) / bks * 100.0
        ax.plot(rr, gap, color=C_BEST, linewidth=2.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("打ち切り round")
        ax.set_ylabel("その時点の最良解の gap [%]")
        ax.axvline(k2, color="#c0392b", linestyle="--", linewidth=2.2)
        ax.annotate(f"round {k2} 以降は\n出力が変わらない\n(残り {total - k2} round = "
                    f"{(total - k2) / total * 100:.0f}%)",
                    xy=(k2, gap[-1]), xytext=(12, 40), textcoords="offset points",
                    fontsize=15, color="#c0392b")
        ax.set_title(f"{ds}: 解が確定する round")
        style(ax)

    handles = [plt.Rectangle((0, 0), 1, 1, color=col) for col in PHASE_COLORS]
    axes[0, 0].legend(
        handles + list(axes[0, 0].get_legend_handles_labels()[0]),
        PHASE_NAMES + list(axes[0, 0].get_legend_handles_labels()[1]),
        loc="center left", fontsize=15, framealpha=0.95)

    fig.suptitle("CIM の 3 段階 — 解はどこで決まっているか", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def fig_offset(offset: dict, traj: dict, dss: list[str], out_path: Path):
    fig, axes = plt.subplots(1, len(dss), figsize=(8.5 * len(dss), 7.2))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, dss):
        res = offset[ds]
        items = sorted(res.values(), key=lambda v: v["pump_offset"])
        x = [v["offset_frac"] * 100 for v in items]
        gap = [v["gap_pct_mean"] for v in items]
        div = [v["mean_pairwise"] for v in items]

        ax.plot(x, gap, "-o", color=C_BEST, markersize=11, linewidth=2.6,
                label="平均 gap [%]")
        ax.set_xlabel("切り落とした第1段階の長さ [全 round の %]")
        ax.set_ylabel("平均 gap [%]", color=C_BEST)
        ax.tick_params(axis="y", colors=C_BEST)
        style(ax)

        ax2 = ax.twinx()
        ax2.plot(x, div, "-s", color=C_DIV, markersize=11, linewidth=2.6,
                 label="平均ペア距離")
        ax2.set_ylabel("平均ペア距離", color=C_DIV)
        ax2.tick_params(axis="y", colors=C_DIV, direction="in")
        ax2.set_ylim(0.30, 0.52)
        ax2.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1.6)

        k1, _ = phase_bounds(traj[ds])
        ax.axvline(k1 / traj[ds]["total_rounds"] * 100, color="#7f8c8d",
                   linestyle="--", linewidth=2.0)
        ax.annotate("第1段階の全長", xy=(k1 / traj[ds]["total_rounds"] * 100, gap[0]),
                    xytext=(8, 10), textcoords="offset points",
                    fontsize=15, color="#7f8c8d")
        ax.set_title(f"{ds}")
        if ax is axes[0]:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=16)

    fig.suptitle("第1段階を削ると品質が落ち、多様性だけ上がる", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    with open(args.run_dir / "results.json", encoding="utf-8") as f:
        p = json.load(f)
    dss = p["meta"]["datasets"]
    fig_phases(p["trajectory"], dss, args.run_dir / "fig1_phases.png")
    fig_offset(p["offset"], p["trajectory"], dss, args.run_dir / "fig2_offset.png")
    print(f"saved 2 figures -> {args.run_dir}")


if __name__ == "__main__":
    main()
