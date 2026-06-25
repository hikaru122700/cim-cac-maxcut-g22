"""plot_anytime_before_after.py — anytime 曲線の「延長前 vs 延長後」重ね描き。

2 つの results.json(例: 壁45s の v7 と 壁120s の ext120)を読み、各データセット・
各アルゴリズムについて gap%(=最良解, BKS への到達差) の時間推移を重ねる。
  破線 + 白抜きマーカー = 延長前(before)
  実線 + 塗りマーカー   = 延長後(after)
縦軸は 0 始まり(gap<0 はあり得ない)、x=実時間(log)。

実行(プロジェクトルートから):
  python scripts/plotting/plot_anytime_before_after.py <before_dir> <after_dir> [out.png]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, ALGOS

ORDER = ["G22", "G55", "G70", "K2000"]


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load(d):
    return json.loads((Path(d) / "results.json").read_text(encoding="utf-8"))["results"]


def gap_pct(bks, pts):
    return [100.0 * (bks - p["cut_max"]) / bks for p in pts]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    before = load(sys.argv[1])
    after = load(sys.argv[2])
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(sys.argv[2]) / "compare_before_after.png"

    datasets = [d for d in ORDER if d in before and d in after]
    plt = setup_style()

    nrow = (len(datasets) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 5.2 * nrow), dpi=130)
    axes = np.atleast_1d(axes).ravel()

    from matplotlib.lines import Line2D

    for i, ds in enumerate(datasets):
        ax = axes[i]
        bks = DATASETS[ds]["bks"]
        for algo_key in ALGOS:
            c = ALGOS[algo_key]["color"]
            b = before.get(ds, {}).get(algo_key) or []
            a = after.get(ds, {}).get(algo_key) or []
            if b:
                ax.plot([p["time"] for p in b], gap_pct(bks, b), "--o", color=c,
                        lw=1.2, ms=4, alpha=0.45, mfc="none")
            if a:
                ax.plot([p["time"] for p in a], gap_pct(bks, a), "-o", color=c,
                        lw=2.0, ms=4, label=ALGOS[algo_key]["label"])
        ax.axhline(0.0, color="k", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(bottom=0.0)
        ax.set_title(f"{ds} (BKS={bks})")
        ax.set_xlabel("実時間 [秒]")
        ax.set_ylabel("BKS gap [%]（最良解）")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(alpha=0.25, which="both")
        if i == 0:
            algo_leg = ax.legend(fontsize=7, ncol=2, loc="upper right",
                                 title="アルゴリズム(実線=延長後)")
            ax.add_artist(algo_leg)
            style_handles = [
                Line2D([0], [0], color="gray", ls="--", marker="o", mfc="none",
                       alpha=0.6, label="延長前（壁45s）"),
                Line2D([0], [0], color="gray", ls="-", marker="o",
                       label="延長後（壁120s）"),
            ]
            ax.legend(handles=style_handles, fontsize=7, loc="lower left")

    for j in range(len(datasets), len(axes)):
        axes[j].axis("off")
    fig.suptitle("anytime 比較: 予算延長前(破線) vs 延長後(実線) — BKS gap の時間推移", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
