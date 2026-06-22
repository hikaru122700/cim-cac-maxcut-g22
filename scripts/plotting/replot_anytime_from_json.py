"""replot_anytime_from_json.py — 既存 results.json から anytime 図を作り直す。

再計算はせず、保存済みの (time, cut_max) のみを使って combined.png と
<DATASET>_gap.png を再描画する。縦軸(BKS gap %)は 0 始まり(gap<0 はあり得ない)。

実行:
  python scripts/plotting/replot_anytime_from_json.py <results_dir>
例:
  python scripts/plotting/replot_anytime_from_json.py \
    results/2026-06-19/anytime_single/v7_G22_K2000_G55_G70_nt16_alltuned
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, ALGOS


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def gap_pct(bks, pts):
    return [100.0 * (bks - p["cut_max"]) / bks for p in pts]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    out_dir = Path(sys.argv[1])
    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    datasets = data["meta"]["datasets"]
    results = data["results"]

    plt = setup_style()

    # --- 個別 <DATASET>_gap.png ---
    for ds in datasets:
        bks = DATASETS[ds]["bks"]
        fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=140)
        for algo_key, pts in results[ds].items():
            if not pts:
                continue
            t = [p["time"] for p in pts]
            ax.plot(t, gap_pct(bks, pts), "-o", color=ALGOS[algo_key]["color"],
                    lw=2.0, ms=5, label=ALGOS[algo_key]["label"])
        ax.axhline(0.0, color="k", ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=0.05)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("実時間 [秒]（バッチ, log）")
        ax.set_ylabel("BKS への gap [%]（最良解, 小さいほど良い）")
        ax.set_title(f"{ds}: BKS への到達 gap の時間推移")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_gap.png", bbox_inches="tight")
        plt.close(fig)

    # --- 統合 combined.png ---
    nds = len(datasets)
    if nds > 1:
        ncol = 2
        nrow = (nds + 1) // 2
        fig, axes = plt.subplots(nrow, ncol, figsize=(13, 5.2 * nrow), dpi=130)
        axes = np.atleast_1d(axes).ravel()
        for i, ds in enumerate(datasets):
            ax = axes[i]
            bks = DATASETS[ds]["bks"]
            for algo_key, pts in results[ds].items():
                if not pts:
                    continue
                t = [p["time"] for p in pts]
                ax.plot(t, gap_pct(bks, pts), "-o", color=ALGOS[algo_key]["color"],
                        lw=1.8, ms=4, label=ALGOS[algo_key]["label"])
            ax.axhline(0.0, color="k", ls=":", lw=1.0)
            ax.set_xscale("log")
            ax.set_yscale("symlog", linthresh=0.05)
            ax.set_ylim(bottom=0.0)
            ax.set_title(f"{ds} (BKS={bks})")
            ax.set_xlabel("実時間 [秒]")
            ax.set_ylabel("BKS gap [%]")
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.grid(alpha=0.25, which="both")
            if i == 0:
                ax.legend(fontsize=7, ncol=2)
        for j in range(nds, len(axes)):
            axes[j].axis("off")
        fig.suptitle("単体アルゴリズム anytime 比較（BKS gap の時間推移）", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / "combined.png", bbox_inches="tight")
        plt.close(fig)

    print(f"replotted → {out_dir}")


if __name__ == "__main__":
    main()
