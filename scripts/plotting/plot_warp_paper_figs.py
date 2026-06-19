"""plot_warp_paper_figs.py — WARP 論文が参照する 2 図を実データから生成。

- relative_gap_by_instance.png : 各ソルバの BKS への相対 gap(%) を 4 インスタンスで比較(グループ棒)
- warmstart_rescue.png         : 物理ソルバ単体 vs Tabu 精錬ハイブリッドの絶対 gap(各インスタンス)

入力: analysis_summary/v3_final/summary.json
実行: python scripts/plotting/plot_warp_paper_figs.py <summary.json> <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    summ = json.load(open(sys.argv[1], encoding="utf-8"))
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    insts = ["G22", "K2000", "G55", "G70"]
    algos = [("CIM", "#e74c3c"), ("CAC", "#e67e22"), ("SA", "#2980b9"),
             ("SB", "#16a085"), ("PT", "#8e44ad"), ("GA", "#27ae60")]
    labels = {"CIM": "CIM", "CAC": "CAC", "SA": "SA", "SB": "SB(dSB)",
              "PT": "PT-ICM", "GA": "GA(memetic)"}

    # --- Fig 1: relative gap (%) by instance (grouped bars) ---
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    x = np.arange(len(insts))
    w = 0.13
    for i, (a, c) in enumerate(algos):
        vals = [summ[ds]["single"][a]["gap_pct"] for ds in insts]
        ax.bar(x + (i - 2.5) * w, vals, w, label=labels[a], color=c)
    ax.set_xticks(x); ax.set_xticklabels(insts)
    ax.set_ylabel("BKS への相対 gap [%]（小さいほど良い）")
    ax.set_yscale("log")
    ax.set_title("各ソルバの相対 gap（4 インスタンス）")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(alpha=0.25, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out / "relative_gap_by_instance.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: warm-start rescue (standalone vs hybrid, absolute gap) ---
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=150)
    series = [
        ("CIM 単体", lambda ds: summ[ds]["single"]["CIM"]["gap"], "#e74c3c", 0.0),
        ("CIM→TS", lambda ds: summ[ds]["hybrid"].get("CIM_TS", {}).get("gap"), "#c0392b", 1.0),
        ("CAC 単体", lambda ds: summ[ds]["single"]["CAC"]["gap"], "#e67e22", 2.0),
        ("CAC→TS", lambda ds: summ[ds]["hybrid"].get("CAC_TS", {}).get("gap"), "#b9770e", 3.0),
    ]
    w = 0.2
    for nm, fn, c, off in series:
        vals = [fn(ds) for ds in insts]
        ax.bar(x + (off - 1.5) * w, vals, w, label=nm, color=c)
    ax.set_xticks(x); ax.set_xticklabels(insts)
    ax.set_ylabel("BKS への絶対 gap（小さいほど良い）")
    ax.set_yscale("log")
    ax.set_title("ウォームスタートによる救済：物理ソルバ単体 vs Tabu 精錬ハイブリッド")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, axis="y", which="both")
    fig.tight_layout()
    fig.savefig(out / "warmstart_rescue.png", bbox_inches="tight")
    plt.close(fig)

    print("saved:", out / "relative_gap_by_instance.png")
    print("saved:", out / "warmstart_rescue.png")


if __name__ == "__main__":
    main()
