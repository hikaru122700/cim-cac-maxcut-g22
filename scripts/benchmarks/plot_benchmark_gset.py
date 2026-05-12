"""Generate comparison plots of CIM/CAC/SA across all Gset inputs in results/benchmark_gset.csv."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "benchmark_gset.csv"
OUT_DIR = ROOT / "results"

METHODS = ["CIM", "CAC", "SA"]
COLORS = {"CIM": "#4c78a8", "CAC": "#f58518", "SA": "#54a24b"}


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    graphs: list[str] = []
    data: dict[str, dict[str, dict]] = {}
    for r in rows:
        g = r["graph"]
        if g not in data:
            graphs.append(g)
            data[g] = {}
        data[g][r["method"]] = {
            "mean": float(r["mean"]),
            "best": float(r["best"]),
            "std": float(r["std"]),
            "wall_s": float(r["wall_s"]),
            "bks": float(r["bks"]),
            "best_pct_bks": float(r["best_pct_bks"]),
            "n_trials": int(r["n_trials"]),
        }
    return graphs, data


def plot_best_pct(graphs, data, path):
    x = np.arange(len(graphs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, m in enumerate(METHODS):
        vals = [data[g][m]["best_pct_bks"] for g in graphs]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m, color=COLORS[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
    ax.axhline(100, color="red", ls="--", lw=1, label="known best (100%)")
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.set_ylabel("best cut  /  known best  ×100  (%)")
    ax.set_title("Best cut as % of known best — CIM vs CAC vs SA (all Gset inputs)")
    ax.legend(ncol=4, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_best_absolute(graphs, data, path):
    """Per-graph subplot: absolute best vs mean with BKS line. Uses log-free axes per graph."""
    ncol = 4
    nrow = int(np.ceil(len(graphs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.8 * nrow))
    axes = axes.flatten()
    for idx, g in enumerate(graphs):
        ax = axes[idx]
        means = [data[g][m]["mean"] for m in METHODS]
        bests = [data[g][m]["best"] for m in METHODS]
        stds = [data[g][m]["std"] for m in METHODS]
        bks = data[g]["CIM"]["bks"]
        x = np.arange(len(METHODS))
        w = 0.38
        ax.bar(x - w / 2, means, w, yerr=stds, capsize=3,
               color=[COLORS[m] for m in METHODS], alpha=0.55, label="mean ± std")
        ax.bar(x + w / 2, bests, w,
               color=[COLORS[m] for m in METHODS], label="best")
        ax.axhline(bks, color="red", ls="--", lw=1, label=f"BKS {int(bks)}")
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, fontsize=8)
        ax.set_title(f"{g}  (n={data[g]['CIM']['n_trials']})", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        lo = min(means + bests + [bks]) * 0.995
        hi = max(means + bests + [bks]) * 1.005
        ax.set_ylim(lo, hi)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")
    for j in range(len(graphs), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Mean (with std) and best cut per method — all Gset inputs",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_walltime(graphs, data, path):
    x = np.arange(len(graphs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.0))
    for i, m in enumerate(METHODS):
        vals = [data[g][m]["wall_s"] for g in graphs]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=COLORS[m])
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.set_ylabel("wall time (s, total over all trials)")
    ax.set_yscale("log")
    ax.set_title("Wall time per method across all Gset inputs (log scale)")
    ax.legend()
    ax.grid(True, axis="y", which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_gap_to_bks(graphs, data, path):
    """Gap = (BKS - best) / BKS * 100.  Negative gap means we beat the listed BKS."""
    x = np.arange(len(graphs))
    w = 0.26
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, m in enumerate(METHODS):
        vals = [(data[g][m]["bks"] - data[g][m]["best"]) / data[g][m]["bks"] * 100
                for g in graphs]
        ax.bar(x + (i - 1) * w, vals, w, label=m, color=COLORS[m])
    ax.axhline(0, color="red", ls="--", lw=1, label="= known best")
    ax.set_xticks(x)
    ax.set_xticklabels(graphs)
    ax.set_ylabel("gap to known best  (%)  — lower is better")
    ax.set_title("Optimality gap vs known best — CIM vs CAC vs SA")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    rows = load_rows()
    graphs, data = grouped(rows)
    print(f"graphs: {graphs}")
    plot_best_pct(graphs, data, OUT_DIR / "compare_all_best_pct.png")
    plot_best_absolute(graphs, data, OUT_DIR / "compare_all_per_graph.png")
    plot_walltime(graphs, data, OUT_DIR / "compare_all_walltime.png")
    plot_gap_to_bks(graphs, data, OUT_DIR / "compare_all_gap.png")
    print("wrote:")
    for p in ["compare_all_best_pct.png", "compare_all_per_graph.png",
              "compare_all_walltime.png", "compare_all_gap.png"]:
        print(" ", OUT_DIR / p)


if __name__ == "__main__":
    main()
