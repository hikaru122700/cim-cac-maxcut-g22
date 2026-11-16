"""bks_set_distance の結果を図にする(既存の図は上書きしない)。

  distance_single_vs_nearest.png  結果③の 4 集合の平均距離:基準解 1 つ vs 集めた BKS 解の中で最も近いもの
  bks_pairwise_hist.png            集めた BKS 解どうしの距離の分布

usage: python scripts/plotting/plot_bks_set_distance.py <bks_set_distance の出力ディレクトリ>
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic",
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "legend.fontsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 14,
    "font.size": 14,
})

SETS = [("random_start", "ランダム\n開始"), ("random_sa", "ランダム\n→SA"),
        ("cim_start", "CIM\n開始"), ("cim_sa", "CIM\n→SA")]
BLUE, RED = "#1f77b4", "#d62728"


def _save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def fig_single_vs_nearest(summary: dict, out: Path) -> None:
    res = summary["results"]
    single = [res[k]["single_reference"]["mean"] for k, _ in SETS]
    nearest = [res[k]["nearest_bks"]["mean"] for k, _ in SETS]
    x = np.arange(len(SETS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w / 2, single, w, color=BLUE, edgecolor="black", lw=0.8, label="基準の BKS 解 1 つ")
    b2 = ax.bar(x + w / 2, nearest, w, color=RED, edgecolor="black", lw=0.8,
                label=f"最も近い BKS 解({summary['meta']['reference_set_size']} 個から)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.3f}",
                    ha="center", va="bottom", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in SETS], fontsize=20)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_ylabel("BKS 解との距離(平均)")
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", frameon=False)
    _save(fig, out / "distance_single_vs_nearest.png")


def fig_pairwise_hist(raw: np.lib.npyio.NpzFile, out: Path) -> None:
    d = raw["bks_pairwise"]
    iu = np.triu_indices(d.shape[0], k=1)
    spins = np.rint(d[iu] * raw["bks_signs"].shape[1]).astype(int)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.arange(spins.min() - 0.5, spins.max() + 1.5, 1.0)
    ax.hist(spins, bins=bins, color=BLUE, edgecolor="black", lw=0.6)
    ax.set_xlabel("BKS 解どうしで違うスピンの数(2000 個中)")
    ax.set_ylabel("組の数")
    _save(fig, out / "bks_pairwise_hist.png")


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    raw = np.load(out / "raw.npz")
    fig_single_vs_nearest(summary, out)
    fig_pairwise_hist(raw, out)


if __name__ == "__main__":
    main()
