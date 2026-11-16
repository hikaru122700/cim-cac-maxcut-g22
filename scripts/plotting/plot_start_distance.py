"""実験 A(sa_path_bks_distance)と実験 B(start_distance_causal)の図(既存の図は上書きしない)。

  A: path_d_bks.png      SA の途中の「最も近い BKS 解までの距離」(中央値)
     final_d_hist.png    SA 後の距離の分布(CIM 開始 / ランダム開始)
     final_gap_by_distance.png  SA 後の距離ごとの BKS との差(CIM 開始 / ランダム開始)
  B: b2_gap_vs_distance.png  BKS 解から距離 d だけ崩した開始解 → SA 後の BKS との差。CIM・SA の解・ランダムの点も重ねる
     b1_start_distance.png   カット値をそろえた開始解(CIM / SA の解)の距離の分布
     b1_binned.png           開始解の距離ごとの SA 後の BKS との差(CIM / SA の解)

usage:
  python scripts/plotting/plot_start_distance.py A <sa_path_bks_distance の出力ディレクトリ> [slide]
  python scripts/plotting/plot_start_distance.py B <start_distance_causal の出力ディレクトリ> [slide]

  slide を付けると、スライドに小さく貼る用に図を小さく描き(文字が相対的に大きくなる)、*_slide.png に保存する。
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
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.size": 16,
})

ARM_STYLE = {"cim_T2": ("CIM 開始（T0 = 2.0）", "#d62728"),
             "random_T2": ("ランダム開始（T0 = 2.0）", "#1f77b4"),
             "cim_T0p5": ("CIM 開始（T0 = 0.5）", "#ff7f0e")}
RED, BLUE, GREEN, GRAY = "#d62728", "#1f77b4", "#2ca02c", "#555555"
VALLEY = 0.05
FINAL_BINS = (0.0, 0.1, 0.2, 0.3, 0.51)
FIGSIZE = (9, 5.2)
SUFFIX = ""


def _save(fig, path: Path) -> None:
    path = path.with_name(path.stem + SUFFIX + path.suffix)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _ticks(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)


def plot_a(out: Path) -> None:
    z = np.load(out / "raw.npz")
    arms = [a for a in ARM_STYLE if f"d_bks_{a}" in z.files]
    sw = z["sweeps"]
    m = sw > 0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for a in arms:
        lab, col = ARM_STYLE[a]
        ax.plot(sw[m], np.median(z[f"d_bks_{a}"], axis=0)[m], color=col, lw=2.4, label=lab)
    ax.axhline(VALLEY, color=GRAY, ls="--", lw=1.2)
    ax.text(0.012, VALLEY + 0.008, "BKS 解の谷（距離 0.05 未満）", color=GRAY, fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("SA のスイープ数")
    ax.set_ylabel("最も近い BKS 解までの距離")
    if SUFFIX:  # スライド用：凡例が線に重なるので、線の横に直接書く
        for a, (x, y) in {"random_T2": (0.012, 0.43), "cim_T2": (0.012, 0.31), "cim_T0p5": (0.012, 0.115)}.items():
            if a in arms:
                ax.text(x, y, ARM_STYLE[a][0], color=ARM_STYLE[a][1], fontsize=16)
    else:
        ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.64), frameon=False)
    _ticks(ax)
    _save(fig, out / "path_d_bks.png")

    bks = json.loads((out / "summary.json").read_text(encoding="utf-8"))["meta"]["bks"]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(FINAL_BINS) - 1)
    w = 0.38
    for k, a in enumerate(("cim_T2", "random_T2")):
        lab, col = ARM_STYLE[a]
        d = z[f"d_bks_{a}"][:, -1]
        g = (bks - z[f"final_{a}"]) / bks * 100
        vals = [g[(d >= lo) & (d < hi)].mean() for lo, hi in zip(FINAL_BINS[:-1], FINAL_BINS[1:])]
        bars = ax.bar(x + (k - 0.5) * w, vals, w, color=col, edgecolor="black", lw=0.6, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo:.1f}〜{hi:.1f}" for lo, hi in zip(FINAL_BINS[:-1], FINAL_BINS[1:])], fontsize=18)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.set_xlabel("SA 後の、最も近い BKS 解までの距離")
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.legend(loc="upper left", frameon=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    _save(fig, out / "final_gap_by_distance.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 0.5, 51)
    for a in ("random_T2", "cim_T2"):
        if a in arms:
            lab, col = ARM_STYLE[a]
            ax.hist(z[f"d_bks_{a}"][:, -1], bins=bins, color=col, alpha=0.6, edgecolor="black", lw=0.5, label=lab)
    ax.axvline(VALLEY, color=GRAY, ls="--", lw=1.2)
    ax.set_xlabel("SA 後の、最も近い BKS 解までの距離")
    ax.set_ylabel("本数")
    ax.legend(loc="upper right", frameon=False)
    _ticks(ax)
    _save(fig, out / "final_d_hist.png")


def plot_b(out: Path) -> None:
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    res = s["results"]
    b2 = res["b2_perturbed"]
    ref = res["reference_points"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # d = 0 は開始解が BKS 解そのもので、SA の「最良値」が開始解のまま BKS になるため除く
    pts = [v for k, v in b2.items() if float(k) > 0]
    d = np.array([v["start_d_bks"]["median"] for v in pts])
    g = np.array([v["final_gap"]["mean"] for v in pts])
    e = np.array([v["final_gap_ci95"] for v in pts])
    ax.errorbar(d, g, yerr=e, color=GRAY, marker="o", ms=7, lw=2, capsize=4, label="BKS 解を崩した開始解")
    ax.plot(ref["cim_start_d_bks_median"], ref["cim_T2_final_gap_mean"], "*", color=RED, ms=22, label="CIM の解")
    ax.plot(ref["random_start_d_bks_median"], ref["random_T2_final_gap_mean"], "s", color=BLUE, ms=13,
            label="ランダム解")
    if "b1_matched" in res:
        b1 = res["b1_matched"]
        ax.plot(b1["start_d_bks"]["sa_solution"]["median"], b1["final_gap"]["sa_solution"]["mean"], "D",
                color=GREEN, ms=13, label="SA の解（同じカット値）")
    ax.set_xlim(-0.01, 0.52)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("開始解と BKS 解の距離")
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.legend(loc="upper left", frameon=False, fontsize=16)
    _ticks(ax)
    _save(fig, out / "b2_gap_vs_distance.png")

    if "b1_matched" not in res:
        return
    z = np.load(out / "raw.npz")
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 0.5, 51)
    ax.hist(z["b1_d_sa_start"], bins=bins, color=GREEN, alpha=0.6, edgecolor="black", lw=0.5, label="SA の解")
    ax.hist(z["b1_d_cim_start"], bins=bins, color=RED, alpha=0.6, edgecolor="black", lw=0.5, label="CIM の解")
    ax.set_xlabel("開始解と BKS 解の距離")
    ax.set_ylabel("個数")
    ax.legend(loc="upper right", frameon=False)
    _ticks(ax)
    _save(fig, out / "b1_start_distance.png")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, lab, col, mk in (("cim", "CIM の解", RED, "o"), ("sa_solution", "SA の解", GREEN, "D")):
        rows = res["b1_matched"]["binned"][key]
        x = [(r["lo"] + r["hi"]) / 2 for r in rows]
        ax.errorbar(x, [r["gap_mean"] for r in rows], yerr=[r["gap_ci95"] for r in rows], color=col, marker=mk,
                    ms=8, lw=2, capsize=4, label=lab)
    ax.set_xlabel("開始解と BKS 解の距離")
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", frameon=False)
    _ticks(ax)
    _save(fig, out / "b1_binned.png")


def main() -> None:
    global FIGSIZE, SUFFIX
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("A", "B") or sys.argv[3:] not in ([], ["slide"]):
        sys.exit(__doc__)
    if sys.argv[3:] == ["slide"]:
        FIGSIZE, SUFFIX = (6.2, 4.6), "_slide"
    out = Path(sys.argv[2])
    (plot_a if sys.argv[1] == "A" else plot_b)(out)


if __name__ == "__main__":
    main()
