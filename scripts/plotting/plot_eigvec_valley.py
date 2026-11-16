"""実験 C(eigvec_valley)の図(既存の図は上書きしない)。

  c1_overlap.png      解の種類・谷ごとの、u1・u2 との重なり(棒グラフ)
  c2_spectral_sa.png  開始解ごとの SA 後の BKS との差(CIM の解 / sign(u1) / sign(u2) / sign(u3) / ランダム解)

  c1_overlap_counts.png  c1_overlap と同じ値で、各棒の本数と分け方をラベルに書いた版(counts を付けたときだけ)

usage:
  python scripts/plotting/plot_eigvec_valley.py <eigvec_valley の出力ディレクトリ> [slide] [counts]
  slide を付けると、スライドに小さく貼る用に小さく描き、*_slide.png に保存する。
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

RED, BLUE, GREEN, GRAY, ORANGE = "#d62728", "#1f77b4", "#2ca02c", "#555555", "#ff7f0e"
FIGSIZE = (9, 5.2)
SUFFIX = ""


def _save(fig, path: Path) -> None:
    path = path.with_name(path.stem + SUFFIX + path.suffix)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_c1(out: Path, res: dict) -> None:
    c1 = res["c1_location"]
    groups = [("BKS 解", c1["bks"], None), ("CIM の解", c1["cim_start"], None),
              ("SA 後\nBKS 側", c1["random_final"], "near"), ("SA 後\n別の谷", c1["random_final"], "far")]
    ov1 = [g["ov1"]["mean"] if k is None else g[k]["ov1_mean"] for _, g, k in groups]
    ov2 = [g["ov2"]["mean"] if k is None else g[k]["ov2_mean"] for _, g, k in groups]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(groups))
    w = 0.38
    for off, vals, col, lab in ((-0.5, ov1, RED, "$u_1$ 方向"), (0.5, ov2, BLUE, "$u_2$ 方向")):
        bars = ax.bar(x + off * w, vals, w, color=col, edgecolor="black", lw=0.6, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=16 if SUFFIX else 18)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    ax.set_ylabel("固有ベクトルとの重なり")
    ax.set_ylim(0, max(ov1 + ov2) * 1.3)
    ax.legend(loc="upper left", frameon=False, ncol=2 if SUFFIX else 1)
    _save(fig, out / "c1_overlap.png")


def plot_c1_counts(out: Path, res: dict) -> None:
    """各棒が何個の解の平均かをラベルに書く。SA 後はランダム開始の SA を、最後の解の d(< 0.3 か)で分けたもの。"""
    c1 = res["c1_location"]
    rnd = c1["random_final"]
    groups = [(f"最良解\n({c1['bks']['n']} 個)", c1["bks"]["ov1"]["mean"], c1["bks"]["ov2"]["mean"]),
              (f"CIM の解\n({c1['cim_start']['n']} 本)", c1["cim_start"]["ov1"]["mean"],
               c1["cim_start"]["ov2"]["mean"]),
              (f"SA 後\n谷の中\n({rnd['near']['n']} 本)", rnd["near"]["ov1_mean"], rnd["near"]["ov2_mean"]),
              (f"SA 後\n別の谷\n({rnd['far']['n']} 本)", rnd["far"]["ov1_mean"], rnd["far"]["ov2_mean"])]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(groups))
    w = 0.38
    for off, k, col, lab in ((-0.5, 1, RED, "$u_1$ との一致度"), (0.5, 2, BLUE, "$u_2$ との一致度")):
        vals = [g[k] for g in groups]
        bars = ax.bar(x + off * w, vals, w, color=col, edgecolor="black", lw=0.6, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=15 if SUFFIX else 18)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    ax.set_ylabel("一致度の平均")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=15 if SUFFIX else 18)
    _save(fig, out / "c1_overlap_counts.png")


def plot_c2(out: Path, res: dict) -> None:
    c2, ref = res["c2_spectral_start"], res["reference_points"]
    rows = [("CIM\nの解", ref["cim_T2_final_gap_mean"], None, RED),
            ("sign($u_1$)", c2["sign_u1"]["final_gap"]["mean"], c2["sign_u1"]["final_gap_ci95"], ORANGE),
            ("sign($u_2$)", c2["sign_u2"]["final_gap"]["mean"], c2["sign_u2"]["final_gap_ci95"], GRAY),
            ("sign($u_3$)", c2["sign_u3"]["final_gap"]["mean"], c2["sign_u3"]["final_gap_ci95"], GRAY),
            ("ランダム\n解", ref["random_T2_final_gap_mean"], None, BLUE)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.bar(x, vals, 0.6, color=[r[3] for r in rows], edgecolor="black", lw=0.6,
                  yerr=[r[2] or 0 for r in rows], capsize=4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=16 if SUFFIX else 18)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.set_ylim(0, max(vals) * 1.25)
    _save(fig, out / "c2_spectral_sa.png")


def main() -> None:
    global FIGSIZE, SUFFIX
    flags = sys.argv[2:]
    if len(sys.argv) < 2 or any(f not in ("slide", "counts") for f in flags) or len(set(flags)) != len(flags):
        sys.exit(__doc__)
    if "slide" in flags:
        FIGSIZE, SUFFIX = (6.2, 4.6), "_slide"
    out = Path(sys.argv[1])
    res = json.loads((out / "summary.json").read_text(encoding="utf-8"))["results"]
    if "counts" in flags:
        plot_c1_counts(out, res)
        return
    plot_c1(out, res)
    plot_c2(out, res)


if __name__ == "__main__":
    main()
