"""実験 E(valley_other_graphs)の図(既存の図は上書きしない)。

  b2_curve.png   参照解から距離 d だけ崩した開始解の SA 後のギャップ(d ごとの平均と 95% 区間)。CIM の解からの SA を星で重ねる
  start_d.png    最良解(参照解)までの距離 d の分布:CIM の解(上位 16/256)、CIM の解(選ぶ前)、ランダム開始の SA 後

usage:
  python scripts/plotting/plot_valley_other_graphs.py <valley_other_graphs の出力ディレクトリ> [slide]
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": 22, "axes.titlesize": 22,
    "legend.fontsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16,
})
RED, BLUE, GRAY, BLACK = "#d62728", "#1f77b4", "#888888", "#222222"


def save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_b2(out: Path, res: dict, meta: dict, suffix: str, size) -> None:
    b2 = res["b2_perturbed"]
    ds = np.array([float(k) for k in b2])
    g = np.array([b2[k]["final_gap"]["mean"] for k in b2])
    ci = np.array([b2[k]["final_gap_ci95"] for k in b2])
    fig, ax = plt.subplots(figsize=size)
    ax.axvspan(0, 0.3, color="#eeeeee", zorder=0)
    ax.errorbar(ds, g, yerr=ci, fmt="o-", color=BLACK, lw=2.5, ms=8, capsize=4, label="最良解を d だけ崩して SA")
    st = res["starts"]
    ax.plot(st["cim_start"]["d"]["median"], st["cim_final"]["gap"]["mean"], "*", color=RED, ms=22,
            label="CIM の解から SA")
    ax.plot(0.5, st["random_final"]["gap"]["mean"], "s", color=BLUE, ms=12, label="ランダム解から SA")
    ax.set_xlabel("開始解と最良解の距離 d")
    ax.set_ylabel("SA 後のギャップ (%)")
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0, max(g.max(), st["random_final"]["gap"]["mean"]) * 1.15)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper left", frameon=False, fontsize=15 if suffix else 18)
    save(fig, out / f"b2_curve{suffix}.png")


def plot_start_d(out: Path, raw, suffix: str, size) -> None:
    bins = np.linspace(0.1, 0.5, 41)
    fig, ax = plt.subplots(figsize=size)
    for key, col, lab in (("d_cim_start", RED, "CIM の解(上位 16/256)"), ("d_cim_raw", "#ff9896", "CIM の解(選ぶ前)"),
                          ("d_random_final", BLUE, "ランダム開始の SA 後")):
        ax.hist(raw[key], bins=bins, color=col, alpha=0.6, label=lab)
    ax.axvline(0.3, color=BLACK, ls="--", lw=2)
    ax.set_xlabel("最良解までの距離 d")
    ax.set_ylabel("個数")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper left", frameon=False, fontsize=15 if suffix else 18)
    save(fig, out / f"start_d{suffix}.png")


def main() -> None:
    flags = sys.argv[2:]
    if len(sys.argv) < 2 or flags not in ([], ["slide"]):
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    raw = np.load(out / "raw.npz")
    suffix, size = ("_slide", (6.2, 4.6)) if flags else ("", (9, 5.2))
    plot_b2(out, summary["results"], summary["meta"], suffix, size)
    plot_start_d(out, raw, suffix, size)


if __name__ == "__main__":
    main()
