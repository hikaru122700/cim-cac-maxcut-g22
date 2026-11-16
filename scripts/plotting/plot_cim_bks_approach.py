"""実験 F(cim_bks_approach)の図(既存の図は上書きしない)。

  approach_d.png    ラウンドごとの最良解までの距離 d(中央値と四分位)。カット値上位 16 本と残り 240 本を分けて描く。
  spin_settle.png   1 本の CIM で、各スピンが最良解と同じ向きか(上位の解 / 選ばれない解)。横軸ラウンド、縦軸スピン。

usage:
  python scripts/plotting/plot_cim_bks_approach.py <cim_bks_approach の出力ディレクトリ> [slide]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": 22, "axes.titlesize": 22,
    "legend.fontsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16,
})
RED, BLUE, GRAY = "#d62728", "#1f77b4", "#555555"
X_MAX = 800


def save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_approach(z, out: Path, suffix: str, size) -> None:
    r, d, sel = z["rounds"], z["d"], z["selected"]
    k = float(np.median(z["k_lin"]))
    m = r <= X_MAX
    fig, ax = plt.subplots(figsize=size)
    ax.axvspan(0, k, color="#ececec", zorder=0)
    ax.axhline(0.3, color=GRAY, ls=":", lw=2)
    for mask, col, lab in ((sel, RED, "選ばれる解(上位 16/256)"), (~sel, BLUE, "選ばれない解(残り 240)")):
        q1, med, q3 = np.percentile(d[mask][:, m], [25, 50, 75], axis=0)
        ax.fill_between(r[m], q1, q3, color=col, alpha=0.2, lw=0)
        ax.plot(r[m], med, color=col, lw=3, label=lab)
    ax.axvline(k, color=RED, ls="--", lw=2)
    ax.text(k / 2, 0.06, "線形段", ha="center", fontsize=20)
    ax.text(X_MAX * 0.98, 0.31, "谷の境 0.3", ha="right", va="bottom", fontsize=18, color=GRAY)
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("CIM のラウンド")
    ax.set_ylabel("最良解までの距離 d")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper right", frameon=False, fontsize=18 if suffix else 20)
    save(fig, out / f"approach_d{suffix}.png")


def plot_settle(z, out: Path, suffix: str, size) -> None:
    r = z["rounds"]
    m = r <= X_MAX
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    cmap = ListedColormap(["#9ecae1", "#d62728"])
    for ax, nm, title in zip(axes, ("selected", "other"), ("選ばれる解", "選ばれない解")):
        agree = (z[f"{nm}_aligned"][m] > 0).astype(np.int8)
        settle = z[f"{nm}_settle"].astype(np.float64)
        order = np.argsort(np.where(settle < 0, np.inf, settle), kind="stable")
        ax.imshow(agree[:, order].T, aspect="auto", cmap=cmap, interpolation="nearest",
                  extent=(r[m][0], r[m][-1], len(order), 0))
        ax.axvline(float(np.median(z["k_lin"])), color="black", ls="--", lw=2)
        ax.set_title(f"{title}(最後に違う {int((settle < 0).sum())} 個)", fontsize=20)
        ax.set_xlabel("CIM のラウンド")
        ax.tick_params(direction="in", which="both", top=True, right=True)
    axes[0].set_ylabel("スピン(向きが決まった順)")
    fig.text(0.5, -0.04, "赤:最良解と同じ向き  水色:逆向き", ha="center", fontsize=20)
    fig.tight_layout()
    save(fig, out / f"spin_settle{suffix}.png")


def main() -> None:
    flags = sys.argv[2:]
    if len(sys.argv) < 2 or flags not in ([], ["slide"]):
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    z = np.load(out / "raw.npz")
    suffix = "_slide" if flags else ""
    plot_approach(z, out, suffix, (6.2, 4.6) if flags else (9, 5.2))
    plot_settle(z, out, suffix, (9, 4.6) if flags else (12, 5.2))


if __name__ == "__main__":
    main()
