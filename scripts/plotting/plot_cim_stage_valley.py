"""実験 D(cim_stage_valley)の図(既存の図は上書きしない)。

  d1_path.png         CIM の途中の sign(c) の、最も近い BKS 解までの距離と E = ov1 − ov2(中央値)
  d2_linear_stop.png  線形段で止めた CIM の解 / 最後まで回した CIM の解 / sign(u1) / ランダム解 からの SA 後の差
  d2_start_d.png      開始解と BKS 解の距離の分布(線形段で止めた解 vs 最後まで回した解)

usage:
  python scripts/plotting/plot_cim_stage_valley.py <cim_stage_valley の出力ディレクトリ> [slide] [eig=<eigvec_valley の出力ディレクトリ>]
  eig= は sign(u1) とランダム解の値を取る実験 C の run(既定は 2026-11-13 の v1、960 本)。
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "xtick.direction": "in", "ytick.direction": "in",
    "axes.labelsize": 20, "axes.titlesize": 20, "legend.fontsize": 18, "xtick.labelsize": 16,
    "ytick.labelsize": 16, "font.size": 16,
})
RED, BLUE, ORANGE, GRAY = "#d62728", "#1f77b4", "#ff7f0e", "#555555"
EIG_RUN = Path("results/2026-11-13/eigvec_valley/v1_G22_S5000_T2_runs960")
FIGSIZE, SUFFIX = (9, 5.2), ""


def _save(fig, path: Path) -> None:
    path = path.with_name(path.stem + SUFFIX + path.suffix)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def _style(ax) -> None:
    ax.tick_params(direction="in", which="both", top=True, right=True)


def plot_path(out: Path, z) -> None:
    r = z["d1_rounds"]
    d = np.median(z["d1_d_bks"], axis=0)
    e = np.median(z["d1_ov1"] - z["d1_ov2"], axis=0)
    k = float(np.median(z["d1_k_lin"]))
    fig, ax = plt.subplots(figsize=FIGSIZE)
    m = r >= 100
    ax.plot(r[m], d[m], color=RED, lw=3, label="最良解までの距離")
    ax.plot(r[m], e[m], color=BLUE, lw=3, label="$u_1$ − $u_2$ の重なり")
    ax.axvline(k, color=GRAY, ls="--", lw=2)
    ax.text(k * 1.05, 0.46, "線形段の終わり", fontsize=16, color=GRAY)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 0.52)
    ax.set_xlabel("CIM のラウンド")
    ax.set_ylabel("中央値")
    ax.legend(loc="center right", frameon=False)
    _style(ax)
    _save(fig, out / "d1_path.png")


def plot_linear_stop(out: Path, res: dict) -> None:
    d2 = res["d2_linear_stop"]
    c = json.loads((EIG_RUN / "summary.json").read_text(encoding="utf-8"))["results"]
    rows = [("最後まで\n回した CIM", d2["final_gap"]["full_cim"]["mean"], d2["final_gap_ci95"]["full_cim"], RED),
            ("線形段で\n止めた CIM", d2["final_gap"]["linear_stop"]["mean"], d2["final_gap_ci95"]["linear_stop"], ORANGE),
            ("sign($u_1$)", c["c2_spectral_start"]["sign_u1"]["final_gap"]["mean"],
             c["c2_spectral_start"]["sign_u1"]["final_gap_ci95"], GRAY),
            ("ランダム\n解", c["reference_points"]["random_T2_final_gap_mean"], 0.0, BLUE)]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(rows))
    vals = [q[1] for q in rows]
    bars = ax.bar(x, vals, 0.6, color=[q[3] for q in rows], edgecolor="black", lw=0.6,
                  yerr=[q[2] for q in rows], capsize=4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([q[0] for q in rows], fontsize=16 if SUFFIX else 18)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.set_ylim(0, max(vals) * 1.3)
    _save(fig, out / "d2_linear_stop.png")


def plot_start_d(out: Path, z) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, 0.5, 51)
    ax.hist(z["d2_d_full"], bins=bins, color=RED, alpha=0.7, label="最後まで回した CIM")
    ax.hist(z["d2_d_linear_stop"], bins=bins, color=ORANGE, alpha=0.7, label="線形段で止めた CIM")
    ax.axvline(0.3, color=GRAY, ls="--", lw=2)
    ax.set_xlabel("開始解と BKS 解の距離")
    ax.set_ylabel("個数")
    ax.legend(loc="upper right", frameon=False)
    _style(ax)
    _save(fig, out / "d2_start_d.png")


def main() -> None:
    global FIGSIZE, SUFFIX
    global EIG_RUN
    rest = sys.argv[2:]
    eig = [a for a in rest if a.startswith("eig=")]
    flags = [a for a in rest if not a.startswith("eig=")]
    if len(sys.argv) < 2 or flags not in ([], ["slide"]) or len(eig) > 1:
        sys.exit(__doc__)
    if eig:
        EIG_RUN = Path(eig[0][4:])
        if not (EIG_RUN / "summary.json").exists():
            sys.exit(f"summary.json not found in {EIG_RUN}")
    if flags == ["slide"]:
        FIGSIZE, SUFFIX = (6.2, 4.6), "_slide"
    out = Path(sys.argv[1])
    res = json.loads((out / "summary.json").read_text(encoding="utf-8"))["results"]
    z = np.load(out / "raw.npz")
    plot_path(out, z)
    plot_linear_stop(out, res)
    plot_start_d(out, z)


if __name__ == "__main__":
    main()
