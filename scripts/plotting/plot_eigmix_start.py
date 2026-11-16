"""実験 G(eigmix_start)の図(既存の図は上書きしない)。

  eigmix_gap.png  開始解ごとの SA 後の BKS との差(%)。固有ベクトルを混ぜた解と、CIM の解・ランダム解を並べる。

usage:
  python scripts/plotting/plot_eigmix_start.py <eigmix_start の通常 run> <eigmix_start の --select run> [slide]
  出力は 1 つ目のディレクトリに置く。CIM・ランダム解・線形段で止めた CIM の値は、実験 C・D の summary.json から読む。
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
    "legend.fontsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16,
})
ROOT = Path(__file__).resolve().parents[2]
EIG_RUN = ROOT / "results/2026-11-15/eigvec_valley/v1_G22_S5000_T2_runs1600"
STAGE_RUN = ROOT / "results/2026-11-15/cim_stage_valley/v1_G22_D1P4_D2P100_sat0.01"
RED, BLUE, ORANGE, GRAY = "#d62728", "#1f77b4", "#ff7f0e", "#999999"


def load(path: Path) -> dict:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))["results"]


def main() -> None:
    flags = sys.argv[3:]
    if len(sys.argv) < 3 or flags not in ([], ["slide"]):
        sys.exit(__doc__)
    plain, sel = load(Path(sys.argv[1])), load(Path(sys.argv[2]))
    ref = load(EIG_RUN)["reference_points"]
    lin = np.load(STAGE_RUN / "raw.npz")["d2_gap_linear_stop"]
    rows = [("ランダム\n解", ref["random_T2_final_gap_mean"], BLUE),
            ("10 本を\n混ぜる", plain["K10"]["final_gap"]["mean"], GRAY),
            ("育ち方で\n混ぜる", plain["K50_grow"]["final_gap"]["mean"], GRAY),
            ("育ち方＋\n上位選択", sel["K50_grow_sel"]["final_gap"]["mean"], GRAY),
            ("sign\n($u_1$)", plain["K1"]["final_gap"]["mean"], ORANGE),
            ("CIM の解", ref["cim_T2_final_gap_mean"], RED)]
    if flags:
        rows.insert(5, ("線形段で\n止めた CIM", float(lin.mean()), RED))
    fig, ax = plt.subplots(figsize=(9.5, 4.8) if flags else (12, 5.2))
    x = np.arange(len(rows))
    vals = [r[1] for r in rows]
    bars = ax.bar(x, vals, 0.62, color=[r[2] for r in rows], edgecolor="black", lw=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=15)
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=15 if flags else 17)
    ax.tick_params(axis="x", which="both", top=False, bottom=False)
    ax.tick_params(axis="y", direction="in", which="both", right=True)
    ax.set_ylabel("SA 後の BKS との差 (%)")
    ax.set_ylim(0, max(vals) * 1.2)
    out = Path(sys.argv[1]) / ("eigmix_gap_slide.png" if flags else "eigmix_gap.png")
    if out.exists():
        raise FileExistsError(out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
