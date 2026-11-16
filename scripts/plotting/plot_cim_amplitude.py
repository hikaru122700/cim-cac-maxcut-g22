"""実験 D(cim_stage_valley)の D1 から、CIM の振幅の大きさ(rms)がラウンドとともにどう育つかを描く(既存の図は上書きしない)。

  d1_amplitude.png  左:振幅(普通の軸)、右:振幅(対数軸)。1024 本の中央値。線形段の終わり(中央値 236)を赤い破線で示す。

usage:
  python scripts/plotting/plot_cim_amplitude.py <cim_stage_valley の出力ディレクトリ> [slide]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": 20, "axes.titlesize": 20,
    "legend.fontsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16,
})
RED, BLACK = "#d62728", "#222222"


def main() -> None:
    flags = sys.argv[2:]
    if len(sys.argv) < 2 or flags not in ([], ["slide"]):
        sys.exit(__doc__)
    out = Path(sys.argv[1])
    path = out / ("d1_amplitude_slide.png" if flags else "d1_amplitude.png")
    if path.exists():
        raise FileExistsError(path)
    z = np.load(out / "raw.npz")
    r, rms, k = z["d1_rounds"], np.median(z["d1_rms"], axis=0), float(np.median(z["d1_k_lin"]))
    m = r <= 600
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6) if flags else (13, 5.2))
    for ax, log in zip(axes, (False, True)):
        ax.axvspan(0, k, color="#e8e8e8", zorder=0)
        ax.plot(r[m], rms[m], color=BLACK, lw=3)
        ax.axvline(k, color=RED, ls="--", lw=2.5)
        ax.set_xlim(0, 600)
        ax.set_xlabel("CIM のラウンド")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        if log:
            ax.set_yscale("log")
            ax.set_title("縦軸を対数に", fontsize=20)
        else:
            ax.set_ylabel("振幅の大きさ(rms)")
            ax.set_title("普通の軸", fontsize=20)
        top = ax.get_ylim()[1]
        ax.text(k / 2, top * (0.5 if log else 0.9), "線形段", ha="center", fontsize=20)
        ax.text(k + (12 if log else 60), top * (0.02 if log else 0.25), f"線形段の終わり\nラウンド {k:.0f}", color=RED, fontsize=18)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
