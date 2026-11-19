"""実験 E(valley_other_graphs)の G55 と G70 を 1 枚に並べる比較図(スライド用)。

  b2_compare_slide.png      最良解を d だけ崩した開始解からの SA 後のギャップ(グラフごとの線)。CIM の解からの SA を星で重ねる
  valley_frac_slide.png     カット値上位 16/256 の CIM の解のうち、最良解の近傍(d < 0.3)にある割合(G22・G55・G70)

出力先は results/<今日>/valley_graphs_compare/v{N}_<グラフ名>/(既存の図は上書きしない)。

usage:
  python scripts/plotting/plot_valley_graphs_compare.py <valley_other_graphs の出力ディレクトリ> [...]
"""

import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": 22, "axes.titlesize": 22,
    "legend.fontsize": 18, "xtick.labelsize": 18, "ytick.labelsize": 16, "font.size": 16,
})
EXPERIMENT_KIND = "valley_graphs_compare"
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#ff7f0e")
VALLEY_EDGE = "0.3"
# G22 の値は results/2026-11-14/cim_stage_valley/v2_G22_D1P4_D2P60_sat0.01/RESULTS.md(上位 16/256 の 99.7% が最良解の近傍内)
G22_VALLEY_FRAC = 0.997
SLIDE_SIZE = (6.2, 4.6)


def next_out_dir(names: list[str]) -> Path:
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    vs = [int(p.name.split("_", 1)[0][1:]) for p in root.iterdir()
          if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    out = root / f"v{max(vs, default=0) + 1}_{'_'.join(names)}"
    out.mkdir()
    return out


def save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def plot_b2(out: Path, runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    ax.axvspan(0, 0.3, color="#eeeeee", zorder=0)
    for run, col in zip(runs, COLORS):
        b2 = run["results"]["b2_perturbed"]
        ds = np.array([float(k) for k in b2])
        g = np.array([b2[k]["final_gap"]["mean"] for k in b2])
        ax.plot(ds, g, "o-", color=col, lw=2.5, ms=7, label=run["meta"]["graph"])
        st = run["results"]["starts"]
        ax.plot(st["cim_start"]["d"]["median"], st["cim_final"]["gap"]["mean"], "*", color=col, ms=22,
                markeredgecolor="black")
    ax.plot([], [], "*", color="#888888", ms=18, markeredgecolor="black", label="CIM の解から SA")
    ax.set_xlabel("開始解と最良解の距離 d")
    ax.set_ylabel("SA 後のギャップ (%)")
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0, None)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper left", frameon=False)
    save(fig, out / "b2_compare_slide.png")


def plot_valley_frac(out: Path, runs: list[dict]) -> None:
    names = ["G22"] + [r["meta"]["graph"] for r in runs]
    fracs = [G22_VALLEY_FRAC * 100] + [r["results"]["starts"]["cim_start"]["in_valley"][VALLEY_EDGE] * 100 for r in runs]
    fig, ax = plt.subplots(figsize=SLIDE_SIZE)
    bars = ax.bar(names, fracs, color=["#222222"] + list(COLORS[:len(runs)]), width=0.6)
    ax.bar_label(bars, labels=[f"{f:.1f}%" for f in fracs], fontsize=18, padding=3)
    ax.set_ylabel("最良解の近傍内の割合 (%)")
    ax.set_ylim(0, 115)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    save(fig, out / "valley_frac_slide.png")


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    runs = [json.loads((Path(d) / "summary.json").read_text(encoding="utf-8")) for d in sys.argv[1:]]
    out = next_out_dir([r["meta"]["graph"] for r in runs] + ["T0p5"] * all(r["meta"]["t_start"] == 0.5 for r in runs))
    plot_b2(out, runs)
    plot_valley_frac(out, runs)


if __name__ == "__main__":
    main()
