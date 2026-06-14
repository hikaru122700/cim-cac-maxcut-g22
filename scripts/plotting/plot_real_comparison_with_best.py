"""pumpbench_real_cim_g22 の cuts.npz から、平均(棒)に最良(マーカー)を重ねた図を描く。

comparison.png は平均±stdの棒のみで「最良(CAC=13358)」が見えない、という誤読を防ぐ改良版。
平均±std の棒 + 最良カットの星マーカー + 既知ベスト線 を 1 枚に描く。

実行: .venv/Scripts/python.exe scripts/plotting/plot_real_comparison_with_best.py
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "2026-06-14" / "pumpbench_real_cim_g22" / "v1_5cond_100trial_real"
KNOWN_BEST = 13359

LABELS = {"cac": "CAC(閉ループ)", "gain_linear": "線形利得ランプ\n(最良開ループ)",
          "linear_power": "線形電力ランプ\n(現行)", "sigmoid": "シグモイド",
          "power_early_p05": "べき乗 早上げ p=0.5"}
COLORS = {"cac": "#c0392b", "gain_linear": "#16a085", "linear_power": "#7f8c8d",
          "sigmoid": "#2c5f8a", "power_early_p05": "#d35400"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    d = np.load(RUN / "cuts.npz")
    order = sorted(d.files, key=lambda k: -d[k].max())  # 最良の高い順

    fig, ax = plt.subplots(figsize=(10, 5.6))
    xs = range(len(order))
    means = [float(d[k].mean()) for k in order]
    stds = [float(d[k].std()) for k in order]
    bests = [float(d[k].max()) for k in order]
    cols = [COLORS[k] for k in order]

    # 平均±std の棒
    ax.bar(xs, means, yerr=stds, color=cols, alpha=0.45, capsize=4,
           label="平均 ± 標準偏差")
    # 最良カットの星マーカー + 値ラベル
    ax.scatter(xs, bests, marker="*", s=320, color=cols, edgecolor="black",
               linewidth=0.8, zorder=5, label="最良カット(100 trial 中)")
    for x, b in zip(xs, bests):
        ax.annotate(f"{b:.0f}", (x, b), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=10, fontweight="bold")
    # 既知ベスト線
    ax.axhline(KNOWN_BEST, color="red", ls="--", lw=1.6, label=f"既知ベスト {KNOWN_BEST}")

    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[k] for k in order], fontsize=9)
    ax.set_ylabel("カット値", fontsize=13)
    ax.set_ylim(min(means) - 70, KNOWN_BEST + 18)
    ax.set_title("本物モデルでの平均と最良 — G22 (100 trial)\n"
                 "棒=平均±std / 星=最良。CAC のみ最良が既知ベスト線に到達", fontsize=12)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = RUN / "comparison_best.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")
    # 論文用コピー
    docs = ROOT / "docs" / "20260614" / "realmodel_comparison_best.png"
    fig_bytes = out.read_bytes()
    docs.write_bytes(fig_bytes)
    print(f"copied to: {docs}")


if __name__ == "__main__":
    main()
