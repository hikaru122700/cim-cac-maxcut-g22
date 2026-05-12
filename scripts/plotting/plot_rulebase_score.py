"""ルールベース貪欲法の全 Gset スコアを画像化する。"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

from modules.CIM import load_graph
from rulebase import rulebase_maxcut


# 単純(重みなし)Gset の既知最良値
KNOWN_BEST = {
    "G1": 11624, "G14": 3064, "G15": 3050, "G22": 13359, "G23": 13344,
    "G55": 10299, "G70": 9591,
}

# 参考: G22 における他手法のスコア(これまでの結果から)
G22_REFERENCES = {
    "CIM (論文値)": 13276,
    "CIM (Optuna 最適)": 13340,
    "既知最良値": 13359,
}


def main():
    rows = []
    for name, kb in KNOWN_BEST.items():
        n, k, _, edges = load_graph(f"input/{name}.txt")
        t0 = time.perf_counter()
        _, cut, _ = rulebase_maxcut(n, edges)
        ms = (time.perf_counter() - t0) * 1000
        rows.append({
            "name": name, "n": n, "k": k,
            "cut": cut, "known_best": kb,
            "ratio": cut / kb, "time_ms": ms,
        })
        print(f"{name:<5} N={n:>5} K={k:>6} cut={cut:>6} ratio={cut/kb:.4f}")

    names = [r["name"] for r in rows]
    cuts = np.array([r["cut"] for r in rows])
    kbs = np.array([r["known_best"] for r in rows])
    ratios = cuts / kbs * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=130,
                                    gridspec_kw={"width_ratios": [2, 1]})

    # --- 左パネル: 各グラフでのスコア比率 ---
    bars = ax1.barh(names, ratios, color="#1f77b4", alpha=0.85,
                    edgecolor="black", linewidth=0.5)
    ax1.axvline(100, color="goldenrod", linestyle="--", linewidth=1.5,
                label="既知最良値 (100%)")
    for bar, r, cut, kb in zip(bars, ratios, cuts, kbs):
        ax1.text(r + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{r:.1f}%  ({cut} / {kb})",
                 va="center", fontsize=10)
    ax1.set_xlim(0, 115)
    ax1.set_xlabel("カット数 / 既知最良値 [%]")
    ax1.set_title("ルールベース貪欲法のスコア (Gset 重みなしグラフ)")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(axis="x", alpha=0.3)
    ax1.invert_yaxis()
    ax1.tick_params(direction="in", which="both", top=True, right=True)

    # --- 右パネル: G22 で他手法と比較 ---
    methods = ["ルールベース\n(本実装)", "CIM\n(論文値)", "CIM\n(Optuna 最適)"]
    g22_row = next(r for r in rows if r["name"] == "G22")
    values = [g22_row["cut"], G22_REFERENCES["CIM (論文値)"], G22_REFERENCES["CIM (Optuna 最適)"]]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    bars2 = ax2.bar(methods, values, color=colors, alpha=0.85,
                     edgecolor="black", linewidth=0.5)
    ax2.axhline(G22_REFERENCES["既知最良値"], color="goldenrod",
                linestyle="--", linewidth=1.5,
                label=f"既知最良値 {G22_REFERENCES['既知最良値']}")
    for bar, v in zip(bars2, values):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 60,
                 f"{v}\n({v/13359*100:.1f}%)",
                 ha="center", fontsize=10)
    ax2.set_ylim(10000, 13700)
    ax2.set_ylabel("カット数")
    ax2.set_title("G22 における手法比較")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.tick_params(direction="in", which="both", top=True, right=True)

    fig.tight_layout()

    os.makedirs("results", exist_ok=True)
    out = "results/v1_rulebase_score.png"
    i = 1
    while os.path.exists(out):
        i += 1
        out = f"results/v{i}_rulebase_score.png"
    fig.savefig(out)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
