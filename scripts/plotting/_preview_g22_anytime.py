"""一時プレビュー: 実行中 run の results.json から G22 の anytime 図だけ描く。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.benchmarks.algo_registry import DATASETS, ALGOS
from scripts.benchmarks.anytime_bench import setup_style

OUT_DIR = Path("results/2026-06-22/anytime_single/v1_G22_K2000_G55_G70_nt16_full120")
DS = "G22"

data = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))["results"][DS]
bks = DATASETS[DS]["bks"]
plt = setup_style()

fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=140)
for algo_key, pts in data.items():
    if not pts:
        continue
    t = [p["time"] for p in pts]
    gap = [100.0 * (bks - p["cut_max"]) / bks for p in pts]
    ax.plot(t, gap, "-o", color=ALGOS[algo_key]["color"],
            lw=2.0, ms=5, label=ALGOS[algo_key]["label"])
ax.axhline(0.0, color="k", ls=":", lw=1.2)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=0.05)
ax.set_ylim(bottom=0.0)
ax.set_xlabel("実時間 [秒]（バッチ, log）")
ax.set_ylabel("BKS への gap [%]（最良解, 小さいほど良い）")
ax.set_title(f"{DS}: BKS への到達 gap の時間推移")
ax.tick_params(direction="in", which="both", top=True, right=True)
ax.legend(fontsize=8, ncol=2)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
out = OUT_DIR / "G22_gap_preview.png"
fig.savefig(out, bbox_inches="tight")
print(f"saved → {out}")
