"""検証 4(GA を CIM の生成時間込みで比べる)の棒グラフ。ga_equal_time_analysis.py の result.json から描く。

評価時刻 T_eval での BKS との差(16 回の平均、ひげは標準誤差)。CIM の時間の倍率 c は --c で選ぶ(主条件は 1)。

出力:results/<日付>/ga_equal_time_fig/v{N}_<tag>/ に PNG と PDF。

使い方(プロジェクトルートから):
  python scripts/plotting/plot_ga_equal_time.py --result results/2026-10-19/ga_equal_time/v2_.../analysis/result.json --problems G55 G70
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "ga_equal_time_fig"
# (条件, 横軸の名前, 色)
BARS = [("random", "ランダム", "#8A9BA8"), ("cim_all", "CIM\n（選別なし）", "#E0A030"),
        ("cim_cut", "CIM＋\nカット値", "#2F6B9A"), ("cim_div", "CIM＋カット値＋\nハミング距離", "#C34436")]
FS = 22  # 文字・数値とも 20 以上

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": FS,
    "axes.titlesize": FS, "xtick.labelsize": 20, "ytick.labelsize": FS, "font.size": FS,
})


def get_out_dir(tag: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{tag}"
    out.mkdir(parents=True)
    return out


def draw(ax, row: dict, title: str):
    gaps = [np.asarray(row["gap_pct"][k]) for k, _, _ in BARS]
    m = np.array([g.mean() for g in gaps])
    se = np.array([g.std(ddof=1) / np.sqrt(len(g)) for g in gaps])
    x = np.arange(len(BARS))
    ax.bar(x, m, width=0.62, color=[c for _, _, c in BARS], yerr=se, capsize=8, error_kw={"lw": 2})
    for xx, mm, ss in zip(x, m, se):
        ax.text(xx, mm + ss + m.max() * 0.02, f"{mm:.3f}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(x, [name for _, name, _ in BARS])
    ax.set_ylabel("BKSとの差（%）")
    ax.set_ylim(0, (m + se).max() * 1.2)
    ax.set_title(title, pad=12)
    ax.tick_params(direction="in", which="both", top=True, right=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--problems", nargs="+", default=["G55", "G70"])
    ap.add_argument("--c", default="1.0", help="CIM の時間の倍率(result.json の by_c のキー)")
    ap.add_argument("--tag", default="c1")
    args = ap.parse_args()
    res = json.loads(args.result.read_text(encoding="utf-8"))
    missing = [p for p in args.problems if p not in res["datasets"]]
    if missing:
        ap.error(f"not in result: {missing}")
    n = len(args.problems)
    fig, axes = plt.subplots(1, n, figsize=(12 * n, 8), squeeze=False)
    for ax, p in zip(axes[0], args.problems):
        draw(ax, res["datasets"][p]["by_c"][args.c], p)
    fig.tight_layout()
    out = get_out_dir(args.tag)
    fig.savefig(out / "ga_equal_time.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "ga_equal_time.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
