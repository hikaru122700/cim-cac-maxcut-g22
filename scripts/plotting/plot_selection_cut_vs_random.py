"""G22 で、カット値による選別は無作為より良く、副指標は上積みしないことを示す図。

保存済みの selection_eval の主条件(T0=0.5, 5000 掃引, K=256, m=16)の結果だけから描く。

出力:results/<日付>/selection_cut_vs_random/v{N}_<tag>/ に PNG と PDF。

使い方(プロジェクトルートから):
  python scripts/plotting/plot_selection_cut_vs_random.py --tag g22
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "selection_cut_vs_random"
RESULT = ROOT / "results" / "2026-10-19" / "selection_eval" / "v17_main_T0p5_S5000_K256_m16" / "result.json"
PROBLEM = "G22"
# (方法, 横軸の名前, 色)
BARS = [("cim_random", "無作為", "#8A9BA8"), ("cim_cut", "カット値", "#2F6B9A"),
        ("cim_div", "カット値＋\nハミング距離", "#C34436")]
FS = 22  # 文字・数値とも 20 以上

plt.rcParams.update({
    "font.family": "Yu Gothic",
    "axes.unicode_minus": False,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "font.size": FS,
})


def get_out_dir(tag: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{tag}"
    out.mkdir(parents=True)
    return out


def draw(ax, strategies: dict, title: str | None):
    gaps = [strategies[s]["gap_to_bks_pct"] for s, _, _ in BARS]
    x = range(len(BARS))
    ax.bar(x, gaps, width=0.6, color=[c for _, _, c in BARS])
    for xx, g in zip(x, gaps):
        ax.text(xx, g + max(gaps) * 0.02, f"{g:.3f}", ha="center", va="bottom", fontsize=FS)
    ax.set_xticks(list(x), [name for _, name, _ in BARS])
    ax.set_ylabel("BKSとの差（%）")
    ax.set_ylim(0, max(gaps) * 1.18)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    if title:
        ax.set_title(title, fontsize=FS, pad=12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=[PROBLEM])
    ap.add_argument("--tag", default="g22")
    args = ap.parse_args()
    per_problem = json.loads(RESULT.read_text(encoding="utf-8"))["per_problem"]
    missing = [p for p in args.problems if p not in per_problem]
    if missing:
        ap.error(f"not in {RESULT.parent.name}: {missing} (available: {list(per_problem)})")

    n = len(args.problems)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 7.5), squeeze=False)
    for ax, p in zip(axes[0], args.problems):
        draw(ax, per_problem[p]["strategies"], p if n > 1 else None)
    fig.tight_layout()

    out = get_out_dir(args.tag)
    fig.savefig(out / "selection_cut_vs_random.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "selection_cut_vs_random.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
