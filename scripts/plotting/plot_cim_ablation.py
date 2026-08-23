"""plot_cim_ablation.py — cim_diversity_ablation の結果を作図する。

使い方(プロジェクトルートから):
    .venv/Scripts/python.exe -u scripts/plotting/plot_cim_ablation.py \
        results/<date>/cim_diversity_ablation/v{N}_... \
        [--ref results/2026-08-18/solution_diversity/v2_.../results.json]

出力(同じ run ディレクトリ):
    fig1_quality_diversity.png … 品質×多様性平面。他手法(参照 run)との位置関係つき
    fig2_sweeps.png            … 実験ごとのノブ応答(gap% と平均ペア距離の二軸)
    fig3_threshold.png         … 線形増幅段の長さ vs 多様性(仮説2の直接検証)
    fig4_refined.png           … 共通 Tabu で磨いた後の品質×多様性(--refined-suffix 指定時)
    fig5_alignment.png         … spectral 緩和解との一致度 vs 多様性
                                 (spectral_alignment.csv がある場合)

「磨いた後」の結果があると、品質と多様性の交絡(悪い解ほど散らばる)を落とした
比較になる。生成には先に diversity_refine.py を同じ run_dir に対して走らせる。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "figure.titlesize": 22,
    "legend.fontsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.size": 14,
})

DEFAULT_REF = (ROOT / "results" / "2026-08-18" / "solution_diversity"
               / "v2_G22_K2000_G55_G70_nt32_main" / "results.json")

EXP_INFO = {
    1: {"title": "実験1: 初期振幅のランダム化", "color": "#2980b9", "knob": "初期振幅"},
    2: {"title": "実験2: ポンプ ramp 速度",     "color": "#27ae60", "knob": "ramp 倍率"},
    3: {"title": "実験3: 真空雑音の振幅",       "color": "#8e44ad", "knob": "ノイズ倍率"},
    4: {"title": "実験4: 同時更新をやめる",     "color": "#d35400", "knob": "更新方式"},
}
BASE_COLOR = "#e74c3c"


def style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)


def load(run_dir: Path):
    with open(run_dir / "results.json", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["meta"], payload["results"]


def load_ref(path: Path) -> dict:
    """参照 run(6 手法の多様性ベンチ)から {ds: {algo: (gap, dist)}} を作る。"""
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        p = json.load(f)
    out = {}
    for ds, algos in p["results"].items():
        out[ds] = {a: (s["gap_pct_mean"], s["mean_pairwise"]) for a, s in algos.items()}
    return out


def ordered_configs(res_ds: dict, exp: int) -> list[tuple[str, dict]]:
    """指定実験の条件を results.json の並び順で返す(baseline は含めない)。"""
    return [(k, v) for k, v in res_ds.items() if v["exp"] == exp]


# ============================================================
#  図1: 品質 × 多様性
# ============================================================
def fig_quality_diversity(meta, results, ref, out_path: Path):
    dss = meta["datasets"]
    fig, axes = plt.subplots(1, len(dss), figsize=(9.0 * len(dss), 8.0))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, dss):
        res = results[ds]
        base = res["baseline"]

        # --- 参照: 他手法(2026-08-18 の多様性ベンチ) ---
        for algo, (g, d) in ref.get(ds, {}).items():
            if algo == "CIM":
                continue
            ax.scatter([g], [d], s=260, marker="s", facecolor="none",
                       edgecolor="#7f8c8d", linewidth=2.0, zorder=2)
            ax.annotate(algo, (g, d), textcoords="offset points", xytext=(10, 8),
                        fontsize=16, color="#5d6d7e")

        # --- 各実験の掃引 ---
        for exp, info in EXP_INFO.items():
            items = ordered_configs(res, exp)
            if not items:
                continue
            gx = [base["gap_pct_mean"]] + [v["gap_pct_mean"] for _, v in items]
            gy = [base["mean_pairwise"]] + [v["mean_pairwise"] for _, v in items]
            ax.plot(gx, gy, "-o", color=info["color"], markersize=11, linewidth=2.4,
                    label=info["title"], zorder=3)
            for (_, v) in items:
                ax.annotate(v["label"], (v["gap_pct_mean"], v["mean_pairwise"]),
                            textcoords="offset points", xytext=(8, -16),
                            fontsize=13, color=info["color"])

        # --- baseline CIM ---
        ax.scatter([base["gap_pct_mean"]], [base["mean_pairwise"]], s=600, marker="*",
                   color=BASE_COLOR, edgecolor="black", linewidth=1.2, zorder=5,
                   label="CIM baseline")

        ax.set_xscale("log")
        ax.set_xlabel("平均 gap [%](小さいほど高品質)")
        ax.set_ylabel("平均ペア距離(大きいほど多様)")
        ax.set_title(f"{ds}(n={base['n']})")
        ax.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1.5)
        ax.text(ax.get_xlim()[0], 0.5, " ランダム解の水準 0.5", va="bottom",
                fontsize=13, color="#7f8c8d")
        style(ax)

    axes[0].legend(loc="lower right", framealpha=0.92)
    fig.suptitle("CIM の多様性アブレーション — 品質と多様性のトレードオフ平面", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ============================================================
#  図2: ノブごとの応答
# ============================================================
def fig_sweeps(meta, results, out_path: Path):
    dss = meta["datasets"]
    exps = sorted(EXP_INFO)
    fig, axes = plt.subplots(len(exps), len(dss),
                             figsize=(7.5 * len(dss), 5.4 * len(exps)))
    axes = np.atleast_2d(axes)

    for r, exp in enumerate(exps):
        for c, ds in enumerate(dss):
            ax = axes[r, c]
            res = results[ds]
            base = res["baseline"]
            items = ordered_configs(res, exp)
            labels = ["baseline"] + [v["label"] for _, v in items]
            gap = [base["gap_pct_mean"]] + [v["gap_pct_mean"] for _, v in items]
            dist = [base["mean_pairwise"]] + [v["mean_pairwise"] for _, v in items]
            x = np.arange(len(labels))

            ax.plot(x, gap, "-o", color="#c0392b", markersize=10, linewidth=2.4,
                    label="平均 gap [%]")
            ax.set_ylabel("平均 gap [%]", color="#c0392b")
            ax.tick_params(axis="y", colors="#c0392b")
            ax.set_yscale("log")

            ax2 = ax.twinx()
            ax2.plot(x, dist, "-s", color="#2471a3", markersize=10, linewidth=2.4,
                     label="平均ペア距離")
            ax2.set_ylabel("平均ペア距離", color="#2471a3")
            ax2.tick_params(axis="y", colors="#2471a3")
            ax2.set_ylim(0.0, 0.52)
            ax2.axhline(0.5, color="#95a5a6", linestyle=":", linewidth=1.5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=13)
            ax.axvline(0, color="#95a5a6", linestyle="--", linewidth=1.2)
            ax.set_title(f"{EXP_INFO[exp]['title']} — {ds}", fontsize=18)
            style(ax)
            ax2.tick_params(direction="in", which="both")

            if r == 0 and c == 0:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=16)

    fig.suptitle("ノブごとの応答 — 赤=品質(下ほど良い) / 青=多様性(上ほど多様)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ============================================================
#  図3: 線形増幅段の長さ vs 多様性
# ============================================================
def fig_threshold(meta, results, out_path: Path):
    dss = meta["datasets"]
    fig, axes = plt.subplots(1, len(dss), figsize=(8.5 * len(dss), 7.2))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, dss):
        res = results[ds]
        for key, v in res.items():
            exp = v["exp"]
            col = BASE_COLOR if exp == 0 else EXP_INFO[exp]["color"]
            mk = "*" if exp == 0 else "o"
            sz = 620 if exp == 0 else 190
            ax.scatter([v["threshold_frac"] * 100], [v["mean_pairwise"]],
                       s=sz, marker=mk, color=col, edgecolor="black",
                       linewidth=0.9, zorder=4)
            ax.annotate(v["label"], (v["threshold_frac"] * 100, v["mean_pairwise"]),
                        textcoords="offset points", xytext=(9, -6),
                        fontsize=13, color=col)
        ax.set_xlabel("線形増幅段が run に占める割合 [%]")
        ax.set_ylabel("平均ペア距離")
        ax.set_title(f"{ds}")
        style(ax)

    handles = [plt.Line2D([], [], marker="*", markersize=20, linestyle="",
                          color=BASE_COLOR, markeredgecolor="black", label="baseline")]
    handles += [plt.Line2D([], [], marker="o", markersize=12, linestyle="",
                           color=i["color"], markeredgecolor="black", label=i["title"])
                for i in EXP_INFO.values()]
    axes[0].legend(handles=handles, loc="best", fontsize=16, framealpha=0.92)
    fig.suptitle("しきい値到達が遅いほど固有モード競合が長く効く — 多様性との関係", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ============================================================
#  図5: spectral 緩和解との一致度 vs 多様性
# ============================================================
def fig_alignment(rows: list[dict], dss: list[str], out_path: Path):
    """全条件 + 他手法を「固有ベクトル一致度 × 多様性」平面に置く。

    仮説(b)が正しければ、一致度が高い(＝漏斗にはまっている)ほど多様性は低い、
    という右下がりの関係が全条件を貫いて現れるはず。
    """
    fig, axes = plt.subplots(1, len(dss), figsize=(8.5 * len(dss), 7.4))
    axes = np.atleast_1d(axes)

    for ax, ds in zip(axes, dss):
        sub = [r for r in rows if r["dataset"] == ds]
        xs = [float(r["align_mean"]) for r in sub]
        ys = [float(r["mean_pairwise"]) for r in sub]
        if len(xs) >= 3:
            r = float(np.corrcoef(xs, ys)[0, 1])
            k, b = np.polyfit(xs, ys, 1)
            xx = np.linspace(min(xs), max(xs), 20)
            ax.plot(xx, k * xx + b, color="#95a5a6", linestyle="--", linewidth=2.0,
                    label=f"回帰直線(相関 r={r:.2f})")

        for rec in sub:
            x, y = float(rec["align_mean"]), float(rec["mean_pairwise"])
            is_other = rec["exp"] in ("", None)          # 参照 run の他手法
            exp = 0 if is_other else int(rec["exp"])
            if is_other:
                col, mk, sz = "#5d6d7e", "s", 220
            elif exp == 0:
                col, mk, sz = BASE_COLOR, "*", 620
            else:
                col, mk, sz = EXP_INFO[exp]["color"], "o", 190
            ax.scatter([x], [y], s=sz, marker=mk, color=col,
                       edgecolor="black", linewidth=0.9, zorder=4)
            ax.annotate(rec["label"], (x, y), textcoords="offset points",
                        xytext=(9, -6), fontsize=13, color=col)

        ax.set_xlabel("spectral 緩和解との一致度")
        ax.set_ylabel("平均ペア距離")
        ax.set_title(f"{ds}")
        ax.legend(loc="best", fontsize=16)
        style(ax)

    fig.suptitle("固有モードの漏斗にはまるほど多様性が失われる(四角=他手法)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def merge_labels(refined: dict, raw: dict) -> dict:
    """磨き後の要約に、生 run が持つ exp/label/threshold_frac を移植する。"""
    out = {}
    for ds, cfgs in refined.items():
        out[ds] = {}
        for key, v in cfgs.items():
            src = raw.get(ds, {}).get(key, {})
            merged = dict(v)
            for f in ("exp", "label", "threshold_frac", "rounds"):
                if f in src:
                    merged[f] = src[f]
            out[ds][key] = merged
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--refined-suffix", default="_ref20k",
                    help="磨き後の results<suffix>.json の接尾辞。無ければ図4を省略。")
    args = ap.parse_args()

    run_dir = args.run_dir
    meta, results = load(run_dir)
    ref = load_ref(args.ref)

    fig_quality_diversity(meta, results, ref, run_dir / "fig1_quality_diversity.png")
    fig_sweeps(meta, results, run_dir / "fig2_sweeps.png")
    fig_threshold(meta, results, run_dir / "fig3_threshold.png")
    n_fig = 3

    ref_path = run_dir / f"results{args.refined_suffix}.json"
    if ref_path.exists():
        with open(ref_path, encoding="utf-8") as f:
            rp = json.load(f)
        refined = merge_labels(rp["results"], results)
        ref_other = load_ref(args.ref.parent / f"results{args.refined_suffix}.json")
        fig_quality_diversity(meta, refined, ref_other or ref,
                              run_dir / "fig4_refined.png")
        n_fig += 1

    align_csv = run_dir / "spectral_alignment.csv"
    if align_csv.exists():
        import csv as _csv
        with open(align_csv, encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        fig_alignment(rows, meta["datasets"], run_dir / "fig5_alignment.png")
        n_fig += 1

    print(f"saved {n_fig} figures -> {run_dir}")


if __name__ == "__main__":
    main()
