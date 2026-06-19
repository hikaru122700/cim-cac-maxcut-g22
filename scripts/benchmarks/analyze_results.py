"""analyze_results.py — anytime(単体) + combo(ハイブリッド) の結果を集約。

- 並列ポートフォリオ包絡線(全単体を同時並列で走らせ最良を採る)を計算
- 時間分割ポートフォリオ(1コアを K 手法で均等分割)も計算
- 単体 / ポートフォリオ / 最良ハイブリッド を重ねた要約図
- データセット別の要約テーブル(到達カット・gap%・0.5%到達時間)を JSON/MD で出力

入力は anytime_single と combo_hybrid の results.json(複数データセット可)。

実行:
  python scripts/benchmarks/analyze_results.py \
     --anytime results/2026-06-19/anytime_single/vX_.../results.json \
     --combo   results/2026-06-19/combo_hybrid/vA_G22.../results.json ... \
     --out-tag final
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, ALGOS

EXPERIMENT_KIND = "analysis_summary"


def get_out_dir(desc):
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("v") and p.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(p.name.split("_", 1)[0][1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def envelope(curves, time_factor=1):
    """複数の (time,cut) 曲線から、各時刻の最良包絡線を作る。

    curves: list of (times[list], cuts[list])
    time_factor: 各点の時間にかける係数。並列PF(Kコア)は 1(そのまま)、
      時間分割PF(1コアをK分割)は K(各手法は総時間 K 倍かかる→右シフト)。
    返り値: (sorted_times, best_cut_at_or_before)
    """
    pts = []
    for times, cuts in curves:
        for t, c in zip(times, cuts):
            pts.append((t * time_factor, c))
    if not pts:
        return [], []
    pts.sort()
    out_t, out_c = [], []
    best = -1e18
    for t, c in pts:
        best = max(best, c)
        out_t.append(t)
        out_c.append(best)
    return out_t, out_c


def time_to_threshold(times, cuts, thr):
    """cut が thr 以上になる最小時刻。到達しなければ None。"""
    for t, c in sorted(zip(times, cuts)):
        if c >= thr:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anytime", required=True, help="anytime results.json")
    ap.add_argument("--combo", nargs="*", default=[], help="combo results.json(複数)")
    ap.add_argument("--out-tag", default="final")
    args = ap.parse_args()

    with open(args.anytime, "r", encoding="utf-8") as f:
        anytime = json.load(f)["results"]

    combos = {}
    for cp in args.combo:
        with open(cp, "r", encoding="utf-8") as f:
            cj = json.load(f)
        combos[cj["meta"]["dataset"]] = cj

    out_dir = get_out_dir(args.out_tag)
    plt = setup_style()

    summary = {}
    datasets = list(anytime.keys())
    for ds in datasets:
        bks = DATASETS[ds]["bks"]
        ds_algos = anytime[ds]
        # 単体曲線
        single_curves = []
        ds_summary = {"bks": bks, "single": {}, "portfolio": {}, "hybrid": {}}
        for ak, pts in ds_algos.items():
            if not pts:
                continue
            times = [p["time"] for p in pts]
            cuts = [p["cut_max"] for p in pts]
            single_curves.append((times, cuts))
            best = max(cuts)
            t05 = time_to_threshold(times, cuts, bks * 0.995)
            ds_summary["single"][ak] = {
                "best_cut": best, "gap": bks - best,
                "gap_pct": 100.0 * (bks - best) / bks,
                "t_to_0.5pct": t05, "max_time": max(times),
            }
        # 並列ポートフォリオ(各手法フル時間) と 時間分割(K 分割)
        K = len(single_curves)
        env_t, env_c = envelope(single_curves, 1)
        envs_t, envs_c = envelope(single_curves, K)
        ds_summary["portfolio"]["parallel_best"] = max(env_c) if env_c else None
        ds_summary["portfolio"]["K"] = K

        # ハイブリッド(combo)曲線
        hyb_curves = {}
        if ds in combos:
            for name, pts in combos[ds]["runs"].items():
                if "cold" in name:
                    continue
                t = [p["total_time"] for p in pts]
                c = [p["cut_max"] for p in pts]
                hyb_curves[name] = (t, c)
                ds_summary["hybrid"][name] = {
                    "best_cut": max(c), "gap": bks - max(c),
                    "t_to_0.5pct": time_to_threshold(t, c, bks * 0.995),
                }

        # --- 図: 単体 + ポートフォリオ包絡 + 最良ハイブリッド ---
        fig, ax = plt.subplots(figsize=(8.6, 6.0), dpi=140)
        for ak, pts in ds_algos.items():
            if not pts:
                continue
            ax.plot([p["time"] for p in pts], [p["cut_max"] for p in pts],
                    "-", color=ALGOS[ak]["color"], lw=1.2, alpha=0.55,
                    label=ALGOS[ak]["label"])
        if env_t:
            ax.plot(env_t, env_c, "-", color="black", lw=2.6,
                    label="並列ポートフォリオ包絡")
        if envs_t:
            ax.plot(envs_t, envs_c, "--", color="black", lw=1.4, alpha=0.7,
                    label=f"時間分割PF(1コア{K}分割)")
        # 最良ハイブリッド(到達カット最大のもの)
        if hyb_curves:
            best_name = max(hyb_curves, key=lambda k: max(hyb_curves[k][1]))
            ht, hc = hyb_curves[best_name]
            ax.plot(ht, hc, "-D", color="#c0392b", lw=2.4, ms=6,
                    label=f"最良ハイブリッド({best_name})")
        ax.axhline(bks, color="k", ls=":", lw=1.3, label=f"BKS={bks}")
        ax.set_xscale("log")
        ax.set_xlabel("実時間 [秒]（log）")
        ax.set_ylabel("最大カット値")
        ax.set_title(f"{ds}: 単体 vs ポートフォリオ vs ハイブリッド")
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(out_dir / f"{ds}_summary.png", bbox_inches="tight")
        plt.close(fig)

        summary[ds] = ds_summary

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    # --- 要約 MD テーブル ---
    lines = ["# 結果サマリ\n"]
    for ds, s in summary.items():
        bks = s["bks"]
        lines.append(f"\n## {ds} (BKS={bks})\n")
        lines.append("| 手法 | 到達カット | gap | gap% | 0.5%到達時間[s] |")
        lines.append("|---|---|---|---|---|")
        for ak, v in sorted(s["single"].items(), key=lambda x: x[1]["gap"]):
            t05 = f"{v['t_to_0.5pct']:.2f}" if v["t_to_0.5pct"] else "—"
            lines.append(f"| {ALGOS[ak]['label']} | {int(v['best_cut'])} | "
                         f"{v['gap']:.0f} | {v['gap_pct']:.2f}% | {t05} |")
        pf = s["portfolio"].get("parallel_best")
        if pf:
            lines.append(f"| **並列PF** | **{int(pf)}** | {bks - pf:.0f} | "
                         f"{100.0 * (bks - pf) / bks:.2f}% | — |")
        for name, v in sorted(s["hybrid"].items(), key=lambda x: x[1]["gap"]):
            t05 = f"{v['t_to_0.5pct']:.2f}" if v["t_to_0.5pct"] else "—"
            lines.append(f"| {name} (warm) | {int(v['best_cut'])} | {v['gap']:.0f} | "
                         f"{100.0 * v['gap'] / bks:.2f}% | {t05} |")
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nsaved → {out_dir}")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
