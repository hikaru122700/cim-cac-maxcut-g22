"""plot_qa_v2_figs.py — 質問解説 v2 用の図表生成。

anytime_single の全手法 tuned 実行(results.json)から:
  (Q6) 各手法の「ベスト到達時間」「gap 0.5% 到達時間」を抜き出して実行時間表を作る。
  (Q7) gap%(対数) vs 実時間(対数) を 4 データセット 2x2 で描く。

入力 : results/2026-06-19/anytime_single/v7_G22_K2000_G55_G70_nt16_alltuned/results.json
出力 : <paper>/qa_v2/gap_vs_time_v1.png, <paper>/qa_v2/runtime_table_v1.json
使い方: python scripts/plotting/plot_qa_v2_figs.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- 共通プロットスタイル(プロジェクト規約: 目盛り内向き, 日本語 Yu Gothic) ---
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/2026-06-19/anytime_single/v7_G22_K2000_G55_G70_nt16_alltuned/results.json"
OUT = ROOT / "results/2026-06-19/maxcut_algo_combo/v1_full/paper/qa_v2"
OUT.mkdir(parents=True, exist_ok=True)

BKS = {"G22": 13359, "K2000": 33337, "G55": 10299, "G70": 9591}
COLORS = {
    "CIM": "#e74c3c", "CAC": "#e67e22", "SA": "#2980b9",
    "SB": "#16a085", "PT": "#8e44ad", "GA": "#d81b9e",
}
LABEL = {"CIM": "CIM", "CAC": "CAC", "SA": "SA", "SB": "dSB", "PT": "PT-ICM", "GA": "GA"}
ALGOS = ["CIM", "CAC", "SA", "SB", "PT", "GA"]
TARGET_PCT = 0.5  # 共通品質目標 gap ≤ 0.5%


def gap_pct(cut, bks):
    return 100.0 * (bks - cut) / bks


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))["results"]

    # ============ Q7: gap%(log) vs 時間(log) 2x2 ============
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for ax, ds in zip(axes.flat, ["G22", "K2000", "G55", "G70"]):
        bks = BKS[ds]
        for algo in ALGOS:
            pts = data[ds].get(algo, [])
            if not pts:
                continue
            # 各予算点で best(cut_max)の gap% を、累積ベスト(単調)で描く
            xs, ys = [], []
            best = -1
            for p in pts:
                best = max(best, p["cut_max"])
                xs.append(p["time"])
                ys.append(max(gap_pct(best, bks), 1e-3))  # 0 は log 不可なので下限
            ax.plot(xs, ys, "-o", ms=4, lw=1.6, color=COLORS[algo], label=LABEL[algo])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{ds}  (BKS={bks})", fontsize=12)
        ax.set_xlabel("実時間 [秒]  (16本同時実行 best-of-16, 4スレッド)")
        ax.set_ylabel("BKS との相対 gap [%]")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.axhline(TARGET_PCT, color="gray", ls="--", lw=1.0, alpha=0.7)
    axes.flat[0].legend(fontsize=9, ncol=2, framealpha=0.9)
    fig.suptitle("各手法の anytime 性能: 相対 gap [%] 対 実時間（両対数）", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "gap_vs_time_v1.png", dpi=130)
    plt.close(fig)

    # ============ Q6: 実行時間表 ============
    table = {}
    for ds in ["G22", "K2000", "G55", "G70"]:
        bks = BKS[ds]
        table[ds] = {}
        for algo in ALGOS:
            pts = data[ds].get(algo, [])
            if not pts:
                continue
            # ベスト到達(最良 cut_max を最初に達成した予算点の時間)
            best = max(p["cut_max"] for p in pts)
            t_best = None
            run = -1
            for p in pts:
                run = max(run, p["cut_max"])
                if run >= best and t_best is None:
                    t_best = p["time"]
            # gap ≤ 0.5% を最初に達成した時間
            t_target = None
            run = -1
            for p in pts:
                run = max(run, p["cut_max"])
                if gap_pct(run, bks) <= TARGET_PCT:
                    t_target = p["time"]
                    break
            table[ds][algo] = {
                "best_cut": int(best),
                "gap": int(bks - best),
                "gap_pct": round(gap_pct(best, bks), 3),
                "time_to_best_sec": round(t_best, 2) if t_best is not None else None,
                "time_to_0p5pct_sec": round(t_target, 2) if t_target is not None else None,
            }
    (OUT / "runtime_table_v1.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown 表を標準出力(コピペ用)
    for ds in ["G22", "K2000", "G55", "G70"]:
        print(f"\n### {ds} (BKS={BKS[ds]})")
        print("| 手法 | gap | gap% | ベスト到達 [s] | gap≤0.5% 到達 [s] |")
        print("|---|---:|---:|---:|---:|")
        for algo in ALGOS:
            if algo not in table[ds]:
                continue
            r = table[ds][algo]
            tb = r["time_to_best_sec"]
            tt = r["time_to_0p5pct_sec"]
            print(f"| {LABEL[algo]} | {r['gap']} | {r['gap_pct']} | "
                  f"{tb} | {tt if tt is not None else '未達'} |")
    print("\n[OK] wrote", OUT / "gap_vs_time_v1.png")
    print("[OK] wrote", OUT / "runtime_table_v1.json")


if __name__ == "__main__":
    main()
