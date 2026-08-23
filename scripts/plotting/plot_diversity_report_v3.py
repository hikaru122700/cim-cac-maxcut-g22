"""plot_diversity_report_v3.py — 多様性評価の要約図（図番号つき・補助線を整理）。

v2 からの変更:
  - 各図のタイトルに **図番号（図1〜図8）** を入れ、ファイル名も本文の並び順に合わせた
  - 4 象限マップの **中央値を示す灰色の破線を削除**（6 手法の中央値という恣意的な基準だったため）
  - 残した補助線は意味のある 2 本だけ:
      * 距離 0.5（= 2 つの解が無相関になる上限）
      * 偏り 0.9（= 「毎回同じ側」とみなすしきい値）

手法ごとにマーカー形状・線種・ハッチを固定する方針は v2 と同じ。
出力は `fig{N}_*_v3.png`（過去の図は上書きしない）。

使い方:
    python scripts/plotting/plot_diversity_report_v3.py <run_dir> [--suffix _v3]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import ALGOS
from scripts.benchmarks.diversity_metrics import (
    to_binary, distance_matrix, cross_distance_matrix, align_to_reference,
    classical_mds,
)

ORDER = ["CIM", "CAC", "SA", "SB", "PT", "GA"]

# 手法ごとの「色に依存しない」識別子: (マーカー, 線種, ハッチ)
MARK = {
    "CIM": ("o", "-",                    "//"),
    "CAC": ("s", (0, (5, 2)),            "\\\\"),
    "SA":  ("^", (0, (1, 1)),            "xx"),
    "SB":  ("D", (0, (3, 1, 1, 1)),      ".."),
    "PT":  ("v", (0, (6, 2, 1, 2)),      "++"),
    "GA":  ("*", (0, (2, 1)),            "oo"),
}
HILITE = "CIM"          # 注目手法（太線で強調）

# ------------------------------------------------------------
#  文字サイズ（可読性優先・全図共通）
#  プロジェクト共通ルール: ラベル/タイトル/凡例/カラム名 >= 20pt、
#  グラフ上の数値 >= 12pt。フォントを大きくしたぶん figsize も広げる。
# ------------------------------------------------------------
F_LABEL = 22     # 軸タイトル
F_TITLE = 24     # パネルタイトル
F_SUP = 26       # 図全体のタイトル
F_ALGO = 20      # 手法名・カテゴリ名（カラム名）
F_ALGO_BIG = 23  # 注目手法の名前
F_TICK = 18      # 目盛りの数値
F_QUAD = 19      # 象限などの補助ラベル
F_NUM = 16       # グラフ上に描く数値
F_NOTE = 16      # 補助的な注記


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=F_TICK)
    ax.grid(alpha=0.25)


def lw_of(a):
    return 3.2 if a == HILITE else 1.9


def ms_of(a):
    return 11 if a == HILITE else 7


def spread(ys, min_gap):
    """ラベルの y 座標が重ならないように押し広げる（順序は保つ）。"""
    idx = np.argsort(ys)
    out = np.asarray(ys, dtype=float).copy()
    for k in range(1, len(idx)):
        i, j = idx[k - 1], idx[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    return out


def legend_strip(fig, algos, y=0.005):
    """図の下端に「色 + 形」の凡例帯を置く。"""
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ALGOS[a]["color"], marker=MARK[a][0],
                      ls=MARK[a][1], lw=lw_of(a), ms=ms_of(a),
                      markeredgecolor="k", markeredgewidth=0.5,
                      label=ALGOS[a]["label"]) for a in algos]
    fig.legend(handles=handles, loc="lower center", ncol=len(algos),
               fontsize=F_ALGO, frameon=False, bbox_to_anchor=(0.5, y))


def draw_quadrant(ax, ds, R, f_tick):
    """図1 の 1 データセット分を ax に描く。

    f_tick は目盛り数値の大きさ。1 枚 1 グラフのときは大きく取る。
    対数軸の目盛りは 10^0 のような指数表記だと小さくて読みにくいので、
    0.1 / 1 / 10 の素直な十進表記にし、2・3・5 倍の補助目盛りにも数値を付ける。
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    algos = [a for a in ORDER if a in R[ds]]
    gx = np.array([R[ds][a]["gap_pct_mean"] for a in algos])
    dy = np.array([R[ds][a]["mean_pairwise"] for a in algos])

    ax.set_xscale("log")
    lo, hi = gx.min() / 2.6, gx.max() * 3.2
    ax.set_xlim(lo, hi)
    pad = max(0.040, (dy.max() - dy.min()) * 0.52)
    ax.set_ylim(max(0.0, dy.min() - pad), min(0.54, dy.max() + pad))

    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 3.0, 5.0)))
    ax.xaxis.set_minor_formatter(fmt)

    # 象限ラベル
    for xx, yy, txt, ha, va in [
        (lo * 1.25, ax.get_ylim()[1], "良い × 多様", "left", "top"),
        (hi / 1.25, ax.get_ylim()[1], "悪い × 多様", "right", "top"),
        (lo * 1.25, ax.get_ylim()[0], "良い × 均質", "left", "bottom"),
        (hi / 1.25, ax.get_ylim()[0], "悪い × 均質", "right", "bottom"),
    ]:
        ax.text(xx, yy, txt, fontsize=F_QUAD, color="#999", ha=ha, va=va)

    # ラベルの上下は「回帰直線のどちら側にいるか」で決める。
    # x 順の交互振りだと、x が近く y が違う 2 点のラベルが中間でぶつかる。
    lx = np.log10(gx)
    slope, icpt = np.polyfit(lx, dy, 1)
    resid = dy - (slope * lx + icpt)
    side = {a: (1 if r >= 0 else -1) for a, r in zip(algos, resid)}
    # それでも同じ側で近接する組は、後から来た方をさらに押し出す
    by_x = sorted(algos, key=lambda a: R[ds][a]["gap_pct_mean"])
    extra = {a: 0.0 for a in algos}
    xr = np.log10(hi) - np.log10(lo)
    yr = ax.get_ylim()[1] - ax.get_ylim()[0]
    for i in range(1, len(by_x)):
        a, b = by_x[i - 1], by_x[i]
        if side[a] != side[b]:
            continue
        dxn = abs(np.log10(R[ds][b]["gap_pct_mean"])
                  - np.log10(R[ds][a]["gap_pct_mean"])) / xr
        dyn = abs(R[ds][b]["mean_pairwise"] - R[ds][a]["mean_pairwise"]) / yr
        if dxn < 0.16 and dyn < 0.16:
            extra[b] = extra[a] + 26.0

    for a in algos:
        x, y = R[ds][a]["gap_pct_mean"], R[ds][a]["mean_pairwise"]
        big = a == HILITE
        ax.scatter([x], [y], s=760 if big else 400, marker=MARK[a][0],
                   color=ALGOS[a]["color"], edgecolors="k",
                   lw=2.6 if big else 1.1, zorder=3)
        off = ((30 if big else 24) + extra[a]) * side[a] - (14 if side[a] < 0 else 0)
        ax.annotate(ALGOS[a]["label"], (x, y), textcoords="offset points",
                    xytext=(0, off), ha="center",
                    va="bottom" if side[a] > 0 else "top",
                    fontsize=F_ALGO_BIG if big else F_ALGO,
                    fontweight="bold" if big else "normal",
                    bbox=dict(fc="white", ec="none", alpha=0.8,
                              boxstyle="round,pad=0.18"))

    ax.set_xlabel("平均 gap [%]（左ほど高品質・対数軸）", fontsize=F_LABEL)
    ax.set_ylabel("平均ペアワイズ距離（上ほど多様）", fontsize=F_LABEL)
    style(ax)
    # style() の一括指定より後で、主・補助の両方の目盛り数値を大きくする
    ax.tick_params(axis="both", which="both", labelsize=f_tick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--suffix", default="_v3")
    ap.add_argument("--only", nargs="+", type=int, default=None,
                    help="この図番号だけを書き出す(省略時は全図)。既存図の上書きを避けたいときに使う。")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    sfx = args.suffix
    plt = setup_plt()
    want = set(args.only) if args.only else None

    def save(fig, num: int, name: str):
        """--only で選ばれた図だけを保存する。"""
        if want is not None and num not in want:
            plt.close(fig)
            return
        fig.savefig(run_dir / f"{name}{sfx}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {name}{sfx}.png", flush=True)

    raw = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ref = json.loads((run_dir / "results_ref20k.json").read_text(encoding="utf-8"))
    met = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    npz = np.load(run_dir / "signs.npz")
    meta, R, RF = raw["meta"], raw["results"], ref["results"]
    datasets = [d for d in meta["datasets"] if d in R and R[d]]

    # ============================================================
    # fig1: 4 象限マップ（良い/悪い × 均質/多様）
    #   1 枚に 4 パネル詰めると対数軸の目盛りが読めないので、
    #   データセットごとに 1 枚ずつ大きく出す。
    # ============================================================
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(13.0, 10.0), dpi=140)
        draw_quadrant(ax, ds, R, f_tick=26)
        ax.set_title(f"図1  手法の位置づけ：解は良いか／解はばらけるか  —  {ds}",
                     fontsize=F_SUP)
        fig.tight_layout()
        save(fig, 1, f"fig1_quadrant_{ds}")

    # ============================================================
    # fig1: 俯瞰（棒 + ハッチ + マーカー）
    # ============================================================
    fig, axes = plt.subplots(len(datasets), 2, figsize=(20.0, 5.6 * len(datasets)),
                             dpi=140, squeeze=False,
                             gridspec_kw={"width_ratios": [1.5, 1.0]})
    for k, ds in enumerate(datasets):
        algos = [a for a in ORDER if a in R[ds]]
        dist = np.array([R[ds][a]["mean_pairwise"] for a in algos])
        gap = np.array([R[ds][a]["gap_pct_mean"] for a in algos])
        idx = np.argsort(dist)
        y = np.arange(len(algos))
        srt = [algos[i] for i in idx]

        ax = axes[k][0]
        for yy, a in zip(y, srt):
            v = R[ds][a]["mean_pairwise"]
            ax.barh(yy, v, color=ALGOS[a]["color"], alpha=0.85, edgecolor="k",
                    lw=0.6, hatch=MARK[a][2])
            ax.plot([v], [yy], marker=MARK[a][0], color=ALGOS[a]["color"],
                    ms=ms_of(a) + 2, markeredgecolor="k", markeredgewidth=0.8,
                    clip_on=False, zorder=4)
            ax.text(v + 0.022, yy, f"{v:.3f}", va="center", fontsize=F_NUM,
                    fontweight="bold" if a == HILITE else "normal")
        ax.set_yticks(y)
        ax.set_yticklabels([ALGOS[a]["label"] for a in srt], fontsize=F_ALGO)
        for lb, a in zip(ax.get_yticklabels(), srt):
            if a == HILITE:
                lb.set_fontweight("bold")
        ax.axvline(0.5, color="gray", ls=":", lw=1.4)
        ax.set_xlim(0, 0.70)
        ax.set_xlabel("平均ペアワイズ距離（右ほど多様）", fontsize=F_LABEL)
        ax.set_title(f"{ds}: 出力解の散らばり", fontsize=F_TITLE)
        ax.text(0.5, 0.04, " 無相関の水準", color="gray", fontsize=F_NOTE,
                transform=ax.get_xaxis_transform(), va="bottom")
        style(ax)

        ax = axes[k][1]
        for yy, a in zip(y, srt):
            v = R[ds][a]["gap_pct_mean"]
            ax.barh(yy, v, color=ALGOS[a]["color"], alpha=0.4, edgecolor="k",
                    lw=0.6, hatch=MARK[a][2])
            ax.text(v + gap.max() * 0.04, yy, f"{v:.2f}", va="center", fontsize=F_NUM,
                    fontweight="bold" if a == HILITE else "normal")
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlim(0, gap.max() * 1.55)
        ax.set_xlabel("平均 gap [%]（左ほど高品質）", fontsize=F_LABEL)
        ax.set_title(f"{ds}: 解の品質", fontsize=F_TITLE)
        style(ax)
    fig.suptitle("図2  同じ実時間で 32 回解いたときの「散らばり」と「良さ」", fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=2.8)
    save(fig, 2, "fig2_overview")

    # ============================================================
    # fig2: 磨く前後（直接ラベル）
    # ============================================================
    fig, axes = plt.subplots(1, len(datasets), figsize=(7.2 * len(datasets), 8.8),
                             dpi=140, squeeze=False)
    for k, ds in enumerate(datasets):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in RF[ds]]
        ys = np.array([RF[ds][a]["mean_pairwise"] for a in algos])
        lab_y = spread(ys, 0.042)
        lab_y = lab_y - (lab_y.mean() - ys.mean())   # 上方向への偏りを戻す
        for a, ly in zip(algos, lab_y):
            b, af = RF[ds][a]["mean_pairwise_before"], RF[ds][a]["mean_pairwise"]
            ax.plot([0, 1], [b, af], ls=MARK[a][1], marker=MARK[a][0],
                    color=ALGOS[a]["color"], lw=lw_of(a) * 1.5,
                    ms=ms_of(a) * 1.5, markeredgecolor="k", markeredgewidth=0.7)
            ax.annotate(ALGOS[a]["label"], (1.12, ly), fontsize=F_ALGO,
                        color=ALGOS[a]["color"], va="center",
                        fontweight="bold" if a == HILITE else "normal")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["磨く前", "磨いた後"], fontsize=F_ALGO)
        ax.set_xlim(-0.28, 2.85)
        ax.set_ylim(0, 0.60)
        ax.set_ylabel("平均ペアワイズ距離" if k == 0 else "", fontsize=F_LABEL)
        ax.set_title(ds, fontsize=F_TITLE)
        style(ax)
    fig.suptitle("図4  共通の Tabu Search で磨いても、解の散らばりはほぼ変わらない",
                 fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 4, "fig4_refine")

    # ============================================================
    # fig3: big valley（疎 vs 密）
    # ============================================================
    pair = [d for d in ["G22", "K2000"] if d in datasets]
    fig, axes = plt.subplots(1, len(pair), figsize=(11.0 * len(pair), 9.0), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(pair):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in R[ds]]
        allS = np.concatenate([to_binary(npz[f"{ds}__{a}_signs"]) for a in algos])
        allC = np.concatenate([npz[f"{ds}__{a}_cuts"] for a in algos])
        owner = np.concatenate([[i] * 32 for i in range(len(algos))])
        bks = float(meta["bks"][ds])
        dref = cross_distance_matrix(allS, allS[int(np.argmax(allC))][None, :]).ravel()
        gaps = (bks - allC) / bks * 100.0
        # 極端な外れ値（未収束の CAC 等）で軸が潰れないよう上限を切る
        ytop = float(np.percentile(gaps, 90)) * 1.30 + 1e-3
        n_out = int((gaps > ytop).sum())
        cents = []
        for i, a in enumerate(algos):
            sel = owner == i
            big = a == HILITE
            ax.scatter(dref[sel], gaps[sel], s=190 if big else 95,
                       marker=MARK[a][0], color=ALGOS[a]["color"],
                       alpha=0.9 if big else 0.75, edgecolors="k",
                       lw=1.0 if big else 0.4, zorder=3 if big else 2)
            cents.append((a, float(dref[sel].mean()),
                          float(np.median(gaps[sel]))))
        # 重心ラベルを縦に押し広げ、細い引出線で結ぶ
        ys = spread([min(c[2], ytop * 0.88) for c in cents], ytop * 0.130)
        for (a, cx, cy), ly in zip(cents, ys):
            ax.annotate(ALGOS[a]["label"], xy=(cx, min(cy, ytop * 0.97)),
                        xytext=(cx, ly), fontsize=F_ALGO, color="k",
                        ha="center", va="center", zorder=6,
                        bbox=dict(fc="white", ec=ALGOS[a]["color"], lw=1.5,
                                  boxstyle="round,pad=0.22", alpha=0.95),
                        arrowprops=dict(arrowstyle="-", lw=1.0,
                                        color=ALGOS[a]["color"]))
        fdc = met["per_dataset"][ds]["fdc"].get("ALL", float("nan"))
        ax.set_xlabel("見つかった最良解からの距離", fontsize=F_LABEL)
        ax.set_ylabel("BKS からの gap [%]" if k == 0 else "", fontsize=F_LABEL)
        ax.set_xlim(-0.02, 0.52)
        ax.set_ylim(-ytop * 0.04, ytop)
        if n_out:
            ax.text(0.02, 0.97, f"※ 軸外に {n_out} 点（最大 {gaps.max():.1f}%）",
                    transform=ax.transAxes, fontsize=F_NOTE, color="#a33", va="top")
        kind = "疎グラフ" if ds != "K2000" else "密グラフ"
        note = "良い解は 1 か所に集中" if fdc > 0.6 else "良い解が広く散在"
        ax.set_title(f"{ds}（{kind}）  相関 = {fdc:.2f} → {note}", fontsize=F_TITLE)
        style(ax)
    fig.suptitle("図5  良い解は 1 か所に集まるのか、あちこちに散るのか（枠は各手法の重心）",
                 fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 5, "fig5_bigvalley")

    # ============================================================
    # fig4: 骨格プロファイル（線種 + 直接ラベル）
    # ============================================================
    show = [d for d in ["G22", "K2000"] if d in datasets]
    fig, axes = plt.subplots(1, len(show), figsize=(11.0 * len(show), 8.6), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(show):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in R[ds]]
        curves = {}
        for a in algos:
            S = to_binary(npz[f"{ds}__{a}_signs"])
            c = npz[f"{ds}__{a}_cuts"]
            A = align_to_reference(S, S[int(np.argmax(c))])
            m = np.sort(np.abs(2.0 * A.mean(axis=0) - 1.0))[::-1]
            curves[a] = m
            xs = np.arange(m.size) / m.size * 100.0
            ax.plot(xs, m, ls=MARK[a][1], lw=lw_of(a) * 1.5,
                    color=ALGOS[a]["color"],
                    solid_capstyle="round")
        # 50% 地点に直接ラベル
        vals = np.array([curves[a][curves[a].size // 2] for a in algos])
        lab_y = spread(vals, 0.082)
        for a, ly in zip(algos, lab_y):
            ax.annotate(ALGOS[a]["label"], (52, ly), fontsize=F_ALGO,
                        color="k", va="center", ha="left",
                        bbox=dict(fc="white", ec=ALGOS[a]["color"], lw=1.4,
                                  boxstyle="round,pad=0.2", alpha=0.92))
        ax.axhline(0.9, color="gray", ls=":", lw=1.3)
        ax.text(99, 0.925, "「毎回同じ側」とみなす水準 ", color="gray",
                fontsize=F_NOTE, ha="right")
        ax.set_xlabel("頂点（安定な順に並べた百分位）[%]", fontsize=F_LABEL)
        ax.set_ylabel("32 試行での偏り $|m_i|$" if k == 0 else "", fontsize=F_LABEL)
        ax.set_ylim(0, 1.08); ax.set_xlim(0, 100)
        ax.set_title(ds, fontsize=F_TITLE)
        style(ax)
    fig.suptitle("図3  どれだけの頂点が「毎回同じ側」に決まるか（曲線が高いほど骨格が太い）",
                 fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 3, "fig3_backbone")

    # ============================================================
    # fig5: 上位 25% の占有（ハッチ + 0 本の明示）
    # ============================================================
    fig, ax = plt.subplots(figsize=(16.5, 8.6), dpi=140)
    xs = np.arange(len(datasets))
    bottom = np.zeros(len(datasets))
    zero_algos = {ds: [] for ds in datasets}
    for a in ORDER:
        vals = []
        for ds in datasets:
            v = met["per_dataset"][ds]["pooled_top25"].get(a, {"n_top": 0})["n_top"]
            vals.append(v)
            if v == 0:
                zero_algos[ds].append(ALGOS[a]["label"])
        vals = np.asarray(vals, dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=ALGOS[a]["color"], alpha=0.85,
               edgecolor="k", lw=0.7, hatch=MARK[a][2], label=ALGOS[a]["label"])
        for x, v, b in zip(xs, vals, bottom):
            if v >= 3:
                ax.text(x, b + v / 2, f"{ALGOS[a]['label']}  {int(v)} 本",
                        ha="center", va="center", fontsize=F_ALGO,
                        bbox=dict(fc="white", ec="none", alpha=0.9,
                                  boxstyle="round,pad=0.3"))
        bottom += vals
    for x, ds in zip(xs, datasets):
        ax.text(x, bottom[x] + 2.0, "0 本: " + " / ".join(zero_algos[ds]),
                ha="center", fontsize=F_NOTE, color="#a33")
    ax.set_xticks(xs); ax.set_xticklabels(datasets, fontsize=F_ALGO)
    ax.set_xlim(-0.68, len(datasets) - 0.32)
    ax.set_ylim(0, bottom.max() + 13)
    ax.set_ylabel("上位 25% に入った解の本数（192 本中 約 48 本）", fontsize=F_LABEL)
    ax.set_title("図6  良い解を出しているのは誰か（全手法プールの上位 25%）",
                 fontsize=F_SUP)
    style(ax)
    fig.tight_layout()
    save(fig, 6, "fig6_topshare")

    # ============================================================
    # fig6: 手法間距離ヒートマップ（近い＝濃い、しきい線つき）
    # ============================================================
    hm = [d for d in ["G22", "K2000"] if d in datasets]
    fig, axes = plt.subplots(1, len(hm), figsize=(11.0 * len(hm), 9.2), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(hm):
        ax = axes[0][k]
        algos = met["per_dataset"][ds]["algos"]
        M = np.array(met["per_dataset"][ds]["cross_mean"])
        order = [algos.index(a) for a in ORDER if a in algos]
        M = M[np.ix_(order, order)]
        names = [ALGOS[algos[i]]["label"] for i in order]
        im = ax.imshow(M, cmap="magma_r", vmin=0.1, vmax=0.5)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=F_NUM + 2, color="w" if M[i, j] > 0.38 else "k",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=F_ALGO)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=F_ALGO)
        for lb in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            if lb.get_text() == ALGOS[HILITE]["label"]:
                lb.set_fontweight("bold")
        ax.set_title(f"{ds}: 手法間の解の隔たり（対角=手法内）", fontsize=F_TITLE)
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label("平均距離（濃いほど遠い）", fontsize=F_LABEL)
        cb.ax.tick_params(labelsize=F_TICK)
    fig.suptitle("図7  どの手法どうしが同じ場所を見ているか", fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, 7, "fig7_cross")

    # ============================================================
    # fig7: MDS（マーカー形状つき・最良解を注記）
    # ============================================================
    fig, axes = plt.subplots(1, len(hm), figsize=(11.0 * len(hm), 9.6), dpi=140,
                             squeeze=False)
    for k, ds in enumerate(hm):
        ax = axes[0][k]
        algos = [a for a in ORDER if a in R[ds]]
        allS = np.concatenate([to_binary(npz[f"{ds}__{a}_signs"]) for a in algos])
        allC = np.concatenate([npz[f"{ds}__{a}_cuts"] for a in algos])
        owner = np.concatenate([[i] * 32 for i in range(len(algos))])
        emb = classical_mds(distance_matrix(allS), 2)
        bks = float(meta["bks"][ds])
        gaps = (bks - allC) / bks * 100.0
        lo, hi = gaps.min(), gaps.max()
        for i, a in enumerate(algos):
            sel = owner == i
            sz = 70 + 340 * (1.0 - (gaps[sel] - lo) / max(hi - lo, 1e-9))
            ax.scatter(emb[sel, 0], emb[sel, 1], s=sz, marker=MARK[a][0],
                       color=ALGOS[a]["color"], alpha=0.8, edgecolors="k",
                       lw=1.0 if a == HILITE else 0.4)
        b = int(np.argmax(allC))
        ax.annotate("最良解", (emb[b, 0], emb[b, 1]), textcoords="offset points",
                    xytext=(34, 34), fontsize=F_ALGO,
                    arrowprops=dict(arrowstyle="->", lw=2.0))
        ax.set_xlabel("MDS 第 1 軸", fontsize=F_LABEL)
        ax.set_ylabel("MDS 第 2 軸" if k == 0 else "", fontsize=F_LABEL)
        ax.set_title(f"{ds}（点が大きいほど良い解）", fontsize=F_TITLE)
        style(ax)
    legend_strip(fig, [a for a in ORDER if a in R[hm[0]]], y=0.0)
    fig.suptitle("図8  解空間のどこに着地したか（近い点ほど似た解）", fontsize=F_SUP)
    fig.tight_layout(rect=[0, 0.09, 1, 0.945])
    save(fig, 8, "fig8_mds")

    print(f"fig1..fig8{sfx} -> {run_dir}")


if __name__ == "__main__":
    main()
