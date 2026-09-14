"""Reanalyse saved CIM solutions for a discussion of diversity (no solver runs).

Run from the project root. Figures distinguish CIM solution-set diversity from
within-GA population diversity. Matching uses cut values only, never distances.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "cim_diversity_evidence"
DEFAULT_SOURCE = Path("results/2026-08-23/cim_diversity_ablation/v3_G22_G55_G70_nt32_ex1234_main")
COLORS = {"baseline": "#405569", "noise1000x": "#C34436"}
LABELS = {"baseline": "基準CIM", "noise1000x": "ノイズ振幅1,000倍"}
CONDITIONS = tuple(COLORS)


def distance_matrix(signs):
    spins = signs.astype(np.float64) * 2 - 1
    h = (signs.shape[1] - spins @ spins.T) / 2
    d = np.minimum(h, signs.shape[1] - h) / signs.shape[1]
    np.fill_diagonal(d, 0)
    return d


def mean_distance(d):
    n = len(d)
    if n < 2:
        raise ValueError("At least two solutions are needed")
    return float(d.sum() / (n * (n - 1)))


def jackknife_halfwidth(d):
    """Delete-one-trial jackknife; pairs are NOT independent observations."""
    n = len(d)
    leave_one = (d.sum() - 2 * d.sum(axis=1)) / ((n - 1) * (n - 2))
    se = np.sqrt((n - 1) / n * np.sum((leave_one - leave_one.mean()) ** 2))
    return float(student_t.ppf(.975, n - 1) * se)


def exact_match(cuts_a, cuts_b, rng=None):
    """Match maximum available counts at each EXACT integer cut value."""
    out_a, out_b = [], []
    for cut in np.intersect1d(cuts_a, cuts_b):
        ia, ib = np.flatnonzero(cuts_a == cut), np.flatnonzero(cuts_b == cut)
        count = min(len(ia), len(ib))
        if rng is not None:
            ia, ib = rng.permutation(ia), rng.permutation(ib)
        out_a.extend(ia[:count])
        out_b.extend(ib[:count])
    ia, ib = np.array(out_a, dtype=int), np.array(out_b, dtype=int)
    if len(ia) < 3 or not np.array_equal(cuts_a[ia], cuts_b[ib]):
        raise ValueError("Insufficient or invalid exact matches")
    return ia, ib


def new_output(args):
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(m.group(1)) for p in root.iterdir()
                if (m := re.match(r"v(\d+)_", p.name))]
    desc = "_".join([*args.datasets, "matched", f"r{args.selection_repeats}", args.tag]).strip("_")
    if not re.fullmatch(r"[A-Za-z0-9_]+", desc):
        raise ValueError("Use ASCII alphanumeric characters and underscores for the tag")
    out = root / f"v{max(versions, default=0) + 1}_{desc}"
    out.mkdir(exist_ok=False)
    return out


def style():
    plt.rcParams.update({
        "font.family": "Yu Gothic", "axes.unicode_minus": False,
        "font.size": 12, "axes.titlesize": 16, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "svg.fonttype": "path", "savefig.facecolor": "white",
    })


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=220)
    plt.close(fig)


def finish_axes(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", color="#d8dde2", alpha=.7, linewidth=.7)
    ax.set_axisbelow(True)


def full_figure(data, out):
    fig, axs = plt.subplots(2, len(data), figsize=(15, 8.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.15, top=.84, wspace=.32, hspace=.43)
    fig.suptitle("CIMの制御を変えると、品質と多様性はどう変わるか", y=.97, fontsize=21)
    fig.text(.5, .91, "保存済みの各条件32試行を再解析 ／ 共通Tabu Search 20,000反復後", ha="center", color="#53616e")
    rng = np.random.default_rng(734)
    for col, (ds, a) in enumerate(data.items()):
        top, bottom = axs[:, col]
        for i, cond in enumerate(CONDITIONS):
            cuts = a[cond]["cuts"]
            gaps = 100 * (a["bks"] - cuts) / a["bks"]
            dist = a[cond]["distance"]
            top.scatter(i + rng.uniform(-.12, .12, len(cuts)), gaps,
                        s=20, alpha=.42, color=COLORS[cond], edgecolors="none")
            ci = student_t.ppf(.975, len(cuts) - 1) * gaps.std(ddof=1) / np.sqrt(len(cuts))
            top.errorbar(i, gaps.mean(), yerr=ci, fmt="D", color=COLORS[cond],
                         markersize=8, capsize=5, linewidth=2)
            top.annotate(f"{gaps.mean():.3f}%", (i, gaps.mean()), xytext=(12, 0),
                         textcoords="offset points", va="center", fontsize=11)
            md = mean_distance(dist)
            bottom.errorbar(i, md, yerr=jackknife_halfwidth(dist), fmt="D",
                            color=COLORS[cond], markersize=9, capsize=6, linewidth=2)
            bottom.annotate(f"{md:.3f}", (i, md), xytext=(12, 0),
                            textcoords="offset points", va="center", fontsize=12)
        top.set_title(ds)
        top.set_ylim(bottom=0)
        top.set_ylabel("平均gap [%]（小さいほど良い）")
        bottom.set_ylabel("平均ペア距離（大きいほど多様）")
        bottom.set_ylim(0, .5)
        for ax in (top, bottom):
            ax.set_xlim(-.5, 1.65)
            ax.set_xticks([0, 1], ["基準", "ノイズ1,000倍"])
            finish_axes(ax)
    fig.text(.075, .075, "上段：点は独立試行、ひし形は平均、線は平均の95% t区間。下段：試行を単位とするジャックナイフ95%近似区間。", fontsize=10)
    fig.text(.075, .035, "測っているのはCIMが出力した解集合の多様性。GAの初期集団・世代内の多様性や、GA性能への因果効果ではない。", fontsize=11, color="#934437")
    save(fig, out, "01_quality_and_diversity")


def matched_figure(data, out):
    fig, axs = plt.subplots(2, len(data), figsize=(15, 8.9), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.16, top=.81, wspace=.33, hspace=.55)
    fig.suptitle("同じカット値の集団でも、多様性は同じとは限らない", y=.975, fontsize=20)
    fig.text(.5, .925, "カット値ごとに同数を抽出：各グラフで品質の分布を完全一致", ha="center", fontsize=13)
    fig.legend(handles=[Line2D([], [], marker="o", color=COLORS["baseline"], label="基準CIM"),
                        Line2D([], [], marker="s", color=COLORS["noise1000x"], label="ノイズ振幅1,000倍")],
               loc="upper center", bbox_to_anchor=(.5, .9), ncol=2, frameon=False)
    for col, (ds, a) in enumerate(data.items()):
        top, bottom = axs[:, col]
        match = a["matched"]
        for cond in CONDITIONS:
            cuts = a[cond]["cuts"][match[cond]["indices"]]
            top.plot(np.arange(1, len(cuts) + 1), cuts, marker="o" if cond == "baseline" else "s",
                     markersize=7 if cond == "baseline" else 10,
                     markerfacecolor=COLORS[cond] if cond == "baseline" else "none",
                     linestyle="-" if cond == "baseline" else "--", color=COLORS[cond])
        top.set_title(f"{ds}：各{a['match_count']}解")
        top.set_xlabel("カット値の昇順（2条件で完全一致）")
        top.set_ylabel("カット値")
        top.ticklabel_format(axis="y", style="plain", useOffset=False)
        top.set_xticks(np.arange(1, a["match_count"] + 1, 2))
        for i, cond in enumerate(CONDITIONS):
            values = a["selection_distance"][cond]
            # Range is sensitivity to alternative exact matches, not a CI.
            lo, hi = np.quantile(values, [.05, .95])
            mid = np.median(values)
            bottom.plot([i, i], [lo, hi], color=COLORS[cond], lw=6, alpha=.35)
            bottom.plot(i, mid, "_", color=COLORS[cond], markersize=18)
            md = match[cond]["mean_distance"]
            bottom.plot(i, md, "D", color=COLORS[cond], markersize=9)
            bottom.annotate(f"{md:.3f}", (i, md), xytext=(12, 0),
                            textcoords="offset points", va="center", fontsize=12)
        bottom.set_xlim(-.5, 1.6)
        bottom.set_ylim(0, .5)
        bottom.set_xticks([0, 1], ["基準", "ノイズ1,000倍"])
        bottom.set_ylabel("平均ペア距離（大きいほど多様）")
        for ax in (top, bottom):
            finish_axes(ax)
    fig.text(.075, .095, "ひし形：同じカット値内で保存順に選んだ集団。太線：同値解の選び方を1,000回変えた5–95%範囲（信頼区間ではない）。", fontsize=10)
    fig.text(.075, .062, "距離 d = min(H, N − H) / N。全スピン反転は同一解。品質だけで集団の違いは表せるか、を調べる補助解析。", fontsize=10)
    fig.text(.075, .028, "注意：各7–11解の小標本・事後的な抽出。多様性を増やせばGAが改善する、という証明にはならない。", color="#934437", fontsize=11)
    save(fig, out, "02_exact_quality_matching")


def all_conditions_figure(payload, out):
    groups = {
        1: ("初期振幅", "#53988b", "o"),
        2: ("ポンプ速度", "#cd9530", "^"),
        3: ("ノイズ強度", "#C34436", "s"),
        4: ("更新方式", "#8277ab", "x"),
    }
    fig, axs = plt.subplots(1, len(payload["results"]), figsize=(15, 5.4), squeeze=False)
    fig.subplots_adjust(left=.075, right=.98, bottom=.24, top=.75, wspace=.3)
    fig.suptitle("多様性の増加に、品質低下が伴う条件もある", y=.97, fontsize=20)
    fig.text(.5, .895, "8月23日の全15条件を表示 ／ 共通Tabu Search 20,000反復後 ／ 各32試行の平均", ha="center", fontsize=12)
    for ax, (ds, results) in zip(axs[0], payload["results"].items()):
        for exp, (label, color, marker) in groups.items():
            configs = [c["key"] for c in payload["meta"]["configs"] if c["exp"] == exp]
            points = [results[c] for c in configs]
            ax.scatter([r["gap_pct_mean"] for r in points], [r["mean_pairwise"] for r in points],
                       color=color, marker=marker, s=60, label=label)
        b = results["baseline"]
        ax.scatter(b["gap_pct_mean"], b["mean_pairwise"], color=COLORS["baseline"],
                   marker="*", s=180, zorder=6, label="基準CIM")
        ax.annotate("基準", (b["gap_pct_mean"], b["mean_pairwise"]), xytext=(8, -15), textcoords="offset points", fontsize=10)
        ax.set_title(ds)
        ax.set_xlabel("平均gap [%]（左ほど高品質）")
        ax.set_ylabel("平均ペア距離（上ほど多様）")
        ax.set_ylim(.15, .51)
        finish_axes(ax)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .09), ncol=5, frameon=False)
    fig.text(.075, .035, "解集合の品質と多様性を併せて評価する必要性を示す図。GAの収束曲線とは別の実験であり、点同士を対応付けて因果を主張しない。", fontsize=10, color="#53616e")
    save(fig, out, "03_all_conditions")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--datasets", nargs="+", default=["G22", "G55", "G70"])
    ap.add_argument("--screenshot", type=Path)
    ap.add_argument("--selection-repeats", type=int, default=1000)
    ap.add_argument("--tag", default="discussion")
    args = ap.parse_args()
    if args.selection_repeats < 1:
        raise ValueError("selection-repeats must be positive")
    source = ROOT / args.source
    payload = json.loads((source / "results_ref20k.json").read_text(encoding="utf-8"))
    arrays = np.load(source / "signs_ref20k.npz", allow_pickle=False)
    out = new_output(args)
    style()
    data, metrics, matched_rows, matched_arrays = {}, {}, [], {}
    for ds in args.datasets:
        a = {"bks": payload["meta"]["bks"][ds]}
        for cond in CONDITIONS:
            signs = arrays[f"{ds}__{cond}_signs"]
            cuts = arrays[f"{ds}__{cond}_cuts"]
            # Independent score verification against the actual graph.
            graph = np.loadtxt(ROOT / "input" / f"{ds}.txt", skiprows=1)
            ea, eb = graph[:, 0].astype(int) - 1, graph[:, 1].astype(int) - 1
            w = graph[:, 2]
            recomputed = ((signs[:, ea] != signs[:, eb]) * w).sum(axis=1)
            if not np.array_equal(recomputed, cuts):
                raise ValueError(f"Saved scores mismatch graph: {ds}/{cond}")
            d = distance_matrix(signs)
            expected = payload["results"][ds][cond]["mean_pairwise"]
            if not np.isclose(mean_distance(d), expected, atol=1e-12, rtol=0):
                raise ValueError(f"Distance mismatch: {ds}/{cond}")
            a[cond] = {"signs": signs, "cuts": cuts, "distance": d}
        ia, ib = exact_match(a["baseline"]["cuts"], a["noise1000x"]["cuts"])
        a["matched"] = {}
        a["match_count"] = len(ia)
        a["selection_distance"] = {cond: [] for cond in CONDITIONS}
        for cond, indices in zip(CONDITIONS, (ia, ib)):
            md = mean_distance(a[cond]["distance"][np.ix_(indices, indices)])
            a["matched"][cond] = {"indices": indices, "mean_distance": md}
            matched_arrays[f"{ds}__{cond}_signs"] = a[cond]["signs"][indices]
            matched_arrays[f"{ds}__{cond}_cuts"] = a[cond]["cuts"][indices]
            for rank, index in enumerate(indices, 1):
                matched_rows.append({"dataset": ds, "condition": cond, "rank": rank,
                                     "source_trial_index": int(index), "cut": float(a[cond]["cuts"][index])})
        rng = np.random.default_rng(1907)
        for _ in range(args.selection_repeats):
            i, j = exact_match(a["baseline"]["cuts"], a["noise1000x"]["cuts"], rng)
            for cond, indices in zip(CONDITIONS, (i, j)):
                a["selection_distance"][cond].append(mean_distance(a[cond]["distance"][np.ix_(indices, indices)]))
        metrics[ds] = {"bks": a["bks"], "match_count": len(ia), "conditions": {}}
        for cond in CONDITIONS:
            indices = a["matched"][cond]["indices"]
            values = np.asarray(a["selection_distance"][cond])
            metrics[ds]["conditions"][cond] = {
                "full_n": len(a[cond]["cuts"]), "full_cut_mean": float(a[cond]["cuts"].mean()),
                "full_gap_pct_mean": 100 * (a["bks"] - float(a[cond]["cuts"].mean())) / a["bks"],
                "full_mean_distance": mean_distance(a[cond]["distance"]),
                "matched_cut_mean": float(a[cond]["cuts"][indices].mean()),
                "matched_mean_distance": a["matched"][cond]["mean_distance"],
                "selection_distance_q05_median_q95": np.quantile(values, [.05, .5, .95]).tolist(),
                "source_indices": indices.tolist(),
            }
        data[ds] = a
    full_figure(data, out)
    matched_figure(data, out)
    selected_payload = {**payload, "results": {ds: payload["results"][ds] for ds in args.datasets}}
    all_conditions_figure(selected_payload, out)
    np.savez_compressed(out / "matched_solutions.npz", **matched_arrays)
    with (out / "matched_members.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=matched_rows[0].keys())
        writer.writeheader()
        writer.writerows(matched_rows)
    (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    assets = [source / "results_ref20k.json", source / "signs_ref20k.npz", Path(__file__)]
    shutil.copyfile(Path(__file__), out / "reproduce.py")
    if args.screenshot:
        shutil.copyfile(args.screenshot, out / "original_screenshot.png")
        assets.append(args.screenshot)
    assets.extend(ROOT / "input" / f"{ds}.txt" for ds in args.datasets)
    manifest = {"created": date.today().isoformat(), "source": str(source),
                "selection_repeats": args.selection_repeats, "selection_seed": 1907,
                "solver_experiments_run": False, "screenshot_digitized": False,
                "files": [{"path": str(p.resolve()), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in assets]}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# CIMの多様性について説明するための既存結果の再解析", "",
             "## 最初に伝える結論", "",
             "スクリーンショットは、良い初期解から始めても、その後の改善が大きいとは限らないことを示している。",
             "今回の再解析は、**同じカット値の分布でも、CIM解集合の多様性が異なる**ことを示す。",
             "この2点は『多様性を研究する根拠』になるが、**多様性がGAの停滞の原因であるという証明ではない**。",
             "GAの実際の初期集団や世代内の多様性を測ったデータは、今回確認した保存データにはない。", "",
             "## 図の説明順", "",
             "### 0. ユーザー提供のGA収束曲線", "",
             "![元のスクリーンショット](original_screenshot.png)", "",
             "G55・G70では全CIM初期集団の曲線は序盤から良いが、その後の改善が小さい。G22・K2000では改善も見られる。",
             "多様性の値はこの図には描かれていない。誤差棒の定義・試行数・世代0の定義は元データ未確認のため不明。",
             "リポジトリの warm_ga_q3.py は1個体だけCIMで初期化する実験であり、この『全CIM』の図とは一致しない。",
             "2026-09-14に git pull --ff-only origin master を実施したが Already up to date。master は bb5f62d（2026-08-24）だった。",
             "リモートの他ブランチも確認したが、今回の全CIM初期集団の元データは見つからなかった。",
             "画像はそのまま複製した。数値の読み取り・補間・誤差棒の推定は行っていない。", "",
             "### 1. 品質と多様性を併記する", "",
             "![全32試行の比較](01_quality_and_diversity.png)", "",
             "8月23日に保存された基準CIMとノイズ振幅1,000倍条件の各32試行を使用した。",
             "全解に共通のTabu Search（20,000反復、摂動なし、tenure=15）を施した保存結果を使う。",
             "これはCIM解集合を同じ後処理で比較したものであり、GA内で実際に選ばれた初期集団ではない。",
             "平均gapが近いG55でも多様性には差が見られる。G22で差が小さいことも併記する。",
             "上段の区間は試行平均の95% t区間、下段は試行を単位とした削除1ジャックナイフの95%近似区間。",
             "496個のペアを独立標本とは扱っていない。多重比較を含む探索的記述であり、因果や有意性の断定はしない。", "",
             "### 2. カット値の分布を完全にそろえる", "",
             "![同品質での比較](02_exact_quality_matching.png)", "",
             "カット値ごとに、2条件の解数の小さい方だけ取り出す。保存順の先頭を採用し、距離による選別は行わない。",
             "平均値だけでなく、カット値の多重集合が完全一致する。全32試行からの事後抽出である。", "",
             "|グラフ|各条件の解数|同じ平均カット値|基準CIMの距離|ノイズ条件の距離|", "|---|---:|---:|---:|---:|"]
    for ds, a in metrics.items():
        b, n = a["conditions"]["baseline"], a["conditions"]["noise1000x"]
        lines.append(f"|{ds}|{a['match_count']}|{b['matched_cut_mean']:.3f}|{b['matched_mean_distance']:.4f}|{n['matched_mean_distance']:.4f}|")
    lines += ["", "太線は同じカット値の解をランダムに選び直したときの5–95%範囲。統計的な信頼区間ではなく、選び方への感度である。",
              "標本数は7–11解に減っており、未使用のグラフや独立seed集団での追試が必要。品質をそろえても、配置の構造など他の差は残る。", "",
              "### 3. 都合のよい条件だけを取り出さず、全15条件も見る", "",
              "![全条件](03_all_conditions.png)", "",
              "多様性の増加に品質低下が伴う条件もある。『多様性は大きいほどよい』とは結論できない。", "",
              "## 先生への説明例", "",
              "> 全CIM初期集団は初期の品質が良い一方、G55・G70ではその後の改善が小さくなっています。",
              "> したがって、初期カット値以外の性質を調べる必要があると考えました。",
              "> 保存済みのCIM解を解析すると、カット値の分布を完全にそろえても集団の多様性が異なりました。",
              "> そこで、多様性をCIM出力の評価軸として加え、その形成機構を研究したいです。",
              "> 多様性がGAの停滞を引き起こしているかは、初期品質をそろえた集団を同一GAに渡して検証します。", "",
              "## 因果効果を確かめる次の比較", "",
              "同じ品質分布・同じ集団サイズの中で距離が低い／高い集団を複数構成し、同一GA・同一計算予算・対応する乱数で比較する。",
              "比較するのはGA開始時の品質、初期局所探索後の品質と多様性、各世代の多様性、交叉直後と局所探索後の改善である。",
              "CIM生成条件だけを変えた比較では、多様性以外の構造的変化も生じる。複数の生成条件・集団構成を使い結果の頑健性を調べる。",
              "matched_solutions.npz は今回の同品質部分集合であり、次の検証への入力候補。今回GAを再実行してはいない。", "",
              "## 再現・検証", "",
              f"元データ: `{source.relative_to(ROOT).as_posix()}`", "",
              "元のグラフ辺重みから全解のカット値を独立に再計算し、保存値と完全一致することを確認した。",
              "全体反転を同一視した平均距離も保存JSONと照合済み。元ファイルのSHA-256はmanifest.jsonに記録。",
              "抽出した全メンバーの試行番号はmatched_members.csv、集計値はmetrics.jsonに保存。",
              "PNGは発表用、SVGは拡大用。既存ファイルは上書きしていない。", "",
              "再実行: `python scripts/plotting/plot_cim_diversity_evidence.py --screenshot <元画像のパス>`", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out_dir": str(out), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
