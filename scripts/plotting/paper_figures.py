"""論文(docs/paper)の図を作る。各図は保存済みの結果だけから描く(再計算しない)。

出力:results/<日付>/paper_figures/v{N}_<tag>/ に PNG と PDF。docs/paper/figures/ にも PDF を写す。

使い方(プロジェクトルートから):
  python scripts/plotting/paper_figures.py --figs prereg small --tag draft
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.seed_value_pilot import residual  # noqa: E402

EXPERIMENT_KIND = "paper_figures"
PAPER_FIG_DIR = ROOT / "docs" / "paper" / "figures"
RESULTS = ROOT / "results" / "2026-10-18"

plt.rcParams.update({
    "font.family": "Yu Gothic",
    "axes.unicode_minus": False,
    "axes.labelsize": 20,
    "axes.titlesize": 20,
    "figure.titlesize": 22,
    "legend.fontsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.size": 14,
})
INK = "#263D50"
ACCENT = "#C34436"
MUTED = "#8A9BA8"
BLUE = "#2F6B9A"


def style(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, ls=":", alpha=0.4)


def get_out_dir(tag: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{tag}"
    out.mkdir(parents=True)
    return out


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out / f"{name}.pdf", PAPER_FIG_DIR / f"{name}.pdf")


def pilot_dir(label: str) -> Path:
    return next((RESULTS / "seed_value_pilot").glob(f"*_{label}"))


def prereg_result(label: str, budget: int) -> dict:
    for d in sorted((RESULTS / "prereg_eigen_analysis").iterdir()):
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if r["label"] == label and r["budget"] == budget:
            last = r
    return last


MAIN_BUDGET = {"R1": 10_000_000, "R2": 10_000_000, "R3": 10_000_000, "R4": 25_000_000, "R5": 50_000_000}
RUN_NAME = {"R1": "R1 G22\n600ラウンド", "R2": "R2 G22\n高温開始", "R3": "R3 G22\n2400ラウンド",
            "R4": "R4 G55", "R5": "R5 G70"}


def fig_prereg(out: Path):
    # (a) R1 の散布図
    z = np.load(pilot_dir("R1") / "raw.npz")
    label = z["cuts_B10000000"].mean(axis=1)
    r = residual(label, np.column_stack([z["init_cut"], z["ls_cut"]]))
    e = z["feat_eig_diff"]
    res1 = prereg_result("R1", 10_000_000)
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(e, r, s=22, color=BLUE, alpha=0.75, edgecolor="none")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("固有ベクトル指標 $E = a_1 - a_2$")
    ax.set_ylabel("SA後の価値の残差")
    lo, hi = res1["primary"]["ci95"]
    ax.set_title(f"G22（R1）  ρ = {res1['primary']['spearman_E_vs_residual']:.2f} [{lo:.2f}, {hi:.2f}]")
    style(ax)
    save(fig, out, "fig_prereg_scatter")

    # (b) R1〜R5 の ρ と区間、C1 の信頼性
    labels = ["R1", "R2", "R3", "R4", "R5"]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    ax = axes[0]
    for i, L in enumerate(labels):
        p = prereg_result(L, MAIN_BUDGET[L])["primary"]
        color = ACCENT if p["replicated"] else MUTED
        ax.errorbar(i, p["spearman_E_vs_residual"],
                    yerr=[[p["spearman_E_vs_residual"] - p["ci95"][0]], [p["ci95"][1] - p["spearman_E_vs_residual"]]],
                    fmt="o", ms=10, color=color, capsize=6, lw=2.5)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(range(len(labels)), [RUN_NAME[L] for L in labels], fontsize=20)
    ax.set_ylabel("ρ(E, 残差)")
    ax.set_title("(a) 事前登録の判定（主予算）")
    style(ax)

    ax = axes[1]
    for L, color, marker in [("R1", ACCENT, "o"), ("R3", "#E08E45", "s"), ("R2", MUTED, "D"),
                             ("R4", BLUE, "^"), ("R5", INK, "v")]:
        s = json.loads((pilot_dir(L) / "summary.json").read_text(encoding="utf-8"))
        n = np.load(pilot_dir(L) / "raw.npz")["cim_signs"].shape[1]
        xs, ys = [], []
        for b, v in s["budgets"].items():
            if v["noise_se_per_candidate"] == 0:
                continue  # SA が種から動かず判定不能
            xs.append(int(b) / n)
            ys.append(v["split_half_cut_ls_resid_sb"])
        ax.plot(xs, ys, marker=marker, ms=10, lw=2.5, color=color, label=L)
    ax.axhline(0.3, color=MUTED, ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("SA の予算（掃引）")
    ax.set_ylabel("残差の split-half 信頼性")
    ax.set_title("(b) 同じ cut での価値の差（C1）")
    ax.legend(ncol=2, loc="lower left")
    style(ax)
    fig.tight_layout()
    save(fig, out, "fig_prereg_summary")


def fig_small(out: Path):
    feats = [("opt_distance", "最適解までの距離"), ("path_length", "経路長"), ("barrier", "障壁"),
             ("trial_sa_success", "試し打ち"), ("ls_gain", "LSの改善量"),
             ("ls_n_gain_minus2_to_0", "LS後の利得−2〜0の頂点数"), ("eig_a1", "$a_1$"), ("eig_diff", "$E$")]
    res = {}
    for d in sorted((RESULTS / "small_seed_analysis").iterdir()):
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
        if r["label"] == "success_S10" and r.get("family") is None:
            res[r["split"]] = {row["feature"]: row for row in r["rows"]}  # 後のファイルで上書き
    fig, ax = plt.subplots(figsize=(10, 7.5))
    y = np.arange(len(feats))[::-1]
    for off, split, color, name in [(0.17, "dev", MUTED, "開発用"), (-0.17, "test", ACCENT, "検証用")]:
        t = np.array([res[split][f]["tau"] for f, _ in feats])
        lo = np.array([res[split][f]["tau_ci95"][0] for f, _ in feats])
        hi = np.array([res[split][f]["tau_ci95"][1] for f, _ in feats])
        ax.errorbar(t, y + off, xerr=[t - lo, hi - t], fmt="o", ms=9, capsize=5, lw=2.2, color=color, label=name)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, [n for _, n in feats], fontsize=20)
    ax.set_xlabel("同じ cut の層内の Kendall τ（10掃引の成功率）")
    ax.legend(loc="lower left")
    style(ax)
    save(fig, out, "fig_small_tau")


def fig_ga(out: Path):
    old = ROOT / "results/2026-09-14/ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_measured"
    new = RESULTS / "ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_fixedTS"
    meta = json.loads((new / "metadata.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    styles = [(old, "random", MUTED, "--", "旧TS・ランダム"), (old, "cim", "#E08E45", "--", "旧TS・CIM"),
              (new, "random", BLUE, "-", "新TS・ランダム"), (new, "cim", ACCENT, "-", "新TS・CIM")]
    for ax, ds in zip(axes, ["G55", "G70"]):
        bks = meta["datasets"][ds]["bks"]
        for d, cond, color, ls, name in styles:
            be = np.load(d / f"{ds}_{cond}_trajectory.npz")["best_ever"][1:]
            gap = bks - be
            g = np.arange(len(gap))
            m = gap.mean(axis=1)
            se = gap.std(axis=1, ddof=1) / np.sqrt(gap.shape[1])
            ax.plot(g, m, color=color, ls=ls, lw=2.8, label=name)
            ax.fill_between(g, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.15, lw=0)
        ax.set_yscale("log")
        ax.set_xlabel("世代")
        ax.set_ylabel("BKSとの差（16回の平均）")
        ax.set_title(ds)
        style(ax)
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "fig_ga")


GSET_ORDER = ["G1", "G22", "G23", "G55", "G70", "G14", "G15", "G39", "G32"]
GSET_COLS = [(0.5, 50), (0.5, 500), (0.5, 5000), (2.0, 50), (2.0, 500), (2.0, 5000)]


def gset_grid(rows: list[dict], key) -> tuple[np.ndarray, np.ndarray]:
    """問題 × (温度, 掃引) の値と状態(0: C1 なし, 1: C1 あり, 2: 判定不能)。"""
    val = np.full((len(GSET_ORDER), len(GSET_COLS)), np.nan)
    state = np.zeros_like(val)
    for r in rows:
        i, j = GSET_ORDER.index(r["dataset"]), GSET_COLS.index((r["t_start"], r["sweeps"]))
        state[i, j] = 2 if r["c1_go"] is None else int(r["c1_go"])
        val[i, j] = key(r)
    return val, state


def draw_grid(ax, val, state, title, only_go):
    shown = np.where((state == 1) | ((not only_go) & (state == 0)), val, np.nan)
    ax.imshow(np.ma.masked_invalid(shown), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for i in range(val.shape[0]):
        for j in range(val.shape[1]):
            if state[i, j] == 2:
                ax.text(j, i, "×", ha="center", va="center", fontsize=16, color=MUTED)
            elif only_go and state[i, j] == 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=14, color=MUTED)
            elif np.isfinite(val[i, j]):
                ax.text(j, i, f"{val[i, j]:.2f}", ha="center", va="center", fontsize=13,
                        color="white" if abs(val[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(GSET_COLS)), [str(s) for _, s in GSET_COLS], fontsize=20)
    ax.set_yticks(range(len(GSET_ORDER)), GSET_ORDER, fontsize=20)
    ax.axvline(2.5, color=INK, lw=2)
    ax.set_title(title)
    ax.set_xlabel("掃引数（左：$T_0=0.5$，右：$T_0=2.0$）")
    ax.tick_params(direction="in", which="both", top=True, right=True)


def fig_gset(out: Path):
    r = json.loads((RESULTS / "gset_seed_analysis/v2_step3/result.json").read_text(encoding="utf-8"))
    rows = r["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    panels = [(lambda x: x["c1_sb"], "(a) 残差の信頼性 C1", False),
              (lambda x: x["features"].get("eig_diff", {}).get("rho", np.nan), "(b) $\\rho(E, r)$", True),
              (lambda x: x["features"].get("trial_sa_mean", {}).get("rho", np.nan), "(c) $\\rho$(試し打ち, $r$)", True),
              (lambda x: x["features"].get("ls_n_gain_minus2_to_0", {}).get("rho", np.nan),
               "(d) $\\rho$(LS後の利得−2〜0の数, $r$)", True)]
    for ax, (key, title, only_go) in zip(axes.flat, panels):
        val, state = gset_grid(rows, key)
        draw_grid(ax, val, state, title, only_go)
    fig.tight_layout()
    save(fig, out, "fig_gset")


SEL_NAMES = [("cim_random", "無作為"), ("cim_cut", "cut上位"), ("cim_ls", "LS後cut上位"), ("cim_div", "多様性"),
             ("cim_trial", "試し打ち"), ("model_all", "学習（全指標）"), ("model_eig", "学習（固有）"),
             ("random_init", "ランダム初期化")]


SELECTION_DIR = ROOT / "results" / "2026-10-19" / "selection_eval"  # ほかの計算がない状態の時間で評価した版


def selection_result(prefix: str) -> dict:
    d = next(SELECTION_DIR.glob(f"{prefix}_*"))
    return json.loads((d / "result.json").read_text(encoding="utf-8"))


def mean_gap(r: dict, s: str) -> float:
    """全問題で実行できた方法だけ平均する(1 問題でも本数 0 なら NaN)。"""
    g = [v["strategies"][s]["gap_to_bks_pct"] for v in r["per_problem"].values()]
    return float(np.mean(g)) if np.all(np.isfinite(g)) else np.nan


def fig_selection(out: Path):
    s500, s5000 = selection_result("v18"), selection_result("v17")
    t2_500, t2_5000 = selection_result("v25"), selection_result("v29")
    fig, axes = plt.subplots(1, 2, figsize=(22, 8.5), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    y = np.arange(len(SEL_NAMES))[::-1]
    for off, r, color, name in [(0.2, s500, BLUE, "500掃引"), (-0.2, s5000, ACCENT, "5000掃引")]:
        g = np.array([mean_gap(r, s) for s, _ in SEL_NAMES])
        ax.barh(y + off, np.nan_to_num(g), height=0.38, color=color, label=name)
        for yy, gg in zip(y + off, g):
            ax.text((gg if np.isfinite(gg) else 0) + 0.005, yy, f"{gg:.3f}" if np.isfinite(gg) else "実行不能",
                    va="center", fontsize=13)
    ax.set_yticks(y, [n for _, n in SEL_NAMES], fontsize=20)
    ax.set_xlabel("BKSとの差（%，7問題の平均）")
    ax.set_title("(a) 同じ総時間での比較（$T_0=0.5$）")
    ax.set_xlim(0, 0.85)
    ax.legend(loc="upper right")
    style(ax)
    ax = axes[1]
    probs = list(s5000["per_problem"])
    x = np.arange(len(probs))
    for r, color, marker, name in [(s500, BLUE, "o", "500掃引，$T_0=0.5$"), (t2_500, BLUE, "s", "500掃引，$T_0=2.0$"),
                                   (s5000, ACCENT, "o", "5000掃引，$T_0=0.5$"), (t2_5000, ACCENT, "s", "5000掃引，$T_0=2.0$")]:
        d = [r["per_problem"][k]["strategies"]["random_init"]["gap_to_bks_pct"]
             - r["per_problem"][k]["strategies"]["cim_ls"]["gap_to_bks_pct"] for k in probs]
        ax.plot(x, d, marker=marker, ms=11, ls="none", color=color, mfc=color if marker == "o" else "white",
                mew=2.2, label=name)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x, probs, fontsize=20)
    ax.set_ylabel("ランダム初期化 − CIM（%）")
    ax.set_title("(b) 正なら CIM の種が良い")
    ax.legend(loc="upper left", fontsize=20)
    style(ax)
    fig.tight_layout()
    save(fig, out, "fig_selection")


FIGS = {"prereg": fig_prereg, "small": fig_small, "ga": fig_ga, "gset": fig_gset, "selection": fig_selection}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", nargs="+", default=list(FIGS), choices=list(FIGS))
    ap.add_argument("--tag", default="draft")
    args = ap.parse_args()
    out = get_out_dir(args.tag)
    for name in args.figs:
        FIGS[name](out)
        print(f"{name}: done", flush=True)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
