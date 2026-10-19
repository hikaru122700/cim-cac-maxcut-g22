"""検証 4 の解析:ga_equal_time.py の結果を、CIM の生成時間込みの時刻に換算して判定・図示する。

判定は docs/20261019/verify4_ga_plan.md で実行前に固定した:
  V4a  c=1、T_eval で CIM(選別なし)< ランダム(BKS との差)。問題ごとに片側 Mann-Whitney U、3 問題で Holm
  V4b  c=1、T_eval で CIM+カット値+ハミング距離 < CIM+カット値。問題ごとに片側 Wilcoxon(対応あり)、Holm

出力:入力ディレクトリの中に analysis/ を作り、result.json と図(PNG・PDF)を置く(既にあれば止める)。

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_equal_time_analysis.py --run results/2026-10-19/ga_equal_time/v1_...
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

MULTIPLIERS = [1.0, 0.1, 0.0]  # ga_equal_time.py と同じ(世代数はこの全倍率で足りるように決めてある)
NAMES = {"random": "ランダム", "cim_all": "CIM\n（選別なし）", "cim_cut": "CIM＋\nカット値",
         "cim_div": "CIM＋カット値＋\nハミング距離"}
COLORS = {"random": "#8A9BA8", "cim_all": "#E0A030", "cim_cut": "#2F6B9A", "cim_div": "#C34436"}
FS = 22  # 文字・数値とも 20 以上

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": FS,
    "axes.titlesize": FS, "legend.fontsize": 20, "xtick.labelsize": FS, "ytick.labelsize": FS,
    "font.size": FS,
})


def n_cim(cond: str, info: dict) -> int:
    return {"random": 0, "cim_all": info["pop_size"]}.get(cond, info["K"])


def timing_of(cond: str, t: dict) -> tuple[float, float]:
    return (t["t_init_random"], t["t_gen_random"]) if cond == "random" else (t["t_init_cim"], t["t_gen_cim"])


def best_at(time_s: float, cond: str, c: float, z, info: dict, t: dict) -> np.ndarray:
    """時刻 time_s(1 スレッド CPU 秒)までの各 trial の最良カット値。"""
    hist = z[f"{cond}_history"]
    t_init, t_gen = timing_of(cond, t)
    cim_time = c * n_cim(cond, info) * t["t_cim"]
    after = time_s - cim_time - t_init
    if after >= 0:
        g = math.floor(after / t_gen + 1e-9)
        if g >= len(hist):
            raise ValueError(f"{cond}: needs generation {g} but only {len(hist) - 1} were run")
        return hist[g].astype(float)
    if cond == "random":
        return np.full(hist.shape[1], np.nan)
    # GA の開始前:それまでに作った CIM 解(この条件で作る分)の最大
    made = n_cim(cond, info) if c == 0 else min(math.floor(time_s / (c * t["t_cim"]) + 1e-9), n_cim(cond, info))
    if made == 0:
        return np.full(hist.shape[1], np.nan)
    return z["cim_cuts"][:, :made].max(axis=1).astype(float)  # 選別の有無によらず、作った分の最大


def t_eval(info: dict, t: dict, eval_gens: int) -> float:
    return info["K"] * t["t_cim"] + t["t_init_cim"] + eval_gens * t["t_gen_cim"]


def holm(p: dict) -> dict:
    order = sorted(p, key=p.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[k]))
        out[k] = running
    return out


def analyze(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    eval_gens = int(meta["args"]["eval_gens"])
    res = {"run": str(run), "eval_gens": eval_gens, "datasets": {}}
    for ds, info in meta["datasets"].items():
        t = meta["timing"][ds]
        z = np.load(run / f"{ds}.npz")
        te = t_eval(info, t, eval_gens)
        d = {"t_eval_s": te, "bks": info["bks"], "init": info["init"], "by_c": {}}
        for c in MULTIPLIERS:
            gaps = {cond: (info["bks"] - best_at(te, cond, c, z, info, t)) / info["bks"] * 100
                    for cond in meta["conditions"]}
            row = {"gap_pct": {k: v.tolist() for k, v in gaps.items()},
                   "mean_gap_pct": {k: float(np.mean(v)) for k, v in gaps.items()}}
            row["p_V4a"] = float(mannwhitneyu(gaps["cim_all"], gaps["random"], alternative="less").pvalue)
            diff = gaps["cim_div"] - gaps["cim_cut"]
            row["p_V4b"] = float(wilcoxon(diff, alternative="less").pvalue) if np.any(diff != 0) else 1.0
            row["p_cut_vs_random"] = float(mannwhitneyu(gaps["cim_cut"], gaps["random"], alternative="less").pvalue)
            d["by_c"][str(c)] = row
        res["datasets"][ds] = d
    main = {ds: v["by_c"]["1.0"] for ds, v in res["datasets"].items()}
    res["V4a_holm"] = holm({ds: r["p_V4a"] for ds, r in main.items()})
    res["V4b_holm"] = holm({ds: r["p_V4b"] for ds, r in main.items()})
    return res, meta


def fig_bars(res: dict, c: str, out: Path, name: str):
    dss = list(res["datasets"])
    fig, axes = plt.subplots(1, len(dss), figsize=(11 * len(dss), 8), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        row = res["datasets"][ds]["by_c"][c]
        conds = list(NAMES)
        m = [row["mean_gap_pct"][k] for k in conds]
        se = [np.std(row["gap_pct"][k], ddof=1) / np.sqrt(len(row["gap_pct"][k])) for k in conds]
        x = np.arange(len(conds))
        ax.bar(x, m, width=0.62, color=[COLORS[k] for k in conds], yerr=se, capsize=8,
               error_kw={"lw": 2})
        for xx, mm, ss in zip(x, m, se):
            ax.text(xx, mm + ss + max(m) * 0.02, f"{mm:.3f}", ha="center", va="bottom", fontsize=FS)
        ax.set_xticks(x, [NAMES[k] for k in conds], fontsize=20)
        ax.set_ylabel("BKSとの差（%）")
        ax.set_ylim(0, max(m) * 1.25)
        ax.set_title(ds, pad=12)
        ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_curves(res: dict, meta: dict, run: Path, out: Path):
    dss = list(res["datasets"])
    fig, axes = plt.subplots(1, len(dss), figsize=(11 * len(dss), 8), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        info, t = meta["datasets"][ds], meta["timing"][ds]
        z = np.load(run / f"{ds}.npz")
        te = res["datasets"][ds]["t_eval_s"]
        ts = np.linspace(te * 0.002, te, 400)
        for cond in NAMES:
            g = np.array([np.nanmean((info["bks"] - best_at(x, cond, 1.0, z, info, t)) / info["bks"] * 100)
                          for x in ts])
            ax.plot(ts, g, color=COLORS[cond], lw=3, label=NAMES[cond].replace("\n", ""))
        ax.set_yscale("log")
        ax.set_xlabel("CPU時間（秒，1スレッド換算）")
        ax.set_ylabel("BKSとの差（%）")
        ax.set_title(ds, pad=12)
        ax.tick_params(direction="in", which="both", top=True, right=True)
    axes[0][-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / "curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    if not (args.run / "meta.json").is_file():
        ap.error(f"meta.json not found in {args.run}")
    out = args.run / "analysis"
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    res, meta = analyze(args.run)
    out.mkdir()
    (out / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    for ds, d in res["datasets"].items():
        print(f"{ds}: T_eval={d['t_eval_s']:.1f}s")
        for c, row in d["by_c"].items():
            means = ", ".join(f"{k}={v:.3f}" for k, v in row["mean_gap_pct"].items())
            print(f"  c={c}: {means} | p_V4a={row['p_V4a']:.3g} p_V4b={row['p_V4b']:.3g} "
                  f"p_cut<rand={row['p_cut_vs_random']:.3g}")
        print("  init:", json.dumps(d["init"], ensure_ascii=False))
    print("V4a Holm:", res["V4a_holm"])
    print("V4b Holm:", res["V4b_holm"])
    fig_bars(res, "1.0", out, "bars_c1")
    fig_bars(res, "0.0", out, "bars_c0")
    fig_curves(res, meta, args.run, out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
