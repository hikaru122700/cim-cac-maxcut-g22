"""ステップ 3 の解析:G-set の各問題で、同じ cut の CIM 解の価値の差(C1)と指標の説明力(C2)。

seed_value_pilot.py の出力ディレクトリを複数受け取り、問題 × SA 温度 × 予算ごとに:
  C1  ラベルを(初期 cut, LS 後 cut)で回帰した残差の split-half 信頼性(Spearman-Brown)
  C2  各指標と残差の Spearman ρ(候補ブートストラップ 2000 回の 95% 区間)
を表にする。指標と判定基準は docs/20261018/step3_plan.md で実行前に固定した。

使い方(プロジェクトルートから):
  python scripts/benchmarks/gset_seed_analysis.py --runs <dir> [<dir> ...] --tag step3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.seed_value_pilot import residual, split_half  # noqa: E402

EXPERIMENT_KIND = "gset_seed_analysis"
BOOT = 2000
BOOT_SEED = 20261020
FEATURES = ["eig_diff", "eig_a1", "eig_a2", "n_gain_minus2_to_0", "ls_n_gain_minus2_to_0",
            "n_zero_gain", "ls_n_zero_gain", "trial_sa_mean", "trial_sa_max"]
GO_RELIABILITY = 0.3
# SA が種を一度でも超えた候補がこの割合未満なら、ラベル ≈ 初期 cut で判定不能とする。
# 残差の ptp による判定は、cut ~1e4 の最小二乗の丸め誤差で漏れる(G55 T=2 50 掃引)。
MIN_IMPROVED_FRAC = 0.05


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def rho_ci(feature, label, cov, seed):
    """残差との Spearman ρ と、候補ブートストラップの 95% 区間(各標本で回帰し直す)。"""
    rho = float(spearmanr(feature, residual(label, cov))[0])
    rng = np.random.default_rng(seed)
    k = len(label)
    boots = np.empty(BOOT)
    for i in range(BOOT):
        idx = rng.integers(0, k, k)
        boots[i] = spearmanr(feature[idx], residual(label[idx], cov[idx]))[0]
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return rho, float(lo), float(hi)


def analyze_run(run_dir: Path) -> list[dict]:
    meta = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    m, budgets = meta["meta"], meta["budgets"]
    z = np.load(run_dir / "raw.npz")
    cov = np.column_stack([z["init_cut"], z["ls_cut"]])
    s = z["cim_signs"]
    s = np.where(s[:, :1] > 0, -s, s)
    n_distinct = len({r.tobytes() for r in s})
    rows = []
    for bi, b in enumerate(m["budgets"]):
        cuts = z[f"cuts_B{b}"]
        label = cuts.mean(axis=1)
        res = budgets[str(b)]
        row = {
            "dataset": m["dataset"], "t_start": m["t_start"], "budget": b,
            "sweeps": b // int(z["cim_signs"].shape[1]),
            "candidates": int(len(label)), "distinct_states": n_distinct,
            "distinct_init_cut": int(len(np.unique(z["init_cut"]))),
            "label_mean": float(label.mean()), "bks": m["bks"],
            "dist_from_seed": res.get("mean_distance_from_seed"),
            "c1_sb": res["split_half_cut_ls_resid_sb"],
            "c1_ci95": res["split_half_cut_ls_resid_sb_ci95"],
        }
        row["c1_go"] = bool(row["c1_sb"] >= GO_RELIABILITY and row["c1_ci95"][0] > 0)
        improved = float((cuts > z["init_cut"][:, None]).any(axis=1).mean())
        row["improved_frac"] = improved
        if improved < MIN_IMPROVED_FRAC or np.ptp(residual(label, cov)) < 1e-9:
            # SA がほぼ種を超えず、ラベル ≈ 初期 cut。C1 は判定不能(信頼性 1.0 は見かけ)。
            row["c1_go"] = None
            row["features"] = {}
            rows.append(row)
            continue
        feats = {}
        for fi, f in enumerate(FEATURES):
            key = f"feat_{f}"
            if key not in z.files or np.ptp(z[key]) == 0:
                continue
            rho, lo, hi = rho_ci(z[key].astype(float), label, cov, BOOT_SEED + 31 * bi + fi)
            feats[f] = {"rho": rho, "ci95": [lo, hi]}
        row["features"] = feats
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--tag", default="step3")
    args = ap.parse_args()
    if not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")
    rows = []
    for r in args.runs:
        for row in analyze_run(r.resolve()):
            row["run_dir"] = str(r.resolve().relative_to(ROOT))
            rows.append(row)
            f = row["features"]
            mark = " undetermined" if row["c1_go"] is None else ""
            print(f"{row['dataset']:>4} T={row['t_start']:<4g} sweeps={row['sweeps']:<5} "
                  f"improved={row['improved_frac']:.2f}{mark} C1={row['c1_sb']:+.2f} [{row['c1_ci95'][0]:+.2f}, {row['c1_ci95'][1]:+.2f}] "
                  + " ".join(f"{k}={v['rho']:+.2f}" for k, v in f.items()), flush=True)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.tag}"
    out_dir.mkdir(parents=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as fh:
        json.dump({"features": FEATURES, "rows": rows}, fh, ensure_ascii=False, indent=2)
    with (out_dir / "table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "t_start", "sweeps", "c1_sb", "c1_lo", "c1_hi", "feature", "rho", "lo", "hi"])
        for row in rows:
            for f, v in row["features"].items() or [("", {"rho": "", "ci95": ["", ""]})]:
                w.writerow([row["dataset"], row["t_start"], row["sweeps"], row["c1_sb"], *row["c1_ci95"],
                            f, v["rho"], *v["ci95"]])
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
