"""ステップ 2 の解析:同じ cut の候補どうしで、指標がラベル・厳密量とどれだけ一致するか。

small_seed_study.py の出力を読み、--split の問題だけを使う。開発用(dev)で指標を
選んでから、検証用(test)で確かめる。

  層       (問題, 初期 cut) の組。候補が MIN_STRATUM 個以上の層だけを使う。
  τ        層ごとの Kendall τ_b(指標 vs ラベル)を、層の候補数で重み付け平均した値。
  偏 R²    層ごとに平均を引いた指標とラベルの相関の 2 乗(cut の層の固定効果を除いた説明力)。
  区間     問題を単位にしたブートストラップ(2000 回)の 95% 区間。

使い方(プロジェクトルートから):
  python scripts/benchmarks/small_seed_analysis.py --run <dir> --split dev --label success_S10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPERIMENT_KIND = "small_seed_analysis"
MIN_STRATUM = 5
BOOT = 2000
BOOT_SEED = 20261019
CHEAP = ["n_pos_gain", "n_zero_gain", "sum_pos_gain", "largest_zero_cluster", "n_gain_minus2_to_0",
         "ls_gain", "ls_n_zero_gain", "ls_largest_zero_cluster", "ls_n_gain_minus2_to_0",
         "eig_a1", "eig_a2", "eig_diff", "n_unsat", "unsat_components", "unsat_largest",
         "cim_count", "trial_sa_mean", "trial_sa_success"]
EXACT = ["barrier", "path_length", "opt_distance"]


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


def load(run_dir: Path, split: str) -> list[dict]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    out = []
    for rec in summary["records"]:
        if rec["split"] != split:
            continue
        z = np.load(run_dir / f"{rec['name']}.npz")
        out.append({"rec": rec, **{k: z[k] for k in z.files}})
    return out


def column(p: dict, name: str) -> np.ndarray:
    if name in EXACT or name in ("ls_cut", "init_cut"):
        return p[name].astype(float)
    return p[f"feat_{name}"].astype(float)


def problem_sums(p: dict, feature: str, target: str) -> np.ndarray:
    """1 問題分の集計量 [Σ τ·w, Σ w, Σ x̃ỹ, Σ x̃², Σ ỹ²](x̃, ỹ は層内で平均を引いた値)。"""
    x = column(p, feature)
    y = p[target].astype(float) if target in p else column(p, target)
    out = np.zeros(5)
    for c in np.unique(p["init_cut"]):
        m = p["init_cut"] == c
        if m.sum() < MIN_STRATUM or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
            continue
        t = kendalltau(x[m], y[m]).statistic
        if not np.isfinite(t):
            continue
        xd, yd = x[m] - x[m].mean(), y[m] - y[m].mean()
        out += [t * m.sum(), m.sum(), (xd * yd).sum(), (xd ** 2).sum(), (yd ** 2).sum()]
    return out


def combine(sums: np.ndarray) -> tuple[float, float, int]:
    """問題ごとの集計量 (P, 5) から (重み付き τ, 偏 R², 候補数)。"""
    tw, w, xy, xx, yy = sums.sum(axis=0)
    if w == 0:
        return float("nan"), float("nan"), 0
    r2 = float(xy ** 2 / (xx * yy)) if xx * yy > 0 else float("nan")
    return float(tw / w), r2, int(w)


def strata_stats(problems: list[dict], feature: str, target: str) -> tuple[float, float, int]:
    """(重み付き τ, 偏 R², 使った候補数)。target はラベル名か厳密量の名前。"""
    return combine(np.array([problem_sums(p, feature, target) for p in problems]))


def bootstrap(problems, feature, target, seed):
    sums = np.array([problem_sums(p, feature, target) for p in problems])
    rng = np.random.default_rng(seed)
    taus, r2s = [], []
    for _ in range(BOOT):
        t, r2, _ = combine(sums[rng.integers(0, len(problems), len(problems))])
        taus.append(t)
        r2s.append(r2)
    ci = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]  # noqa: E731
    return ci(taus), ci(r2s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--label", default="success_S10")
    ap.add_argument("--features", nargs="*", default=None, help="解析する指標(既定は全部)")
    ap.add_argument("--family", default=None, help="問題の種類で絞る(探索的な内訳用)")
    args = ap.parse_args()
    run_dir = args.run.resolve()
    problems = load(run_dir, args.split)
    if args.family:
        problems = [p for p in problems if p["rec"]["family"] == args.family]
    if not problems:
        ap.error(f"no problems for split {args.split}")
    if args.label not in problems[0]:
        ap.error(f"unknown label {args.label}")
    features = args.features or (CHEAP + EXACT)

    rows = []
    for i, f in enumerate(features):
        tau, r2, used = strata_stats(problems, f, args.label)
        tau_ci, r2_ci = bootstrap(problems, f, args.label, BOOT_SEED + i)
        row = {"feature": f, "target": args.label, "tau": tau, "tau_ci95": tau_ci,
               "partial_r2": r2, "partial_r2_ci95": r2_ci, "candidates_used": used}
        for g in EXACT:
            if f != g:
                row[f"tau_vs_{g}"] = strata_stats(problems, f, g)[0]
        rows.append(row)
        print(f"{f:>24}: tau={tau:+.3f} [{tau_ci[0]:+.3f}, {tau_ci[1]:+.3f}] "
              f"pR2={r2:.3f} [{r2_ci[0]:.3f}, {r2_ci[1]:.3f}] n={used} "
              + " ".join(f"tau({g})={row.get(f'tau_vs_{g}', float('nan')):+.2f}" for g in EXACT),
              flush=True)

    kind_root = get_kind_root()
    fam = f"_{args.family}" if args.family else ""
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.split}{fam}_{args.label}"
    out_dir.mkdir(parents=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as fh:
        json.dump({"run_dir": str(run_dir.relative_to(ROOT)), "split": args.split,
                   "label": args.label, "family": args.family, "problems": [p["rec"]["name"] for p in problems],
                   "min_stratum": MIN_STRATUM, "rows": rows}, fh, ensure_ascii=False, indent=2)
    with (out_dir / "table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "tau", "tau_lo", "tau_hi", "partial_r2", "r2_lo", "r2_hi", "n"])
        for r in rows:
            w.writerow([r["feature"], r["tau"], *r["tau_ci95"], r["partial_r2"], *r["partial_r2_ci95"],
                        r["candidates_used"]])
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
