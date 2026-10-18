"""事前登録(docs/20261018/prereg_eigen_v1.md)どおりに、固有ベクトル指標を判定する。

seed_value_pilot.py の出力ディレクトリ(raw.npz)を読み、登録した 1 つの予算で
次を計算する。解析の中身は事前登録の時点で固定し、結果を見てから変えない。

  ラベル   y_c = 候補 c に SA を R 回かけた最良 cut の平均
  残差     r_c = y_c を (初期 cut, 1-flip 貪欲LS後の cut) で OLS 回帰した残差
  主指標   E_c = |<s_c, sign(u1)>|/n - |<s_c, sign(u2)>|/n
           (u1, u2 は隣接行列 A の最小固有値側の 2 本)
  主統計量 Spearman ρ(E, r)。候補を 10000 回ブートストラップした 95% 区間
           (各標本で残差を回帰し直す)。下限 > 0 なら「再現」。

副次的に、Pearson r^2(残差の何割を説明するか)、それを残差の split-half
信頼性で割った値(再現できる部分の何割か)、ラベルの二峰性(GMM 1 成分と
2 成分の BIC 差と、2 成分の分離度 Ashman's D)を出す。

使い方(プロジェクトルートから):
  python scripts/benchmarks/prereg_eigen_analysis.py --run <run_dir> --budget 10000000 --label R1
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.seed_value_pilot import residual, split_half  # noqa: E402

EXPERIMENT_KIND = "prereg_eigen_analysis"
BOOT_SAMPLES = 10_000
BOOT_SEED = 20261018
BIMODAL_DBIC = 10.0
BIMODAL_MIN_WEIGHT = 0.1
BIMODAL_ASHMAN_D = 2.0
EXPLORATORY = ["eig_a1", "eig_a2", "n_zero_gain", "n_gain_minus2_to_0", "ls_n_zero_gain",
               "ls_largest_zero_cluster", "largest_zero_cluster", "n_pos_gain"]


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


def spearman_resid(feature: np.ndarray, label: np.ndarray, cov: np.ndarray) -> float:
    return float(spearmanr(feature, residual(label, cov))[0])


def bootstrap_ci(feature, label, cov, samples=BOOT_SAMPLES, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    k = len(label)
    out = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, k, size=k)
        out[i] = spearman_resid(feature[idx], label[idx], cov[idx])
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def bimodality(label: np.ndarray) -> dict:
    x = label.reshape(-1, 1)
    g1 = GaussianMixture(1, random_state=0).fit(x)
    g2 = GaussianMixture(2, n_init=10, random_state=0).fit(x)
    d_bic = float(g1.bic(x) - g2.bic(x))
    order = np.argsort(g2.means_.ravel())
    means = g2.means_.ravel()[order]
    sds = np.sqrt(g2.covariances_.ravel()[order])
    weights = g2.weights_[order]
    ashman_d = float(np.sqrt(2) * abs(means[1] - means[0]) / np.sqrt(sds[0] ** 2 + sds[1] ** 2))
    return {
        "delta_bic_1_minus_2": d_bic,
        "gmm2_means": means.tolist(),
        "gmm2_sds": sds.tolist(),
        "gmm2_weights": weights.tolist(),
        "ashman_d": ashman_d,
        "bimodal": bool(d_bic > BIMODAL_DBIC and weights.min() >= BIMODAL_MIN_WEIGHT
                        and ashman_d > BIMODAL_ASHMAN_D),
    }


def analyze(run_dir: Path, budget: int) -> dict:
    z = np.load(run_dir / "raw.npz")
    cuts = z[f"cuts_B{budget}"]
    label = cuts.mean(axis=1)
    cov = np.column_stack([z["init_cut"], z["ls_cut"]])
    resid = residual(label, cov)
    e = z["feat_eig_diff"]

    rho = spearman_resid(e, label, cov)
    lo, hi = bootstrap_ci(e, label, cov)
    r2 = float(pearsonr(e, resid)[0] ** 2)
    _, _, rel = split_half(cuts, cov)
    exploratory = {}
    for name in EXPLORATORY:
        f = z[f"feat_{name}"]
        if np.ptp(f) > 0:
            exploratory[name] = spearman_resid(f, label, cov)
    return {
        "run_dir": str(run_dir.relative_to(ROOT)),
        "budget": budget,
        "candidates": int(cuts.shape[0]),
        "repeats": int(cuts.shape[1]),
        "primary": {
            "spearman_E_vs_residual": rho,
            "ci95": [lo, hi],
            "replicated": bool(lo > 0),
        },
        "pearson_r2_E_vs_residual": r2,
        "residual_split_half_reliability_sb": float(rel),
        "share_of_reliable_residual": float(r2 / rel) if rel > 0 else None,
        "spearman_E_vs_init_cut": float(spearmanr(e, z["init_cut"])[0]),
        "bimodality": bimodality(label),
        "exploratory_spearman_vs_residual": exploratory,
        "low_eigenvalues_A": z["eig_vals"].tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--label", required=True, help="事前登録の実行名(R1 など)")
    args = ap.parse_args()
    if not args.label.isalnum():
        ap.error("label must be alphanumeric")
    run_dir = args.run.resolve()

    res = analyze(run_dir, args.budget)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.label}_B{args.budget}"
    out_dir.mkdir(parents=True)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump({"label": args.label, **res}, f, ensure_ascii=False, indent=2)
    p = res["primary"]
    print(
        f"{args.label}: rho(E, resid)={p['spearman_E_vs_residual']:+.3f} "
        f"CI95=[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] replicated={p['replicated']} "
        f"r2={res['pearson_r2_E_vs_residual']:.3f} rel={res['residual_split_half_reliability_sb']:.3f} "
        f"bimodal={res['bimodality']['bimodal']} (dBIC={res['bimodality']['delta_bic_1_minus_2']:.1f}, "
        f"D={res['bimodality']['ashman_d']:.2f})"
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
