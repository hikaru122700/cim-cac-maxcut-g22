"""SA 1 本ごとの到達点の分布から「k 本の最良値の期待値」を計算し、同じ総時間の比較(selection_eval)の実測と比べる。

中間発表 v20 の講評(論点 1):1 本あたりの平均では CIM 開始が良いのに、同じ総時間の最良値ではランダム開始が
勝つ理由を、分布(平均とばらつき)と本数で説明できるかを確かめる。新しい SA は回さない。

  CIM 開始  : ステップ 3 の候補 512 個(各 16 回の SA)。K 個を抜き出してカット値上位 16 個を選ぶ手順を MC 回くり返し、
              選ばれた候補の SA 結果を、選ばれた回数で重み付けした分布を「1 本ごとの分布」とする。
              k 本の最良値の期待値は、この分布から独立に k 本引いたとして計算する(候補どうしの相関は入れない)。
  ランダム開始: random_init_sa の結果から k 本を重複なしで選んだときの最大値の期待値(厳密)。
  実測      : selection_eval の result.json(cim_cut と random_init の mean_best)。

使い方(プロジェクトルートから):
  python scripts/benchmarks/best_of_k_analysis.py --runs results/2026-10-18/seed_value_pilot/v*_K512_R16_cim2400_T2_seed1000_S3 \
      --random results/2026-10-18/random_init_sa/v* --evals results/2026-11-10/selection_eval/v*_Kabl_T2_S5000_* \
      --sweeps 5000 --problems G22 G55 G70
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.selection_eval import load_problem, top_m  # noqa: E402
from scripts.benchmarks.selection_eval_bootstrap import expected_max, load_random_pools  # noqa: E402

EXPERIMENT_KIND = "best_of_k"
MC = 4000
SEED = 20261110
M = 16  # SA に渡す本数
K_EQUAL = [1, 16]  # 本数をそろえた比較で見る k


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    heads = [p.name.split("_", 1)[0] for p in kind_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    return max([int(h[1:]) for h in heads if h[1:].isdigit()], default=0) + 1


def iid_expected_max(values: np.ndarray, weights: np.ndarray, k: int) -> float:
    """重み付き分布から独立に k 本引いたときの最大値の期待値。"""
    order = np.argsort(values)
    v = values[order]
    cdf = np.cumsum(weights[order]) / weights.sum()
    prev = np.concatenate([[0.0], cdf[:-1]])
    return float((v * (cdf ** k - prev ** k)).sum())


def cim_distribution(p: dict, pool: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """「pool 個から上位 M 個」で選ばれる候補の SA 結果と、その重み(選ばれた回数)。"""
    count = np.zeros(len(p["cuts"]))
    for _ in range(MC):
        chosen = rng.choice(len(p["cuts"]), size=pool, replace=False)
        count[top_m(p["init_cut"], chosen, M, rng)] += 1
    values = p["cuts"].ravel()
    weights = np.repeat(count, p["cuts"].shape[1])
    keep = weights > 0
    return values[keep], weights[keep]


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    return mean, float(np.sqrt(np.average((values - mean) ** 2, weights=weights)))


def analyze_problem(p: dict, rand: np.ndarray, evals: list[dict], rng) -> dict:
    bks = float(p["bks"])

    def gap(cut: float) -> float:
        return (bks - cut) / bks * 100

    rand_mean, rand_sd = float(rand.mean()), float(rand.std())
    by_k = {}
    for ev in sorted(evals, key=lambda e: e["pool"]):
        pp = ev["per_problem"][p["name"]]
        values, weights = cim_distribution(p, ev["pool"], rng)
        cim_mean, cim_sd = weighted_stats(values, weights)
        n_rand = pp["runs_per_strategy"]["random_init"]
        by_k[str(ev["pool"])] = {
            "cim_per_run_gap_pct": gap(cim_mean), "cim_sd_cut": cim_sd,
            "cim_best_pred_gap_pct": {str(k): gap(iid_expected_max(values, weights, k)) for k in K_EQUAL},
            "cim_best_actual_gap_pct": gap(pp["strategies"]["cim_cut"]["mean_best"]),
            "random_runs": n_rand,
            "random_best_pred_gap_pct": gap(expected_max(rand, n_rand)),
            "random_best_actual_gap_pct": gap(pp["strategies"]["random_init"]["mean_best"]),
            "total_time_s": pp["total_time_s"],
        }
    return {"bks": bks, "random_pool": int(len(rand)), "random_per_run_gap_pct": gap(rand_mean),
            "random_sd_cut": rand_sd,
            "random_best_pred_gap_pct": {str(k): gap(expected_max(rand, k)) for k in K_EQUAL},
            "by_pool": by_k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--random", type=Path, nargs="+", required=True)
    ap.add_argument("--evals", type=Path, nargs="+", required=True, help="selection_eval の出力ディレクトリ")
    ap.add_argument("--sweeps", type=int, required=True)
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    evals = [json.loads((d / "result.json").read_text(encoding="utf-8")) for d in args.evals]
    bad = [e["pool"] for e in evals if e["sweeps"] != args.sweeps or abs(e["t_start"] - 2.0) > 1e-9 or e["m"] != M]
    if bad:
        ap.error(f"selection_eval results with other settings: pools {bad}")
    rand_pools = load_random_pools(args.random, args.sweeps)
    rng = np.random.default_rng(SEED)
    out = {"sweeps": args.sweeps, "m": M, "mc": MC, "seed": SEED, "runs": [str(d) for d in args.runs],
           "evals": [str(d) for d in args.evals], "per_problem": {}}
    for d in args.runs:
        p = load_problem(d.resolve(), args.sweeps)
        if p["name"] not in args.problems or abs(p["t_start"] - 2.0) > 1e-9:
            continue
        if p["name"] not in rand_pools:
            ap.error(f"no random-start runs for {p['name']}")
        res = analyze_problem(p, rand_pools[p["name"]], evals, rng)
        out["per_problem"][p["name"]] = res
        print(p["name"], f"per-run CIM {res['by_pool']['256']['cim_per_run_gap_pct']:.4f} "
              f"rand {res['random_per_run_gap_pct']:.4f}")
        for pool, r in res["by_pool"].items():
            print(f"  K={pool:>3} CIM best16 pred {r['cim_best_pred_gap_pct']['16']:.4f} act {r['cim_best_actual_gap_pct']:.4f}"
                  f" | rand n={r['random_runs']} pred {r['random_best_pred_gap_pct']:.4f}"
                  f" act {r['random_best_actual_gap_pct']:.4f}")
    missing = set(args.problems) - set(out["per_problem"])
    if missing:
        ap.error(f"no step-3 runs for {sorted(missing)}")

    kind_root = get_kind_root()
    desc = f"S{args.sweeps}_" + "_".join(args.problems) + (f"_{args.tag}" if args.tag else "")
    dst = kind_root / f"v{next_version(kind_root)}_{desc}"
    dst.mkdir(parents=True)
    (dst / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", dst)


if __name__ == "__main__":
    main()
