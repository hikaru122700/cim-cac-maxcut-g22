"""selection_eval の「同じ総時間」の比較に、候補を単位にしたブートストラップ区間を付ける。

selection_eval.py の標準誤差は 4000 回の抜き出しを独立とみなしているが、抜き出しは同じ 512 個の
CIM 候補(各 16 回の SA)と、同じランダム開始 SA の結果を使い回している。そこで
  CIM 側   : 512 個の候補を復元抽出し(候補ごとの init_cut と 16 回の SA 結果をまとめて)、
             「256 個を選ぶ → init_cut の上位 16 個 → 各々の SA 結果を 1 つ引く → 最大値」を MC 回
  ランダム側: ランダム開始 SA の結果を復元抽出し、そこから n 本を重複なしで選んだときの最大値の期待値
             (順序統計量で厳密に計算)
を B 回くり返し、平均の 95% 区間と、差(CIM − ランダム)の区間を出す。新しい SA は回さない。
本数 n と総時間は selection_eval の result.json の値をそのまま使う。

使い方(プロジェクトルートから):
  python scripts/benchmarks/selection_eval_bootstrap.py \
      --result results/2026-10-19/selection_eval/v29_sub_T2_S5000_K256_m16/result.json \
      --runs results/2026-10-18/seed_value_pilot/v*_K512_*_S3 --random results/2026-10-18/random_init_sa/v*
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.special import gammaln

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.selection_eval import RANDOM_T_START, load_problem  # noqa: E402

EXPERIMENT_KIND = "selection_eval_bootstrap"
B = 400  # ブートストラップの回数
MC = 1000  # 1 回のブートストラップ内での抜き出し回数
SEED = 20261107
TIE = 1e-6  # これより小さい差は引き分け(期待値の計算の丸め誤差)


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


def expected_max(values: np.ndarray, n: int) -> float:
    """values から n 個を重複なしで選んだときの最大値の期待値(厳密)。"""
    v = np.sort(values)
    total = len(v)
    if not 1 <= n <= total:
        raise ValueError(f"need 1 <= n <= {total}, got {n}")
    k = np.arange(total + 1)
    log_c = np.full(total + 1, -np.inf)
    ok = k >= n
    log_c[ok] = (gammaln(k[ok] + 1) - gammaln(k[ok] - n + 1)) - (gammaln(total + 1) - gammaln(total - n + 1))
    cdf = np.exp(log_c)  # P(最大値 <= k 番目に小さい値)
    return float((np.diff(cdf) * v).sum())


def cim_mean_best(init_cut: np.ndarray, cuts: np.ndarray, pool: int, m: int, rng) -> float:
    """「pool 個を選ぶ → init_cut の上位 m 個 → 各々の SA 結果を 1 つ引く → 最大値」の平均(MC 回)。
    同点は 1 回のブートストラップの中で 1 度だけ乱数で崩す。"""
    k_all, r = cuts.shape
    order = np.lexsort((rng.random(k_all), -init_cut))
    sorted_cuts = cuts[order]
    keys = rng.random((MC, k_all))
    in_pool = keys <= np.partition(keys, pool - 1, axis=1)[:, pool - 1:pool]  # 各行でちょうど pool 個
    chosen = in_pool & (np.cumsum(in_pool, axis=1) <= m)
    draws = sorted_cuts[np.arange(k_all)[None, :], rng.integers(0, r, (MC, k_all))]
    return float(np.where(chosen, draws, -np.inf).max(axis=1).mean())


def load_random_pools(dirs: list[Path], sweeps: int) -> dict[str, np.ndarray]:
    """selection_eval と同じ規則:開始温度 2.0、同じ問題が複数あれば本数の多いもの。"""
    pools: dict[str, np.ndarray] = {}
    key = f"cuts_S{sweeps}"
    for d in dirs:
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        if abs(meta["t_start"] - RANDOM_T_START) > 1e-9:
            continue
        z = np.load(d / "raw.npz")
        if key in z.files and len(z[key]) > len(pools.get(meta["dataset"], [])):
            pools[meta["dataset"]] = z[key].astype(float)
    return pools


def interval(x: np.ndarray) -> list[float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def bootstrap_problem(p: dict, rand: np.ndarray, n_rand: int, pool: int, m: int, rng) -> dict:
    bks = float(p["bks"])
    k_all = len(p["init_cut"])
    point_cim = cim_mean_best(p["init_cut"], p["cuts"], pool, m, rng)
    point_rand = expected_max(rand, n_rand)
    cim_b, rand_b = np.empty(B), np.empty(B)
    for b in range(B):
        idx = rng.integers(0, k_all, k_all)
        cim_b[b] = cim_mean_best(p["init_cut"][idx], p["cuts"][idx], pool, m, rng)
        rand_b[b] = expected_max(rand[rng.integers(0, len(rand), len(rand))], n_rand)
    diff = cim_b - rand_b

    def gap(x):
        return (bks - x) / bks * 100

    return {
        "bks": bks, "n_candidates": k_all, "sa_per_candidate": int(p["cuts"].shape[1]),
        "random_pool": int(len(rand)), "random_runs": int(n_rand),
        "cim_gap_pct": gap(point_cim), "cim_gap_pct_ci95": sorted(gap(np.array(interval(cim_b)))),
        "random_gap_pct": gap(point_rand), "random_gap_pct_ci95": sorted(gap(np.array(interval(rand_b)))),
        "diff_cut": point_cim - point_rand, "diff_cut_ci95": interval(diff),
        "frac_cim_better": float((diff > TIE).mean()), "frac_tie": float((np.abs(diff) <= TIE).mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True, help="selection_eval の result.json")
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="ステップ 3 の出力(seed_value_pilot)")
    ap.add_argument("--random", type=Path, nargs="+", required=True, help="random_init_sa の出力")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.tag and not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    res = json.loads(args.result.read_text(encoding="utf-8"))
    t_start, sweeps, pool, m = res["t_start"], res["sweeps"], res["pool"], res["m"]
    problems = {}
    for d in args.runs:
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))["meta"]
        if abs(meta["t_start"] - t_start) < 1e-9 and meta["dataset"] in res["per_problem"]:
            problems[meta["dataset"]] = load_problem(d.resolve(), sweeps)
    missing = sorted(set(res["per_problem"]) - set(problems))
    if missing:
        ap.error(f"no step-3 run for {missing} at t_start={t_start}")
    rand = load_random_pools(args.random, sweeps)

    rng = np.random.default_rng(SEED)
    out = {}
    for name, pp in res["per_problem"].items():
        n_rand = pp["runs_per_strategy"]["random_init"]
        out[name] = bootstrap_problem(problems[name], rand[name], n_rand, pool, m, rng)
        o = out[name]
        print(f"{name:>4} CIM {o['cim_gap_pct']:.4f}% {np.round(o['cim_gap_pct_ci95'], 4).tolist()} | "
              f"rand {o['random_gap_pct']:.4f}% {np.round(o['random_gap_pct_ci95'], 4).tolist()} | "
              f"diff {o['diff_cut']:+.2f} {np.round(o['diff_cut_ci95'], 2).tolist()} "
              f"P(CIM>rand)={o['frac_cim_better']:.3f}", flush=True)

    kind_root = get_kind_root()
    desc = f"T{t_start:g}_S{sweeps}_K{pool}_m{m}".replace(".", "p") + (f"_{args.tag}" if args.tag else "")
    out_dir = kind_root / f"v{next_version(kind_root)}_{desc}"
    out_dir.mkdir(parents=True)
    summary = {"source": str(args.result), "t_start": t_start, "sweeps": sweeps, "pool": pool, "m": m,
               "bootstrap": B, "mc_per_bootstrap": MC, "seed": SEED, "per_problem": out}
    (out_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
