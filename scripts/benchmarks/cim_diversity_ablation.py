"""cim_diversity_ablation.py — CIM の解の多様性が低い原因を 4 実験で切り分ける。

背景: results/2026-08-18/solution_diversity/v2_* で、CIM は「品質あたりの多様性」
が他手法より明確に劣る(G70 では SA/SB に品質・多様性の両軸で支配される)。
その原因候補 4 つを、baseline とビット一致する ablation 実装
(`modules/2026-08-23_CIM_ablation.py`)で 1 つずつ潰す。

  実験1 init  : 初期振幅をランダム化(baseline は c(0)=0 で全試行同一)
                → 予測: ほぼ変化なし。round 1 で c ← ノイズ に上書きされるため。
  実験2 ramp  : ポンプ ramp 速度 dP_per_round を掃引
                → 予測: 速いほど多様性↑・品質↓。線形増幅段(固有モード競合)が短くなる。
  実験3 noise : 真空雑音の振幅を定数倍
                → 予測: 大きいほど多様性↑。飽和後もノイズが符号を反転できるようになる。
  実験4 async : 同時更新をやめる(部分更新 / 逐次更新)
                → 予測: 多様性↑。更新順序という対称性の破れ源が復活する。

公平性のための制御:
  - 実験4 で更新頂点を割合 f に減らすときは round 数を 1/f 倍し、同時に
    dP_per_round を f 倍する。頂点更新回数とポンプ ramp の「物理的な速さ」の
    両方を baseline に揃えるため(round 数だけ増やすと ramp まで変わってしまう)。
  - baseline の round 数は results/2026-08-18/solution_diversity/v2_* の
    CIM 採用予算(G22 4800 / G55 9600 / G70 4800)に固定し、過去 run と直接比較できるようにする。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe -u scripts/benchmarks/cim_diversity_ablation.py \
      --datasets G22 G55 G70 --num-trials 32 --tag main
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, PARAMS, load_context
from scripts.benchmarks.diversity_metrics import summarize

EXPERIMENT_KIND = "cim_diversity_ablation"

# 日付始まりのモジュールは import 不可 → importlib でロード
_spec = importlib.util.spec_from_file_location(
    "cim_ablation", ROOT / "modules" / "2026-08-23_CIM_ablation.py"
)
CIM_ABL = importlib.util.module_from_spec(_spec)
# numba cache=True のデシリアライズは __module__ 名で import し直すので、
# 先に sys.modules へ登録しておかないと ModuleNotFoundError('<dynamic>') になる。
sys.modules["cim_ablation"] = CIM_ABL
_spec.loader.exec_module(CIM_ABL)

# v2 diversity run で CIM が採用した予算(実時間マッチ済み)
BASE_ROUNDS = {"G22": 4800, "G55": 9600, "G70": 4800, "K2000": 300}


# ============================================================
#  掃引条件の定義
# ============================================================
def build_configs(experiments: list[int]) -> list[dict]:
    """(実験番号, ラベル, ノブ) のリストを作る。baseline は先頭に 1 つだけ。"""
    cfgs: list[dict] = [
        {"exp": 0, "key": "baseline", "label": "baseline",
         "knobs": {}, "dp_mult": 1.0, "rounds_mult": 1.0}
    ]
    if 1 in experiments:
        for s in (1e-5, 1e-3, 1e-1):
            cfgs.append({"exp": 1, "key": f"init{s:g}", "label": f"初期振幅 {s:g}",
                         "knobs": {"init_scale": s}, "dp_mult": 1.0, "rounds_mult": 1.0})
    if 2 in experiments:
        for m in (0.25, 4.0, 16.0, 64.0):
            cfgs.append({"exp": 2, "key": f"dp{m:g}x", "label": f"ramp {m:g}x",
                         "knobs": {}, "dp_mult": m, "rounds_mult": 1.0})
    if 3 in experiments:
        for m in (10.0, 100.0, 1000.0):
            cfgs.append({"exp": 3, "key": f"noise{m:g}x", "label": f"ノイズ {m:g}x",
                         "knobs": {"noise_mult": m}, "dp_mult": 1.0, "rounds_mult": 1.0})
    if 4 in experiments:
        for f, seq in ((0.5, False), (0.25, False), (0.5, True), (0.25, True)):
            tag = "逐次" if seq else "部分"
            cfgs.append({
                "exp": 4, "key": f"async{f:g}{'seq' if seq else 'par'}",
                "label": f"{tag}更新 f={f:g}",
                "knobs": {"async_frac": f, "seq_update": seq},
                # 頂点更新回数とポンプ ramp を baseline に揃える
                "dp_mult": f, "rounds_mult": 1.0 / f,
            })
    return cfgs


# ============================================================
#  出力先(results 規約: 日付 / 実験種別 / v{N}_説明)
# ============================================================
def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    mx = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            h = p.name.split("_", 1)[0]
            if h[1:].isdigit():
                mx = max(mx, int(h[1:]))
    return mx + 1


def build_description(args) -> str:
    parts = ["_".join(args.datasets), f"nt{args.num_trials}",
             "ex" + "".join(str(e) for e in args.experiments)]
    if args.base_rounds:
        parts.append(f"r{args.base_rounds}")
    if args.seed_offset:
        parts.append(f"seed{args.seed_offset}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


# ============================================================
#  1 条件の実行
# ============================================================
def run_config(ctx, cfg: dict, num_trials: int, seeds: np.ndarray, base_rounds: int):
    """1 条件を num_trials 本走らせて (signs, cuts, rounds, dP, elapsed) を返す。"""
    p = PARAMS[ctx.name]["CIM"]
    rounds = int(round(base_rounds * cfg["rounds_mult"]))
    dP = p["dP_per_round"] * cfg["dp_mult"]

    t0 = time.perf_counter()
    _cuts, signs = CIM_ABL.simulate_cim_ablation_batch(
        ctx.n, ctx.J_cim, ctx.edges, rounds, num_trials,
        p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
        p["photon_energy"], dP, seeds, weights=ctx.weights, **cfg["knobs"],
    )
    dt = time.perf_counter() - t0
    S = np.asarray(signs) > 0
    cuts = np.asarray(ctx.score(S), dtype=float)   # 全条件を同一基準で採点
    return S, cuts, rounds, dP, dt


SUMMARY_COLS = [
    "dataset", "exp", "config", "label", "rounds", "dP_per_round", "time",
    "threshold_round", "threshold_frac",
    "cut_mean", "cut_max", "cut_std", "gap_pct_mean", "gap_pct_best",
    "mean_pairwise", "median_pairwise", "min_pairwise", "max_pairwise",
    "n_distinct", "n_distinct_cuts", "frozen_frac", "entropy_bits",
    "n_near", "near_mean_pairwise", "near_n_distinct",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["G22"], choices=list(DATASETS))
    ap.add_argument("--num-trials", type=int, default=32)
    ap.add_argument("--experiments", nargs="+", type=int, default=[1, 2, 3, 4],
                    choices=[0, 1, 2, 3, 4],
                    help="0 を指定すると baseline のみ(シード変動の誤差棒取り用)。")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="乱数シードのオフセット。別ブロックで走らせて再現性を見る。")
    ap.add_argument("--near-gap-pct", type=float, default=0.5)
    ap.add_argument("--base-rounds", type=int, default=None,
                    help="baseline の round 数を上書き(smoke test 用)。")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    kind_root = get_kind_root()
    v = next_version(kind_root)
    out_dir = kind_root / f"v{v}_{build_description(args)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logf = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfgs = build_configs(args.experiments)
    log(f"out_dir = {out_dir}")
    log(f"datasets={args.datasets} num_trials={args.num_trials} "
        f"experiments={args.experiments} 条件数={len(cfgs)}")

    seeds = np.arange(args.num_trials, dtype=np.int64) + args.seed_offset
    arrays: dict[str, np.ndarray] = {}
    results: dict[str, dict] = {}
    rows: list[dict] = []

    for ds in args.datasets:
        ctx = load_context(ds)
        p = PARAMS[ds]["CIM"]
        base_rounds = args.base_rounds or BASE_ROUNDS[ds]
        log(f"\n=== {ds} (n={ctx.n}, BKS={ctx.bks}, baseline {base_rounds} rounds) ===")

        # JIT ウォームアップ(全分岐を踏ませる。計測しない)
        for warm in ({}, {"init_scale": 1e-3}, {"async_frac": 0.5},
                     {"async_frac": 0.5, "seq_update": True}):
            CIM_ABL.simulate_cim_ablation_batch(
                ctx.n, ctx.J_cim, ctx.edges, 5, min(args.num_trials, 4),
                p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
                p["photon_energy"], p["dP_per_round"], seeds[:min(args.num_trials, 4)],
                weights=ctx.weights, **warm)

        results[ds] = {}
        for cfg in cfgs:
            S, cuts, rounds, dP, dt = run_config(
                ctx, cfg, args.num_trials, seeds, base_rounds)
            k_th = CIM_ABL.threshold_round(p["kappa"], p["L"], p["eta"], dP)

            summ = summarize(S, cuts, ctx.bks, near_gap_pct=args.near_gap_pct)
            summ.update({
                "exp": cfg["exp"], "config": cfg["key"], "label": cfg["label"],
                "rounds": rounds, "budget": float(rounds),
                "dP_per_round": dP, "time": dt,
                "threshold_round": float(k_th),
                "threshold_frac": float(min(k_th / rounds, 1.0)),
            })
            results[ds][cfg["key"]] = summ
            arrays[f"{ds}__{cfg['key']}_signs"] = S
            arrays[f"{ds}__{cfg['key']}_cuts"] = cuts
            rows.append({"dataset": ds, **{c: summ.get(c) for c in SUMMARY_COLS[1:]}})

            log(f"  ex{cfg['exp']} {cfg['key']:<14s} rounds={rounds:>6d} "
                f"t={dt:6.1f}s  gap平均={summ['gap_pct_mean']:7.3f}%  "
                f"平均距離={summ['mean_pairwise']:.4f}  相異なる解="
                f"{summ['n_distinct']:>2d}/{summ['num_trials']}  "
                f"固定頂点={summ['frozen_frac']*100:5.2f}%  "
                f"エントロピー={summ['entropy_bits']:.3f}bit  "
                f"しきい値round={k_th:.0f}({summ['threshold_frac']*100:.0f}%)")

    np.savez_compressed(out_dir / "signs.npz", **arrays)

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "datasets": args.datasets, "num_trials": args.num_trials,
        # diversity_refine.py は meta["algos"] を回すので条件キーを同名で置く
        "algos": [c["key"] for c in cfgs],
        "experiments": args.experiments, "near_gap_pct": args.near_gap_pct,
        "tag": args.tag, "seed_offset": args.seed_offset, "base_rounds": {d: (args.base_rounds or BASE_ROUNDS[d]) for d in args.datasets},
        "bks": {d: DATASETS[d]["bks"] for d in args.datasets},
        "cim_params": {d: PARAMS[d]["CIM"] for d in args.datasets},
        "configs": cfgs,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, ensure_ascii=False, indent=1)

    log(f"\nsaved -> {out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
