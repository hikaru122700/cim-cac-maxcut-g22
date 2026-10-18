"""ステップ 5b:CIM の生成条件で、候補がどちらの固有ベクトルの系統に入るかを変えられるか。

G22 で、事前登録 R1 と同じ設定(600 ラウンド、SA 開始温度 2.0・予算 1e7)を基準に、
ポンプの立ち上げの速さ(dP_per_round)と雑音の大きさ(noise_mult)を変える。
ポンプを遅くするときはラウンド数を増やし、最終のポンプ強度(ラウンド数 × dP)を揃える。

候補ごとに a1, a2(A の最小固有値側 2 本への揃い方)、E = a1 - a2、cut、SA 後の平均 cut を測る。
CIM は modules/2026-08-23_CIM_ablation.py(既定の引数で modules/CIM.py とビット一致)を使う。

使い方(プロジェクトルートから):
  python scripts/benchmarks/cim_condition_lineage.py --candidates 256 --repeats 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "24")

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm  # noqa: E402
from scripts.benchmarks.algo_registry import PARAMS, load_context  # noqa: E402
from scripts.benchmarks.seed_value_pilot import (  # noqa: E402
    adjacency, eig_features, greedy_local_opt, low_eigvecs, residual,
)

_spec = importlib.util.spec_from_file_location("cim_ablation", ROOT / "modules" / "2026-08-23_CIM_ablation.py")
CIM_ABL = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CIM_ABL)

EXPERIMENT_KIND = "cim_condition_lineage"
BASE_ROUNDS = 600
CIM_SEED_START = 20_000
SA_SEED_OFFSET = 9_500_000
SA_BUDGET = 10_000_000
SA_T = (2.0, 0.001)
# (名前, ポンプの速さの倍率, 雑音の倍率)
CONDITIONS = [("base", 1.0, 1.0), ("ramp_x0.5", 0.5, 1.0), ("ramp_x2", 2.0, 1.0),
              ("noise_x0.5", 1.0, 0.5), ("noise_x2", 1.0, 2.0), ("noise_x4", 1.0, 4.0)]


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22")
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=4)
    args = ap.parse_args()
    if args.candidates <= 0 or args.repeats <= 0:
        ap.error("candidates and repeats must be positive")

    ctx = load_context(args.dataset)
    p = PARAMS[args.dataset]["CIM"]
    A = adjacency(ctx)
    vals, vecs = low_eigvecs(A)
    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.dataset}_K{args.candidates}_R{args.repeats}"
    out_dir.mkdir(parents=True)

    def log(msg):
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    K, R = args.candidates, args.repeats
    seeds = CIM_SEED_START + np.arange(K, dtype=np.int64)
    arrays, rows = {}, []
    for name, ramp, noise in CONDITIONS:
        rounds = int(round(BASE_ROUNDS / ramp))
        t0 = time.perf_counter()
        _, signs = CIM_ABL.simulate_cim_ablation_batch(
            ctx.n, ctx.J_cim, ctx.edges, rounds, K, p["kappa"], p["L"], p["gamma"], p["eta"],
            p["bandwidth"], p["photon_energy"], p["dP_per_round"] * ramp, seeds,
            weights=ctx.weights, noise_mult=noise,
        )
        cim_s = time.perf_counter() - t0
        s_pm = np.where(np.asarray(signs) > 0, 1, -1).astype(np.int8)
        cut = ctx.score(s_pm).astype(float)
        ls_cut = ctx.score(greedy_local_opt(ctx, s_pm)).astype(float)
        f = eig_features(s_pm, vecs)
        rep = np.repeat(s_pm, R, axis=0)
        sa_seeds = SA_SEED_OFFSET + np.arange(K * R, dtype=np.int64)
        _, out = simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, rep, SA_BUDGET,
                                  t_start=SA_T[0], t_end=SA_T[1], seeds=sa_seeds)
        label = ctx.score(out).reshape(K, R).astype(float).mean(axis=1)
        resid = residual(label, np.column_stack([cut, ls_cut]))
        row = {
            "condition": name, "ramp": ramp, "noise_mult": noise, "rounds": rounds,
            "cim_seconds": cim_s, "cut_mean": float(cut.mean()), "ls_cut_mean": float(ls_cut.mean()),
            "a1_mean": float(f["eig_a1"].mean()), "a2_mean": float(f["eig_a2"].mean()),
            "E_mean": float(f["eig_diff"].mean()), "E_pos_frac": float((f["eig_diff"] > 0).mean()),
            "label_mean": float(label.mean()), "label_sd": float(label.std(ddof=1)),
            "rho_E_resid": float(spearmanr(f["eig_diff"], resid)[0]),
            "rho_E_label": float(spearmanr(f["eig_diff"], label)[0]),
        }
        rows.append(row)
        arrays.update({f"{name}_cut": cut, f"{name}_ls_cut": ls_cut, f"{name}_label": label,
                       **{f"{name}_{k}": v for k, v in f.items()}})
        log(f"{name:>10}: rounds={rounds} cut={row['cut_mean']:.1f} a1={row['a1_mean']:.3f} "
            f"a2={row['a2_mean']:.3f} E>0={row['E_pos_frac']:.2f} SA={row['label_mean']:.1f} "
            f"rho(E,resid)={row['rho_E_resid']:+.2f} ({cim_s:.1f}s CIM)")
    np.savez_compressed(out_dir / "raw.npz", eig_vals=vals, **arrays)
    meta = {"dataset": args.dataset, "candidates": K, "repeats": R, "base_rounds": BASE_ROUNDS,
            "cim_seed_start": CIM_SEED_START, "sa_seed_offset": SA_SEED_OFFSET, "sa_budget": SA_BUDGET,
            "sa_t": SA_T, "conditions": CONDITIONS, "rows": rows}
    (out_dir / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
