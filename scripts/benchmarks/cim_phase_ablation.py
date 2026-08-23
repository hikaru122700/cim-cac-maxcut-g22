"""cim_phase_ablation.py — CIM の 3 段階のうち、どれが解を作っているのか。

CIM の 1 run はふつう 3 つの局面に分かれる:

  第1段階「パタパタ期」 … しきい値下。振幅は小さく、符号は毎 round ランダムに入れ替わる。
  第2段階「形成期」     … 振幅が指数的に立ち上がり、符号が確定していく。
  第3段階「飽和期」     … 振幅が飽和し、符号は動かない。

本スクリプトは 2 つの問いに答える。

  A. 第3段階は要るか?
     軌跡診断で「round k まで走らせたときの出力」= 累積最良カット best_cut(k) を記録し、
     どの round で頭打ちになるかを見る。**k round で打ち切ることは best_cut(k) を読むことと
     厳密に同値**なので、追加の run を回さずに打ち切りの効果が分かる。

  B. 第1段階は要るか?
     pump_offset ノブでポンプを round K0 の水準から始め、残り (rounds - K0) round だけ走らせる。
     ポンプ列は K0 以降 baseline と完全に一致し、終端も同じ。第1段階だけが消える。
     出力(品質・多様性)が変わらなければ、第1段階は解の形成に寄与していない。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe -u scripts/benchmarks/cim_phase_ablation.py \
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

EXPERIMENT_KIND = "cim_phase_ablation"

_spec = importlib.util.spec_from_file_location(
    "cim_ablation", ROOT / "modules" / "2026-08-23_CIM_ablation.py")
CIM_ABL = importlib.util.module_from_spec(_spec)
sys.modules["cim_ablation"] = CIM_ABL
_spec.loader.exec_module(CIM_ABL)

BASE_ROUNDS = {"G22": 4800, "G55": 9600, "G70": 4800, "K2000": 300}
# 第1段階をどこまで削るか(baseline round 数に対する割合)
OFFSET_FRACS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50]


def get_out_dir(desc: str) -> Path:
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    mx = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            h = p.name.split("_", 1)[0]
            if h[1:].isdigit():
                mx = max(mx, int(h[1:]))
    out = kind_root / f"v{mx + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plateau_round(best: np.ndarray, rounds: np.ndarray, tol: float) -> int:
    """best_cut が最終値の (1 - tol) 倍以上に達する最初の round。"""
    final = best[-1]
    idx = int(np.argmax(best >= final * (1.0 - tol)))
    return int(rounds[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["G22"], choices=list(DATASETS))
    ap.add_argument("--num-trials", type=int, default=32)
    ap.add_argument("--traj-trials", type=int, default=8)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--near-gap-pct", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    desc = "_".join(["_".join(args.datasets), f"nt{args.num_trials}"]
                    + ([args.tag] if args.tag else []))
    out_dir = get_out_dir(desc)
    logf = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"out_dir = {out_dir}")
    seeds = np.arange(args.num_trials, dtype=np.int64)
    traj_payload: dict = {}
    offset_results: dict = {}
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for ds in args.datasets:
        ctx = load_context(ds)
        p = PARAMS[ds]["CIM"]
        rounds = BASE_ROUNDS[ds]
        log(f"\n=== {ds} (n={ctx.n}, BKS={ctx.bks}, {rounds} rounds) ===")

        # ---------- A. 軌跡診断(3 段階の境界と best_cut の頭打ち) ----------
        nrm, cut, best, flip, rr = CIM_ABL.probe_cim_trajectory(
            ctx.n, ctx.J_cim, ctx.edges, rounds, args.traj_trials,
            p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
            p["photon_energy"], p["dP_per_round"],
            seeds[:args.traj_trials], weights=ctx.weights, stride=args.stride)
        nm, cm, bm, fm = (a.mean(axis=0) for a in (nrm, cut, best, flip))

        k_final = plateau_round(bm, rr, 0.0)          # 最終値に到達する round
        k_999 = plateau_round(bm, rr, 0.001)          # 最終値の 99.9%
        k_99 = plateau_round(bm, rr, 0.01)
        # 第1段階の終わり: 符号の入れ替わり率が半分に落ちる round
        f0 = fm[:5].mean()
        k_flip_half = int(rr[int(np.argmax(fm <= 0.5 * f0))]) if (fm <= 0.5 * f0).any() else -1
        k_norm10 = int(rr[int(np.argmax(nm >= 0.10 * nm[-1]))])
        k_norm90 = int(rr[int(np.argmax(nm >= 0.90 * nm[-1]))])

        log(f"  [A] 軌跡 ({args.traj_trials} 試行平均)")
        log(f"      パタパタ率 初期={f0:.4f} → 最終={fm[-1]:.6f}"
            f"  半減 round={k_flip_half}")
        log(f"      ノルム 10%到達={k_norm10}  90%到達={k_norm90}")
        log(f"      best_cut 99%到達={k_99}  99.9%到達={k_999}  最終値到達={k_final}"
            f"  → 残り {rounds - k_final} round ({(rounds-k_final)/rounds*100:.0f}%) は出力に無寄与")
        traj_payload[ds] = {
            "rounds": rr.tolist(), "norm": nm.tolist(), "cut": cm.tolist(),
            "best_cut": bm.tolist(), "flip_rate": fm.tolist(),
            "k_final": k_final, "k_999": k_999, "k_99": k_99,
            "k_flip_half": k_flip_half, "k_norm10": k_norm10, "k_norm90": k_norm90,
            "total_rounds": rounds, "bks": ctx.bks,
        }

        # ---------- B. 第1段階を削る(pump_offset) ----------
        log(f"  [B] 第1段階の切り落とし ({args.num_trials} 試行)")
        offset_results[ds] = {}
        for warm in (0, 10):
            CIM_ABL.simulate_cim_ablation_batch(
                ctx.n, ctx.J_cim, ctx.edges, 5, 4, p["kappa"], p["L"], p["gamma"],
                p["eta"], p["bandwidth"], p["photon_energy"], p["dP_per_round"],
                seeds[:4], weights=ctx.weights, pump_offset=warm)

        for frac in OFFSET_FRACS:
            k0 = int(round(rounds * frac))
            t0 = time.perf_counter()
            _c, signs = CIM_ABL.simulate_cim_ablation_batch(
                ctx.n, ctx.J_cim, ctx.edges, rounds - k0, args.num_trials,
                p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
                p["photon_energy"], p["dP_per_round"], seeds,
                weights=ctx.weights, pump_offset=k0)
            dt = time.perf_counter() - t0
            S = np.asarray(signs) > 0
            cuts = np.asarray(ctx.score(S), dtype=float)
            summ = summarize(S, cuts, ctx.bks, near_gap_pct=args.near_gap_pct)
            key = f"off{k0}"
            summ.update({"pump_offset": k0, "offset_frac": frac,
                         "rounds": rounds - k0, "time": dt})
            offset_results[ds][key] = summ
            arrays[f"{ds}__{key}_signs"] = S
            arrays[f"{ds}__{key}_cuts"] = cuts
            rows.append({"dataset": ds, "config": key, "pump_offset": k0,
                         "offset_frac": frac, "rounds": rounds - k0,
                         "time": dt, "gap_pct_mean": summ["gap_pct_mean"],
                         "gap_pct_best": summ["gap_pct_best"],
                         "cut_mean": summ["cut_mean"], "cut_max": summ["cut_max"],
                         "mean_pairwise": summ["mean_pairwise"],
                         "n_distinct": summ["n_distinct"],
                         "frozen_frac": summ["frozen_frac"],
                         "entropy_bits": summ["entropy_bits"]})
            log(f"      offset={k0:>5} ({frac*100:>4.1f}%) rounds={rounds-k0:>6}"
                f" t={dt:5.1f}s  gap平均={summ['gap_pct_mean']:7.3f}%"
                f"  平均距離={summ['mean_pairwise']:.4f}")

    np.savez_compressed(out_dir / "signs.npz", **arrays)
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {"datasets": args.datasets,
                            "num_trials": args.num_trials,
                            "traj_trials": args.traj_trials,
                            "stride": args.stride,
                            "offset_fracs": OFFSET_FRACS,
                            "base_rounds": {d: BASE_ROUNDS[d] for d in args.datasets},
                            "bks": {d: DATASETS[d]["bks"] for d in args.datasets}},
                   "trajectory": traj_payload,
                   "offset": offset_results}, f, ensure_ascii=False, indent=1)
    log(f"\nsaved -> {out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
