"""検証 4 の事後の診断(判定には使わない):初期集団の距離の構造を条件ごとに比べる。

ga_equal_time.py の本実行と同じ seed で CIM 解を作り直し(結果のカット値が一致することを確かめる)、
各条件の初期集団について、平均ペア距離・最近接距離(各個体の最も近い相手との距離の平均)を求める。

出力:results/<日付>/ga_equal_time_diag/v{N}_<説明>/diag.json

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_equal_time_diag.py --run results/2026-10-19/ga_equal_time/v2_G22_G55_G70_nt16_k4_eval300_main --datasets G55 G70
"""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.algo_registry import load_context, run_cim  # noqa: E402
from scripts.benchmarks.ga_equal_time import pairwise_overlap  # noqa: E402

EXPERIMENT_KIND = "ga_equal_time_diag"


def get_out_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 1 + max([int(p.name.split("_")[0][1:]) for p in root.iterdir()
                 if p.is_dir() and p.name.split("_")[0][1:].isdigit()], default=0)
    out = root / f"v{v}_{desc}"
    out.mkdir(parents=True)
    return out


def structure(signs: np.ndarray) -> dict:
    d = (1 - pairwise_overlap(signs)) / 2
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    np.fill_diagonal(d, np.nan)
    return {"mean_pairwise": float(np.nanmean(d)), "mean_nearest": float(nearest.mean()),
            "min_pairwise": float(nearest.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    meta = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))
    out = get_out_dir("_".join(args.datasets))
    res = {"run": str(args.run), "datasets": {}}
    for ds in args.datasets:
        info = meta["datasets"][ds]
        z = np.load(args.run / f"{ds}.npz")
        ctx = load_context(ds)
        T, K = z["cim_cuts"].shape
        cuts, signs = run_cim(ctx, info["cim_rounds"], T * K, z["cim_seeds"])
        if not np.array_equal(cuts.reshape(T, K), z["cim_cuts"]):
            raise SystemExit(f"{ds}: regenerated CIM cuts differ from the main run")
        signs = signs.reshape(T, K, ctx.n)
        row = {}
        for cond in ["cim_all", "cim_cut", "cim_div"]:
            picks = z[f"{cond}_picks"]
            s = [structure(signs[t, picks[t]]) for t in range(T)]
            row[cond] = {k: float(np.mean([x[k] for x in s])) for k in s[0]}
        row["all_K"] = {k: float(np.mean([structure(signs[t])[k] for t in range(T)])) for k in row["cim_all"]}
        res["datasets"][ds] = row
        print(ds, json.dumps(row, ensure_ascii=False), flush=True)
    (out / "diag.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
