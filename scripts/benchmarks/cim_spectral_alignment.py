"""cim_spectral_alignment.py — 解が「J のトップ固有ベクトルの符号」にどれだけ寄っているか測る。

仮説(b)の直接検証。CIM は飽和前 $c_{k+1} \\approx e^{g_0/2}(\\sqrt{\\eta}I + J)c_k + \\xi$ という
線形写像なので、しきい値近傍では $\\sqrt{\\eta}+\\lambda$ が最大のモード、つまり **J のトップ
固有ベクトル $v_1$** が最初に不安定化して独り勝ちで成長する($\\sqrt{\\eta}I$ は全モードを
同じだけ持ち上げるので、順位は J の固有値だけで決まる)。$v_1$ は J だけで決まり試行に
依らないため、これが符号を決めているなら全試行が同じ骨格に漏斗のように吸い込まれる。

測る量(反転対称を畳んだ、値域 [0, 1]):

    align(s, v1) = | 1 - 2 * hamming(sign(s), sign(v1)) / n |

  1.0 に近い … 解が spectral 緩和解の符号とほぼ一致(＝漏斗にはまっている)
  0.0 に近い … 無相関

使い方(プロジェクトルートから):
    .venv/Scripts/python.exe -u scripts/benchmarks/cim_spectral_alignment.py \\
        results/<date>/cim_diversity_ablation/v{N}_... \\
        results/2026-08-18/solution_diversity/v2_...

複数の run ディレクトリを渡すと、それぞれについて alignment を計算して 1 つの
CSV / JSON にまとめる(第 1 引数のディレクトリに出力)。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse.linalg import eigsh

from scripts.benchmarks.algo_registry import load_context


def top_eigvec(J) -> tuple[np.ndarray, float]:
    """J の最大固有値とその固有ベクトル(CIM が最初に増幅するモード)。"""
    vals, vecs = eigsh(J.astype(np.float64), k=1, which="LA")
    return np.asarray(vecs[:, 0]).ravel(), float(vals[0])


def alignment(S: np.ndarray, v: np.ndarray) -> np.ndarray:
    """各試行の解と参照符号ベクトル v の重なり(反転対称を畳んで [0,1])。"""
    a = (np.asarray(S) > 0).astype(np.int8)
    b = (np.asarray(v) > 0).astype(np.int8)
    frac = (a != b).mean(axis=1)
    return np.abs(1.0 - 2.0 * frac)


def collect(run_dir: Path) -> list[dict]:
    """1 つの run ディレクトリの signs.npz を走査して alignment を計算する。"""
    with open(run_dir / "results.json", encoding="utf-8") as f:
        payload = json.load(f)
    meta, results = payload["meta"], payload["results"]
    npz = np.load(run_dir / "signs.npz")

    rows: list[dict] = []
    for ds in meta["datasets"]:
        if ds not in results or not results[ds]:
            continue
        ctx = load_context(ds)
        v1, lam1 = top_eigvec(ctx.J_cim)
        spec_cut = float(np.asarray(ctx.score((v1 > 0)[None, :]))[0])

        for key in results[ds]:
            arr = f"{ds}__{key}_signs"
            if arr not in npz:
                continue
            S = npz[arr]
            al = alignment(S, v1)
            summ = results[ds][key]
            rows.append({
                "run": run_dir.name,
                "dataset": ds,
                "config": key,
                "label": summ.get("label", key),
                "exp": summ.get("exp", ""),
                "lambda_max": lam1,
                "spectral_cut": spec_cut,
                "spectral_gap_pct": (ctx.bks - spec_cut) / ctx.bks * 100.0,
                "align_mean": float(al.mean()),
                "align_std": float(al.std()),
                "align_max": float(al.max()),
                "gap_pct_mean": summ.get("gap_pct_mean"),
                "mean_pairwise": summ.get("mean_pairwise"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--out-name", default="spectral_alignment")
    args = ap.parse_args()

    rows: list[dict] = []
    for d in args.run_dirs:
        print(f"--- {d} ---", flush=True)
        got = collect(d)
        for r in got:
            print(f"  {r['dataset']:6s} {r['config']:<14s} "
                  f"固有ベクトル一致度={r['align_mean']:.4f}±{r['align_std']:.4f} "
                  f"(spectral 解の gap={r['spectral_gap_pct']:.2f}%)  "
                  f"gap={r['gap_pct_mean']:.3f}%  距離={r['mean_pairwise']:.4f}",
                  flush=True)
        rows.extend(got)

    out_dir = args.run_dirs[0]
    cols = list(rows[0].keys())
    with open(out_dir / f"{args.out_name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / f"{args.out_name}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {out_dir / (args.out_name + '.csv')}")


if __name__ == "__main__":
    main()
