"""
G-Set の複数インスタンスで CIM / CAC / SA を回して集計する。

ポイント:
  - wandb は offline スタブに差し替え (ベンチマーク中はログ不要)
  - 各メソッドを同一グラフで同一試行数ずつ回し、mean / best / std / 時間 を集計
  - 結果は標準出力のテーブル + results/benchmark_gset.csv に保存
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
import sys
import time
import math
import random
from dataclasses import dataclass

# --- wandb を軽量スタブ化してから CIM/CAC/SA をインポート ---
class _WandbStub:
    def init(self, *a, **k):
        class _Cfg:
            def __init__(self, d):
                self._d = d
                for kk, vv in d.items():
                    setattr(self, kk, vv)
            def update(self, d, **kw):
                for kk, vv in d.items():
                    setattr(self, kk, vv)
                    self._d[kk] = vv
        self.config = _Cfg(k.get("config", {}))
        self.summary = {}
        return self
    def log(self, *a, **k): pass
    def finish(self, *a, **k): pass
    def Histogram(self, x): return x
    def Table(self, **k): return k
sys.modules.setdefault("wandb", _WandbStub())
import wandb  # noqa: E402  (スタブを読み込む)

import numpy as np  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch  # noqa: E402
from modules.CAC import simulate_cac_batch, compute_gset_parameters  # noqa: E402
from modules.SA import compute_delta  # noqa: E402
from modules.verify import compute_cut_from_edges  # noqa: E402


# 既知最良解 (BKS) — 公開リソース由来
BKS = {
    "G1": 11624, "G14": 3064, "G15": 3050,
    "G22": 13359, "G23": 13344, "G32": 1410, "G39": 2408,
    "G55": 10299, "G70": 9595, "G77": 9926, "G81": 14030,
}


@dataclass
class MethodResult:
    method: str
    graph: str
    n_trials: int
    mean: float
    best: int
    worst: int
    std: float
    wall_s: float


def run_cim(n, edges, num_trials, num_rounds, seed_base=0):
    J = build_coupling_matrix(n, edges, coupling=-0.03)
    seeds = np.arange(seed_base, seed_base + num_trials, dtype=np.int64)
    eta = 10.0 ** (-11.0 / 10.0)
    t0 = time.time()
    best_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges,
        num_rounds=num_rounds, num_trials=num_trials,
        kappa=130.0, L=0.05, gamma=42.09, eta=eta,
        bandwidth=1.0e9, photon_energy=1.28e-19, dP_per_round=0.05e-3,
        seeds=seeds,
    )
    return np.asarray(best_cuts, dtype=np.int64), time.time() - t0


def run_cac(n, edges, num_trials, num_outer_steps, seed_base=0):
    J = build_coupling_matrix(n, edges, coupling=-1.0)
    gp = compute_gset_parameters(J, n)
    seeds = np.arange(seed_base, seed_base + num_trials, dtype=np.int64)
    t0 = time.time()
    best_cuts, _ = simulate_cac_batch(
        n=n, J=J, edges=edges,
        num_outer_steps=num_outer_steps, num_trials=num_trials,
        p=gp["p"], alpha=gp["alpha"], rho=gp["rho"], delta=gp["delta"],
        beta0_error=gp["beta0_error"], gamma_growth=gp["gamma_growth"],
        tau=gp["tau"],
        n_x_inner=gp["n_x_inner"], n_e_inner=gp["n_e_inner"],
        dt_x=gp["dt_x"], dt_e=gp["dt_e"], e_max=gp["e_max"],
        seeds=seeds,
    )
    return np.asarray(best_cuts, dtype=np.int64), time.time() - t0


def run_sa_single(n, adj, edges, time_limit, seed):
    rnd = random.Random(seed)
    x = [rnd.randint(0, 1) for _ in range(n)]
    current = compute_cut_from_edges(x, edges)
    best = current
    t0 = time.time()
    t_start_T, t_end_T = 2.0, 0.001
    while True:
        elapsed = time.time() - t0
        if elapsed >= time_limit:
            break
        progress = elapsed / time_limit
        T = t_start_T * ((t_end_T / t_start_T) ** progress)
        v = rnd.randint(0, n - 1)
        d = compute_delta(x, adj, v)
        if d > 0:
            x[v] ^= 1; current += d
        elif T > 0:
            if rnd.random() < math.exp(d / T):
                x[v] ^= 1; current += d
        if current > best:
            best = current
    return best


def run_sa(n, adj, edges, num_trials, time_limit, seed_base=0):
    results = np.zeros(num_trials, dtype=np.int64)
    t0 = time.time()
    for i in range(num_trials):
        results[i] = run_sa_single(n, adj, edges, time_limit, seed=seed_base + i)
    return results, time.time() - t0


def stats(cuts, method, graph, wall):
    return MethodResult(
        method=method, graph=graph, n_trials=len(cuts),
        mean=float(cuts.mean()),
        best=int(cuts.max()),
        worst=int(cuts.min()),
        std=float(cuts.std()),
        wall_s=wall,
    )


def budget_for(n: int) -> dict:
    """問題サイズに応じた予算 (trial数・step数) を決める。"""
    if n <= 1000:
        return {"trials": 20, "cim_rounds": 1500, "cac_steps": 20000, "sa_time": 5.0}
    if n <= 2000:
        return {"trials": 20, "cim_rounds": 1500, "cac_steps": 20000, "sa_time": 10.0}
    if n <= 5000:
        return {"trials": 10, "cim_rounds": 1500, "cac_steps": 15000, "sa_time": 15.0}
    if n <= 10000:
        return {"trials": 5, "cim_rounds": 1500, "cac_steps": 10000, "sa_time": 15.0}
    return {"trials": 5, "cim_rounds": 1000, "cac_steps": 8000, "sa_time": 20.0}


def fmt_pct(x, bks):
    if bks is None or bks == 0:
        return "—"
    return f"{100.0 * x / bks:.2f}%"


def main(graph_ids, out_csv="results/benchmark_gset.csv"):
    rows = []
    for gid in graph_ids:
        path = f"input/{gid}.txt"
        if not os.path.exists(path):
            print(f"[skip] {gid}: {path} not found")
            continue
        n, k, adj, edges = load_graph(path)
        budget = budget_for(n)
        bks = BKS.get(gid)
        print(f"\n==== {gid}  N={n:>5}  K={k:>6}  BKS={bks}  "
              f"(trials={budget['trials']})  ====")

        # CIM
        cim_cuts, cim_t = run_cim(n, edges, budget["trials"], budget["cim_rounds"])
        r = stats(cim_cuts, "CIM", gid, cim_t)
        rows.append(r)
        print(f"  CIM  mean={r.mean:8.1f}  best={r.best:>6}  std={r.std:5.1f}  "
              f"({r.wall_s:5.1f}s)  best/BKS={fmt_pct(r.best, bks)}")

        # CAC
        cac_cuts, cac_t = run_cac(n, edges, budget["trials"], budget["cac_steps"])
        r = stats(cac_cuts, "CAC", gid, cac_t)
        rows.append(r)
        print(f"  CAC  mean={r.mean:8.1f}  best={r.best:>6}  std={r.std:5.1f}  "
              f"({r.wall_s:5.1f}s)  best/BKS={fmt_pct(r.best, bks)}")

        # SA
        sa_cuts, sa_t = run_sa(n, adj, edges, budget["trials"], budget["sa_time"])
        r = stats(sa_cuts, "SA", gid, sa_t)
        rows.append(r)
        print(f"  SA   mean={r.mean:8.1f}  best={r.best:>6}  std={r.std:5.1f}  "
              f"({r.wall_s:5.1f}s)  best/BKS={fmt_pct(r.best, bks)}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("graph,method,n_trials,mean,best,worst,std,wall_s,bks,best_pct_bks\n")
        for r in rows:
            bks = BKS.get(r.graph)
            pct_s = f"{100.0 * r.best / bks:.3f}" if bks else ""
            bks_s = str(bks) if bks else ""
            f.write(f"{r.graph},{r.method},{r.n_trials},{r.mean:.2f},{r.best},"
                    f"{r.worst},{r.std:.2f},{r.wall_s:.2f},{bks_s},{pct_s}\n")
    print(f"\nSaved: {out_csv}")
    return rows


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--graphs", nargs="+",
                   default=["G1", "G14", "G15", "G22", "G23", "G32", "G39",
                            "G55", "G70", "G77", "G81"])
    p.add_argument("--output-csv", default="results/benchmark_gset.csv")
    args = p.parse_args()
    main(args.graphs, args.output_csv)
