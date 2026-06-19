"""algo_registry.py — anytime ベンチ用の統一アルゴリズムレジストリ。

各アルゴリズム(CIM/CAC/SA/SB/PT-ICM/GA)を「同一インターフェースの run 関数 +
予算グリッド + データセット別パラメータ」で束ねる。anytime_bench.py / combo_bench.py
から共通利用する。

run シグネチャ:
    run(ctx, budget, num_trials, seeds) -> (cuts: np.ndarray, signs: np.ndarray)
  - ctx は GraphContext(n, edges, weights, J_cim, J_cac, J_sb, bks など)
  - budget はそのアルゴリズムの主要計算量ノブ(rounds / steps / iters / sweeps / gens)
  - cuts: (num_trials,)  signs: (num_trials, n) bool

データセット別パラメータは PARAMS[dataset][algo] で引く。G22 は optuna 済み実値、
他は文献推奨の instance-adaptive 設定(透明性のため出典をコメント)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from modules.CIM import build_coupling_matrix
from modules.CIM import simulate_cim_batch
from modules.GA import load_graph as load_graph_3
from modules.CAC import simulate_cac_batch, compute_gset_parameters
from modules.SA import simulate_sa_batch
from modules.SB import simulate_sb_batch, auto_c0
from modules.PT_ICM import simulate_pticm_batch
from modules.GA import simulate_ga_batch, tabu_refine_batch
from modules.GA import _build_csr as _ga_build_csr, _cut_batch as _ga_cut_batch


# ============================================================
#  データセット定義(BKS = 既知ベスト)
# ============================================================
DATASETS = {
    "G22":   {"path": "input/G22.txt",   "bks": 13359, "weighted": False},
    "K2000": {"path": "input/K2000.txt", "bks": 33337, "weighted": True},
    "G55":   {"path": "input/G55.txt",   "bks": 10299, "weighted": False},
    "G70":   {"path": "input/G70.txt",   "bks": 9591,  "weighted": False},
}


@dataclass
class GraphContext:
    name: str
    n: int
    edges: list
    weights: Optional[list]
    bks: int
    J_cim: object = None   # csr_matrix (coupling for CIM)
    J_cac: object = None   # csr_matrix (coupling -1)
    J_sb: object = None    # csr_matrix (coupling -1)
    _csr: object = None     # (indptr, indices, data) 真の重みでの CSR(統一採点用)

    def score(self, signs):
        """signs (num,n) を真の重み付きカットで統一採点(全手法で同一基準)。

        各モジュールの内部 cut(CAC は非重みカウント等)に依らず、返り符号から
        重み付きカットを再計算する。bool/0-1/±1 いずれでも可。
        """
        import numpy as _np
        s = _np.ascontiguousarray((_np.asarray(signs) > 0).astype(_np.int8))
        ip, ix, dt = self._csr
        return _ga_cut_batch(self.n, ip, ix, dt, s)


def load_context(name: str) -> GraphContext:
    d = DATASETS[name]
    n, edges, weights = load_graph_3(d["path"], return_weights=True)
    if not d["weighted"]:
        weights = None  # 非重み(全辺 +1)。各モジュールは None で +1 を仮定。
    # CIM の結合: 非重み G-set は tuned/-0.03 系、K2000 は -1.0(J∈±1)
    cim_coupling = PARAMS.get(name, {}).get("CIM", {}).get("coupling", -0.03)
    J_cim = build_coupling_matrix(n, edges, cim_coupling, weights=weights)
    J_cac = build_coupling_matrix(n, edges, -1.0, weights=weights)
    J_sb = build_coupling_matrix(n, edges, -1.0, weights=weights)
    # 統一採点用の真の重み CSR(非重みなら全辺 +1)
    ea = np.asarray([e[0] for e in edges], dtype=np.int64)
    eb = np.asarray([e[1] for e in edges], dtype=np.int64)
    ew = (np.ones(len(edges)) if weights is None
          else np.asarray(weights, dtype=np.float64))
    csr = _ga_build_csr(n, ea, eb, ew)
    return GraphContext(name, n, edges, weights, d["bks"], J_cim, J_cac, J_sb, csr)


# ============================================================
#  データセット別パラメータ
# ============================================================
# 物理 CIM パラメータ。G22 は optuna(results/2026-05-11, best=13307)実値。
# 他データセットは同じハードウェア定数を流用し coupling のみ調整(透明性のため明記)。
_CIM_G22 = dict(
    kappa=253.83145431791291, L=0.027829145901910754, gamma=7.2102623149028435,
    eta=10.0 ** (-10.369319116625467 / 10.0), bandwidth=1640034004.7080789,
    photon_energy=6.335180067425749e-20, dP_per_round=1.553743564765986e-05,
    coupling=-0.04942805518421356,
)
# 非 G22 G-set 用 CIM(G22 物理定数 + 標準 coupling -0.03)
_CIM_GSET = dict(_CIM_G22); _CIM_GSET["coupling"] = -0.03
# K2000 用 CIM(重み付き。coupling -1 で J∈±1。dP は小さめに)
_CIM_K2000 = dict(_CIM_G22); _CIM_K2000["coupling"] = -1.0
_CIM_K2000["dP_per_round"] = 1.553743564765986e-05

# CAC: G22 は optuna(best=13336)実値、他は compute_gset_parameters(論文 GSET 推奨)
_CAC_G22 = dict(
    p=0.41472107374160905, alpha=4.861264995684808, rho=1.7876032341307913,
    delta=0.000322786869687264, beta0_error=0.11589142280016017,
    gamma_growth=0.019844099082253593, tau=27544.79095737492,
    n_x_inner=6, n_e_inner=4, dt_x=0.015625, dt_e=0.0625, e_max=32.0,
)

PARAMS = {
    "G22": {
        "CIM": _CIM_G22,
        "CAC": _CAC_G22,
        "SA":  dict(t_start=2.0, t_end=0.001),
        "SB":  dict(variant="dSB", dt=0.5, a0=1.0),
        "PT":  dict(num_temps=12, t_min=0.05, t_max=3.0, swap_interval=1, icm_interval=5),
        "GA":  dict(pop_size=10, ts_iters=20000, cr=3000, alpha_tenure=15, beta_quality=0.6),
    },
    "K2000": {
        "CIM": _CIM_K2000,
        "CAC": None,  # compute_gset_parameters を実行時に
        "SA":  dict(t_start=2.0, t_end=0.001),
        "SB":  dict(variant="dSB", dt=0.5, a0=1.0),
        "PT":  dict(num_temps=16, t_min=0.1, t_max=3.0, swap_interval=1, icm_interval=5),
        "GA":  dict(pop_size=10, ts_iters=20000, cr=3000, alpha_tenure=15, beta_quality=0.6),
    },
    "G55": {
        "CIM": _CIM_GSET,
        "CAC": None,
        "SA":  dict(t_start=2.0, t_end=0.001),
        "SB":  dict(variant="dSB", dt=0.5, a0=1.0),
        "PT":  dict(num_temps=16, t_min=0.05, t_max=3.0, swap_interval=1, icm_interval=5),
        "GA":  dict(pop_size=10, ts_iters=40000, cr=3000, alpha_tenure=15, beta_quality=0.6),
    },
    "G70": {
        "CIM": _CIM_GSET,
        "CAC": None,
        "SA":  dict(t_start=2.0, t_end=0.001),
        "SB":  dict(variant="dSB", dt=0.5, a0=1.0),
        "PT":  dict(num_temps=20, t_min=0.05, t_max=3.0, swap_interval=1, icm_interval=5),
        "GA":  dict(pop_size=10, ts_iters=60000, cr=3000, alpha_tenure=15, beta_quality=0.6),
    },
}


# --- 永続的な tuned パラメータ上書き(tune_anytime_params.py が書き込む) ---
import json as _json
import os as _os

_OVERRIDE_PATH = _os.path.join("results", "anytime_tuned_params.json")


def load_overrides():
    """results/anytime_tuned_params.json があれば PARAMS にマージする。

    形式: {dataset: {algo: {param: value, ...}}}。CIM/CAC の per-dataset
    tuned 値を反映する。読めなければ無視。
    """
    if not _os.path.exists(_OVERRIDE_PATH):
        return
    try:
        with open(_OVERRIDE_PATH, "r", encoding="utf-8") as f:
            ov = _json.load(f)
    except Exception:  # noqa: BLE001
        return
    for ds, algos in ov.items():
        if ds not in PARAMS:
            continue
        for algo, params in algos.items():
            if PARAMS[ds].get(algo) is None:
                PARAMS[ds][algo] = dict(params)
            else:
                PARAMS[ds][algo].update(params)


load_overrides()


def cac_params(ctx: GraphContext) -> dict:
    """CAC パラメータ。tuned 上書き or G22 実 tuned or compute_gset_parameters。"""
    pre = PARAMS[ctx.name]["CAC"]
    if pre is not None:
        return pre
    return compute_gset_parameters(ctx.J_cac, ctx.n)


# ============================================================
#  予算グリッド(主要ノブの幾何スイープ)。x 軸は実測時間なので
#  グリッドは「だいたい速い→遅い」を覆えばよい。
# ============================================================
BUDGET_GRIDS = {
    "CIM": [150, 300, 600, 1200, 2400, 4800],
    "CAC": [500, 1500, 4000, 12000, 35000, 80000],
    "SA":  [30000, 100000, 300000, 1000000, 3000000, 10000000],
    "SB":  [150, 400, 1000, 2500, 6000, 15000],
    "PT":  [30, 80, 200, 600, 1800, 5000],
    "GA":  [3, 8, 20, 50, 120],
}


# ============================================================
#  各アルゴリズムの run 関数
# ============================================================
def run_cim(ctx, budget, num_trials, seeds):
    p = PARAMS[ctx.name]["CIM"]
    cuts, signs = simulate_cim_batch(
        ctx.n, ctx.J_cim, ctx.edges, int(budget), num_trials,
        p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
        p["photon_energy"], p["dP_per_round"], seeds, weights=ctx.weights,
    )
    return cuts, signs


def run_cac(ctx, budget, num_trials, seeds):
    p = cac_params(ctx)
    cuts, signs = simulate_cac_batch(
        ctx.n, ctx.J_cac, ctx.edges, int(budget), num_trials,
        p["p"], p["alpha"], p["rho"], p["delta"], p["beta0_error"],
        p["gamma_growth"], p["tau"], p["n_x_inner"], p["n_e_inner"],
        p["dt_x"], p["dt_e"], p["e_max"], seeds,
    )
    return cuts, signs


def run_sa(ctx, budget, num_trials, seeds):
    p = PARAMS[ctx.name]["SA"]
    cuts, signs = simulate_sa_batch(
        ctx.n, ctx.edges, ctx.weights, int(budget), num_trials,
        t_start=p["t_start"], t_end=p["t_end"], seeds=seeds,
    )
    return cuts, signs


def run_sb(ctx, budget, num_trials, seeds):
    p = PARAMS[ctx.name]["SB"]
    cuts, signs = simulate_sb_batch(
        ctx.n, ctx.J_sb, ctx.edges, int(budget), num_trials,
        variant=p["variant"], a0=p["a0"], dt=p["dt"],
        weights=ctx.weights, seeds=seeds,
    )
    return cuts, signs


def run_pt(ctx, budget, num_trials, seeds):
    p = PARAMS[ctx.name]["PT"]
    cuts, signs, _traj = simulate_pticm_batch(
        ctx.n, ctx.edges, ctx.weights, num_trials,
        num_sweeps=int(budget), num_temps=p["num_temps"],
        t_min=p["t_min"], t_max=p["t_max"],
        swap_interval=p["swap_interval"], icm_interval=p["icm_interval"],
        seeds=seeds,
    )
    return cuts, signs


def run_ga(ctx, budget, num_trials, seeds):
    p = PARAMS[ctx.name]["GA"]
    cuts, signs = simulate_ga_batch(
        ctx.n, ctx.edges, ctx.weights, num_trials,
        pop_size=p["pop_size"], max_generations=int(budget),
        ts_iters=p["ts_iters"], cr=p["cr"], alpha_tenure=p["alpha_tenure"],
        beta_quality=p["beta_quality"], seeds=seeds,
    )
    return cuts, signs


ALGOS = {
    "CIM": {"run": run_cim, "label": "CIM",      "color": "#e74c3c"},
    "CAC": {"run": run_cac, "label": "CAC",      "color": "#e67e22"},
    "SA":  {"run": run_sa,  "label": "SA",       "color": "#2980b9"},
    "SB":  {"run": run_sb,  "label": "SB(dSB)",  "color": "#16a085"},
    "PT":  {"run": run_pt,  "label": "PT-ICM",   "color": "#8e44ad"},
    "GA":  {"run": run_ga,  "label": "GA(memetic)", "color": "#d81b9e"},
}
