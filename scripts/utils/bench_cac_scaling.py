"""CAC / 開ループの実行時間スケーリングを、JIT コンパイルを除いて実測する。

「step を 33 倍にしたのに時間が 33 倍にならない」「CAC は開ループの 9 倍遅いはず
なのに合算が合わない」を切り分けるため、
  (1) JIT コンパイル(ウォームアップ)の固定費
  (2) コンパイル後の純粋な実行時間(1500 / 50000 step)
を分けて測る。G22・100 trial。
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402
from modules.CAC import compute_gset_parameters, simulate_cac_batch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "cim_pumpsched", ROOT / "modules" / "2026-06-08_CIM_pumpsched.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

TRIALS = 100


def t(fn):
    t0 = time.time()
    fn()
    return time.time() - t0


def main() -> None:
    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J_cac = build_coupling_matrix(n, edges, -1.0)
    J_cim = build_coupling_matrix(n, edges, -0.03)
    params = compute_gset_parameters(J_cac, n)
    params.pop("d_0", None); params.pop("d_1", None)
    seeds = np.arange(TRIALS, dtype=np.int64)

    kappa, L, gamma = 130.0, 0.05, 42.09
    eta = 10.0 ** (-1.1)
    phys = dict(kappa=kappa, L=L, gamma=gamma, eta=eta,
                bandwidth=1.0e9, photon_energy=1.28e-19)
    dP = 0.05e-3

    def cac(steps):
        return lambda: simulate_cac_batch(
            n=n, J=J_cac, edges=edges, num_outer_steps=steps,
            num_trials=TRIALS, seeds=seeds, **params)

    def cim(rounds):
        P = (np.arange(rounds) + 1) * dP
        return lambda: ps.simulate_cim_sched_batch(
            n, J_cim, edges, P, TRIALS, seeds=seeds, **phys)

    print(f"G22 N={n} E={k_edges} trials={TRIALS}\n")

    # --- CAC ---
    warm_cac = t(cac(2))           # 1回目: コンパイル込み(固定費)
    cac_1500 = t(cac(1500))        # 2回目以降: 純粋実行
    cac_1500b = t(cac(1500))       # 念のためもう1回
    cac_50000 = t(cac(50000))
    print("[CAC]")
    print(f"  warmup(2 step, =JITコンパイル固定費)  : {warm_cac:7.2f} s")
    print(f"  pure 1500  step                       : {cac_1500:7.2f} s ({cac_1500b:.2f} s)")
    print(f"  pure 50000 step                       : {cac_50000:7.2f} s")
    print(f"  比: 50000/1500 = {cac_50000/cac_1500:5.1f}x  (線形なら 33.3x)")

    # --- 開ループ CIM ---
    warm_cim = t(cim(2))
    cim_1500 = t(cim(1500))
    cim_1500b = t(cim(1500))
    print("\n[開ループ CIM]")
    print(f"  warmup(2 round, =JITコンパイル固定費) : {warm_cim:7.2f} s")
    print(f"  pure 1500  round                      : {cim_1500:7.2f} s ({cim_1500b:.2f} s)")

    print("\n[比較(純粋実行, 1500 step/round)]")
    print(f"  CAC@1500 / CIM@1500 = {cac_1500/cim_1500:5.1f}x  (= 真の per-step 倍率)")
    print("\n[“見かけ”の汚染された値(コンパイル込み)]")
    print(f"  CAC@1500  込み = {warm_cac + cac_1500:6.2f} s 相当")
    print(f"  CAC@50000 込み = {warm_cac + cac_50000:6.2f} s 相当")


if __name__ == "__main__":
    main()
