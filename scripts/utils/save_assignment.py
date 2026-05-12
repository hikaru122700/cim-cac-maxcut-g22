"""
CAC を 1 trial 実行し、最終スピン割当を 0/1 テキストファイルに保存する。

Web ビジュアライザ (web/) にアップロードするための出力生成 CLI。
出力形式: N 行, 各行 "0" または "1" (0-indexed, 頂点 i の割当)。

使い方:

    uv run python -m scripts.save_assignment
    uv run python -m scripts.save_assignment --outer-steps 50000 \\
        --output results/assignments/g22_cac.txt

ビジュアライザで開く際は:
  - Graph file: input/G22.txt (Gset 形式)
  - Assignment file: 出力された .txt ファイル
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(
    graph_path: str = "input/G22.txt",
    num_outer_steps: int = 50000,
    seed: int = 0,
    output: str | None = None,
) -> Path:
    # 遅延 import (numba 起動を避けたい純粋ロジックテストのため)
    from CAC import compute_gset_parameters, simulate_cac_batch
    from CIM import build_coupling_matrix, load_graph

    n, k_edges, _adj, edges = load_graph(graph_path)
    graph_name = Path(graph_path).stem
    print(f"Graph: {graph_path}  N={n}  K={k_edges}")

    J = build_coupling_matrix(n, edges, -1.0)
    gset_params = compute_gset_parameters(J, n)

    print(f"Running CAC 1 trial ({num_outer_steps} steps, seed={seed})...")
    seeds = np.array([seed], dtype=np.int64)
    cuts, signs_batch = simulate_cac_batch(
        n=n, J=J, edges=edges,
        num_outer_steps=num_outer_steps, num_trials=1,
        p=gset_params["p"], alpha=gset_params["alpha"],
        rho=gset_params["rho"], delta=gset_params["delta"],
        beta0_error=gset_params["beta0_error"],
        gamma_growth=gset_params["gamma_growth"],
        tau=gset_params["tau"],
        n_x_inner=gset_params["n_x_inner"],
        n_e_inner=gset_params["n_e_inner"],
        dt_x=gset_params["dt_x"], dt_e=gset_params["dt_e"],
        e_max=gset_params["e_max"], seeds=seeds,
    )
    cut = int(cuts[0])
    signs = np.asarray(signs_batch[0], dtype=bool)  # True = +1 side
    print(f"  final cut = {cut}  (+1 spins: {int(signs.sum())}, "
          f"-1 spins: {int(n - signs.sum())})")

    if output is None:
        output = f"results/assignments/{graph_name}_cac_seed{seed}.txt"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 0 (= -1 side) / 1 (= +1 side)  の N 行テキストで出力
    lines = ["1" if s else "0" for s in signs]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Assignment written to: {out_path}")
    return out_path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAC 1 trial を走らせて 0/1 割当ファイルを保存",
    )
    p.add_argument("--graph", default="input/G22.txt")
    p.add_argument("--outer-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--output", default=None,
        help="出力先 (既定: results/assignments/<graph>_cac_seed<S>.txt)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        graph_path=args.graph,
        num_outer_steps=args.outer_steps,
        seed=args.seed,
        output=args.output,
    )
