"""
CAC の実行 + HTML ビジュアライザ生成 CLI。

バッチ実行 (Numba JIT、高速) で集計統計を得つつ、
別途 1 trial をトレーサー (純粋 Python、低速だが詳細記録可) で走らせ、
結合して scripts/visualize で HTML にレンダリングする。

使い方:

    uv run python -m scripts.run_cac_viz
    uv run python -m scripts.run_cac_viz --num-trials 50 --outer-steps 20000
    uv run python -m scripts.run_cac_viz --output results/viz/my_run.html

出力: results/viz/cac_<timestamp>.html (または --output で指定)
      ブラウザで開けば全チャートが見える。
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.plotting.trace_cac import trace_cac_single_trial
from scripts.plotting.visualize import RunRecord, write_html


def main(
    graph_path: str = "input/G22.txt",
    num_trials: int = 100,
    num_outer_steps: int = 50000,
    seed_base: int = 0,
    snapshot_interval: int = 100,
    spin_frame_interval: int = 500,
    output: str | None = None,
    target_cut: int = 13359,
) -> Path:
    # 遅延 import (numba 起動を避けたい純粋ロジックテストのため)
    from CAC import compute_gset_parameters, simulate_cac_batch
    from CIM import build_coupling_matrix, load_graph

    t_total = time.time()
    print("=" * 60)
    print("CAC run + HTML visualizer")
    print("=" * 60)

    # ---- グラフ読み込み & 結合行列 ----
    n, k_edges, _adj, edges = load_graph(graph_path)
    graph_name = Path(graph_path).stem
    print(f"Graph: {graph_path}  N={n}  K={k_edges}")
    J = build_coupling_matrix(n, edges, -1.0)
    gset_params = compute_gset_parameters(J, n)

    # ---- バッチ実行 (num_trials 全件, JIT) ----
    print(f"\n[1/2] Batch simulation ({num_trials} trials x "
          f"{num_outer_steps} steps, Numba JIT)...")
    seeds = np.array(
        [seed_base + i for i in range(num_trials)], dtype=np.int64
    )
    t_batch = time.time()
    batch_cuts, _ = simulate_cac_batch(
        n=n,
        J=J,
        edges=edges,
        num_outer_steps=num_outer_steps,
        num_trials=num_trials,
        p=gset_params["p"],
        alpha=gset_params["alpha"],
        rho=gset_params["rho"],
        delta=gset_params["delta"],
        beta0_error=gset_params["beta0_error"],
        gamma_growth=gset_params["gamma_growth"],
        tau=gset_params["tau"],
        n_x_inner=gset_params["n_x_inner"],
        n_e_inner=gset_params["n_e_inner"],
        dt_x=gset_params["dt_x"],
        dt_e=gset_params["dt_e"],
        e_max=gset_params["e_max"],
        seeds=seeds,
    )
    batch_elapsed = time.time() - t_batch
    print(f"    done in {batch_elapsed:.1f} sec")
    final_cuts = batch_cuts.astype(int).tolist()
    print(f"    mean={np.mean(final_cuts):.2f}  "
          f"max={max(final_cuts)}  min={min(final_cuts)}  "
          f"optimal={sum(1 for c in final_cuts if c == target_cut)}/{num_trials}")

    # ---- 代表 trial のトレース (純粋 Python) ----
    # 最も cut が高かった trial と同じ seed で再現
    best_trial_idx = int(np.argmax(batch_cuts))
    trace_seed = int(seeds[best_trial_idx])
    print(f"\n[2/2] Tracing representative trial "
          f"(best trial #{best_trial_idx + 1}, seed={trace_seed})...")
    trace = trace_cac_single_trial(
        n=n,
        J=J,
        edges=np.asarray(edges, dtype=np.int64),
        num_outer_steps=num_outer_steps,
        config=gset_params,
        seed=trace_seed,
        snapshot_interval=snapshot_interval,
        spin_frame_interval=spin_frame_interval,
    )
    print(f"    done in {trace.wall_time_sec:.1f} sec, "
          f"{len(trace.snapshots)} snapshots, "
          f"{len(trace.spin_frames)} spin frames")
    print(f"    trace final_cut={trace.final_cut}  "
          f"(batch final_cut for this seed={int(batch_cuts[best_trial_idx])})")

    # ---- RunRecord 組み立て & HTML レンダリング ----
    total_elapsed = time.time() - t_total
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 表示用 config (float/int のみ)
    display_config = {
        "p": gset_params["p"],
        "alpha": gset_params["alpha"],
        "rho": gset_params["rho"],
        "delta": gset_params["delta"],
        "beta0_error": gset_params["beta0_error"],
        "gamma_growth": gset_params["gamma_growth"],
        "tau": gset_params["tau"],
        "n_x_inner": gset_params["n_x_inner"],
        "n_e_inner": gset_params["n_e_inner"],
        "dt_x": gset_params["dt_x"],
        "dt_e": gset_params["dt_e"],
        "e_max": gset_params["e_max"],
        "num_trials": num_trials,
        "num_outer_steps": num_outer_steps,
        "seed_base": seed_base,
        "trace_seed": trace_seed,
    }

    record = RunRecord(
        method="CAC",
        graph_name=graph_name,
        n_spins=n,
        n_edges=k_edges,
        num_trials=num_trials,
        num_outer_steps=num_outer_steps,
        config=display_config,
        final_cuts=tuple(final_cuts),
        trajectory=trace.snapshots,
        spin_frames=trace.spin_frames,
        wall_time_sec=total_elapsed,
        timestamp=timestamp,
        target_cut=target_cut,
    )

    # 出力先
    if output is None:
        out_path = Path("results/viz") / (
            f"cac_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
    else:
        out_path = Path(output)

    written = write_html(record, out_path)
    print(f"\nHTML visualization written to: {written}")
    print(f"Total wall time: {total_elapsed:.1f} sec")
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CAC run + HTML visualizer",
    )
    parser.add_argument("--graph", default="input/G22.txt")
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--outer-steps", type=int, default=50000)
    parser.add_argument("--seed-base", type=int, default=0)
    parser.add_argument(
        "--snapshot-interval", type=int, default=100,
        help="トレース時の通常スナップショット周期 "
             "(リセット/改善時は必ず記録)",
    )
    parser.add_argument(
        "--spin-frame-interval", type=int, default=500,
        help="AHC 風プレーヤー用 per-spin フレーム記録周期 "
             "(改善時は必ず記録, 細かすぎると HTML 肥大化)",
    )
    parser.add_argument(
        "--output", default=None,
        help="HTML 出力先 (既定: results/viz/cac_<timestamp>.html)",
    )
    parser.add_argument("--target-cut", type=int, default=13359)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        graph_path=args.graph,
        num_trials=args.num_trials,
        num_outer_steps=args.outer_steps,
        seed_base=args.seed_base,
        snapshot_interval=args.snapshot_interval,
        spin_frame_interval=args.spin_frame_interval,
        output=args.output,
        target_cut=args.target_cut,
    )
