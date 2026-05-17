"""論文パラメータ vs reduced Optuna best を held-out seeds で再評価する。

env で挙動を制御:
  BEST_JSON   : reduced 系 best_params JSON のパス
                既定: results/2026-05-17/reduced_nr3000_optuna_best_params.json
  NUM_ROUNDS  : CIM の round 数 (best JSON の num_rounds に合わせるのが原則)
                既定: BEST_JSON 内の num_rounds
  N_TRIALS    : held-out 試行数 (既定 100)
  SEED_START  : held-out 開始 seed (既定 100; Optuna 中は 0..19)
  TAG         : 出力ファイル名のタグ (既定: BEST_JSON の "tag" or ファイル名)
  OUT_DIR     : 出力ディレクトリ (既定: BEST_JSON と同じ場所)
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modules.CIM import build_coupling_matrix, load_graph, simulate_cim_batch

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


PAPER_PARAMS: dict[str, float] = dict(
    kappa=130.0, L=0.05, gamma=42.09, loss_dB=11.0,
    bandwidth=1.0e9, photon_energy=1.28e-19,
    dP_per_round=0.05e-3, coupling=-0.03,
)
KNOWN_BEST: int = 13359


def load_optuna_params(best_json: Path) -> tuple[dict, dict, int, str]:
    """best_params JSON から探索済み + 固定パラを合体し、(params, meta, num_rounds, tag) を返す。"""
    with open(best_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    bp = data["best_params"]
    fp = data.get("fixed_params", {})

    # reduced 系は best_params に固定パラが入っていない。fixed_params とマージ。
    params: dict[str, float] = {**fp, **bp}
    # coupling 符号変換: abs_coupling -> coupling (負)
    if "abs_coupling" in params:
        params["coupling"] = -float(params.pop("abs_coupling"))

    required = {
        "kappa", "L", "gamma", "loss_dB", "bandwidth",
        "photon_energy", "dP_per_round", "coupling",
    }
    missing = required - params.keys()
    if missing:
        raise KeyError(f"best params に欠損: {missing}")

    num_rounds = int(data.get("num_rounds", 1500))
    tag = str(data.get("tag", best_json.stem))
    return params, data, num_rounds, tag


def run(params: dict, n: int, edges: np.ndarray, num_rounds: int, seeds: np.ndarray) -> np.ndarray:
    eta = 10.0 ** (-params["loss_dB"] / 10.0)
    J = build_coupling_matrix(n, edges, params["coupling"])
    best_cuts, _ = simulate_cim_batch(
        n=n, J=J, edges=edges,
        num_rounds=num_rounds, num_trials=len(seeds),
        kappa=params["kappa"], L=params["L"], gamma=params["gamma"], eta=eta,
        bandwidth=params["bandwidth"], photon_energy=params["photon_energy"],
        dP_per_round=params["dP_per_round"],
        seeds=seeds,
    )
    return best_cuts


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t-statistic (a vs b)。"""
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = len(a), len(b)
    return float((m1 - m2) / np.sqrt(v1 / n1 + v2 / n2))


def main() -> None:
    best_json = Path(os.environ.get(
        "BEST_JSON",
        "results/2026-05-17/reduced_nr3000_optuna_best_params.json",
    ))
    if not best_json.exists():
        raise FileNotFoundError(best_json)

    optuna_params, meta, default_rounds, default_tag = load_optuna_params(best_json)

    num_rounds = int(os.environ.get("NUM_ROUNDS", default_rounds))
    n_trials = int(os.environ.get("N_TRIALS", 100))
    seed_start = int(os.environ.get("SEED_START", 100))
    tag = os.environ.get("TAG", default_tag)
    out_dir = Path(os.environ.get("OUT_DIR", str(best_json.parent)))
    out_dir.mkdir(parents=True, exist_ok=True)

    n, k_edges, _, edges = load_graph("input/G22.txt")
    print(
        f"N={n}, K={k_edges}, num_rounds={num_rounds}, "
        f"n_trials={n_trials}, seeds={seed_start}..{seed_start + n_trials - 1}"
    )
    print(f"  optuna source: {best_json}")
    print(f"  tag: {tag}")

    seeds = np.arange(seed_start, seed_start + n_trials, dtype=np.int64)

    t0 = time.time()
    paper_cuts = run(PAPER_PARAMS, n, edges, num_rounds, seeds)
    paper_t = time.time() - t0

    t0 = time.time()
    optuna_cuts = run(optuna_params, n, edges, num_rounds, seeds)
    optuna_t = time.time() - t0

    diff = float(optuna_cuts.mean() - paper_cuts.mean())
    t_stat = welch_t(optuna_cuts, paper_cuts)

    print()
    print(
        f"{'paper':10s}  mean={paper_cuts.mean():.2f}  std={paper_cuts.std():.2f}  "
        f"best={paper_cuts.max()}  worst={paper_cuts.min()}  "
        f"median={np.median(paper_cuts):.1f}  time={paper_t:.2f}s"
    )
    print(
        f"{'optuna':10s}  mean={optuna_cuts.mean():.2f}  std={optuna_cuts.std():.2f}  "
        f"best={optuna_cuts.max()}  worst={optuna_cuts.min()}  "
        f"median={np.median(optuna_cuts):.1f}  time={optuna_t:.2f}s"
    )
    print(f"\ndiff (optuna - paper): {diff:+.2f}")
    print(f"Welch t-statistic: {t_stat:.3f}  (|t| > 2 で 95%, > 2.6 で 99% 有意)")

    # --- 比較ヒストグラム ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

    all_cuts = np.concatenate([paper_cuts, optuna_cuts])
    x_min = int(all_cuts.min()) - 10
    x_max = int(all_cuts.max()) + 10
    bins = np.linspace(x_min, x_max, 35)

    ax.hist(
        paper_cuts, bins=bins, color="#1f77b4", alpha=0.6,
        edgecolor="black", linewidth=0.4,
        label=f"論文パラメータ (mean={paper_cuts.mean():.1f}, best={paper_cuts.max()})",
    )
    ax.hist(
        optuna_cuts, bins=bins, color="#d62728", alpha=0.6,
        edgecolor="black", linewidth=0.4,
        label=f"Optuna 最適 [{tag}] (mean={optuna_cuts.mean():.1f}, best={optuna_cuts.max()})",
    )

    ax.axvline(paper_cuts.mean(), color="#1f77b4", linestyle=":", linewidth=1.5)
    ax.axvline(optuna_cuts.mean(), color="#d62728", linestyle=":", linewidth=1.5)
    ax.axvline(
        KNOWN_BEST, color="goldenrod", linestyle="--", linewidth=1.3,
        label=f"既知最良値 {KNOWN_BEST}",
    )

    ax.set_xlabel("best_cut")
    ax.set_ylabel("頻度")
    ax.set_title(
        f"論文 vs Optuna 最適 [{tag}]: held-out seeds {seed_start}..{seed_start + n_trials - 1} "
        f"({n_trials} 試行, num_rounds={num_rounds})\n"
        f"差 = {diff:+.2f} (Welch t = {t_stat:.2f})"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    out_png = out_dir / f"{tag}_vs_paper_heldout.png"
    fig.savefig(out_png)
    print(f"Saved: {out_png}")

    # 結果 JSON
    results = {
        "n_trials": n_trials,
        "seed_range": [int(seeds[0]), int(seeds[-1])],
        "num_rounds": num_rounds,
        "tag": tag,
        "source_best_json": str(best_json),
        "paper": {
            "params": PAPER_PARAMS,
            "mean": float(paper_cuts.mean()),
            "std": float(paper_cuts.std()),
            "best": int(paper_cuts.max()),
            "worst": int(paper_cuts.min()),
        },
        "optuna": {
            "params": optuna_params,
            "mean": float(optuna_cuts.mean()),
            "std": float(optuna_cuts.std()),
            "best": int(optuna_cuts.max()),
            "worst": int(optuna_cuts.min()),
        },
        "diff_mean": diff,
        "welch_t": t_stat,
    }
    out_json = out_dir / f"{tag}_vs_paper_heldout.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
