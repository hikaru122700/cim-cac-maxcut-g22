"""Phase 0-2: GA の子が「どこで」負けるかを記録する診断。

9/14 の G70 全 CIM 初期化では 300 世代で集団が一度も変わらなかった。
その原因を、交叉直後・子への TS 後・集団への採否に分けて調べる。

仮説(実行前に記録):
  交叉は親の不一致頂点(G70 では約 37%)をランダムに近く割り当てるため、子は
  大きく壊れる。TS 予算(G70 は 60,000 反復)ではそれを集団の品質まで修復できず、
  子は毎回、集団の最弱個体より悪いまま DisQual で棄却される。
  → TS 予算を 10 倍にすると、子の TS 後 cut が集団に届き、採用が起こるはず。

条件: G70、16 run、集団 10、30 世代。
  - cim_ts60k   : 全 CIM 初期化、TS 60,000(9/14 と同設定)
  - cim_ts600k  : 全 CIM 初期化、TS 600,000
  - rand_ts60k  : ランダム初期化、TS 60,000(対照)

使い方(プロジェクトルートから):
  uv run python scripts/benchmarks/ga_child_fate_diag.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.GA import simulate_ga_batch
from scripts.benchmarks.algo_registry import PARAMS, load_context, run_cim

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

EXPERIMENT_KIND = "ga_child_fate_diag"
CIM_BUDGET = 600  # 9/14 の ga_full_cim_population_bench と同じ


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


def build_description(args) -> str:
    parts = [args.dataset, f"nt{args.trials}", f"gen{args.generations}"]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def full_cim_population(ctx, trials, pop_size):
    """各 run・各個体が異なる CIM seed(0 .. trials*pop_size-1)。"""
    k = trials * pop_size
    _, signs = run_cim(ctx, CIM_BUDGET, k, np.arange(k, dtype=np.int64))
    return np.asarray(signs).reshape(trials, pop_size, ctx.n)


def run_condition(ctx, args, name, ts_iters, init_population):
    p = PARAMS[ctx.name]["GA"]
    records = []
    t0 = time.perf_counter()
    best, _ = simulate_ga_batch(
        ctx.n, ctx.edges, ctx.weights, args.trials,
        pop_size=p["pop_size"], max_generations=args.generations,
        ts_iters=ts_iters, cr=p["cr"], alpha_tenure=p["alpha_tenure"],
        beta_quality=p["beta_quality"], init_population=init_population,
        seeds=np.arange(args.trials, dtype=np.int64),
        child_observer=lambda g, r: records.append(r),
    )
    elapsed = time.perf_counter() - t0
    stack = {k: np.stack([r[k] for r in records]) for k in records[0]}  # (gen, trials)
    return {
        "name": name,
        "ts_iters": ts_iters,
        "time_s": elapsed,
        "final_best_mean": float(np.mean(best)),
        "final_gap_mean": float(ctx.bks - np.mean(best)),
        "accept_rate": float(stack["accepted"].mean()),
        "accept_runs": int(stack["accepted"].any(axis=0).sum()),
        "child_beats_pop_min_rate": float((stack["child_ts"] > stack["pop_min"]).mean()),
        "child_beats_pop_max_rate": float((stack["child_ts"] > stack["pop_max"]).mean()),
        "mean_parent": float(((stack["parent_a"] + stack["parent_b"]) / 2).mean()),
        "mean_pop_min": float(stack["pop_min"].mean()),
        "mean_child_raw": float(stack["child_raw"].mean()),
        "mean_child_ts": float(stack["child_ts"].mean()),
        "mean_disagree_frac": float(stack["n_disagree"].mean() / ctx.n),
    }, stack


def plot(results, stacks, bks, out_dir, title):
    fig, axes = plt.subplots(1, len(stacks), figsize=(5.2 * len(stacks), 4.6), dpi=150,
                             sharey=True, squeeze=False)
    for ax, (name, st) in zip(axes[0], stacks.items()):
        gens = np.arange(1, st["child_ts"].shape[0] + 1)
        ax.plot(gens, bks - st["pop_min"].mean(axis=1), "-", color="#555", label="集団の最弱個体")
        ax.plot(gens, bks - st["pop_max"].mean(axis=1), "--", color="#555", label="集団の最良個体")
        ax.plot(gens, bks - st["child_ts"].mean(axis=1), "-o", ms=3, color="#d62728",
                label="子(TS 後)")
        acc = st["accepted"].mean(axis=1)
        ax2 = ax.twinx()
        ax2.bar(gens, acc, color="#1f77b4", alpha=0.25, label="採用率")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("採用率(16 run 中)")
        ax2.tick_params(direction="in", which="both")
        r = results[name]
        ax.set_title(f"{name}\n採用率 {r['accept_rate']:.2f}、交叉直後の gap "
                     f"{bks - r['mean_child_raw']:.0f}")
        ax.set_xlabel("世代")
        ax.set_ylabel("BKS との差(16 run 平均)")
        ax.tick_params(direction="in", which="both", top=True)
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_dir / "child_fate.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G70")
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--ts-multipliers", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{build_description(args)}"
    out_dir.mkdir(parents=True)

    def log(msg):
        print(msg, flush=True)
        with (out_dir / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    ctx = load_context(args.dataset)
    p = PARAMS[ctx.name]["GA"]
    cim_pop = full_cim_population(ctx, args.trials, p["pop_size"])
    rand_pop = (
        np.random.default_rng(20261016).random((args.trials, p["pop_size"], ctx.n)) < 0.5
    ).astype(np.int8)

    conditions = [(f"cim_ts{p['ts_iters'] * m // 1000}k", p["ts_iters"] * m, cim_pop)
                  for m in args.ts_multipliers]
    conditions.append((f"rand_ts{p['ts_iters'] // 1000}k", p["ts_iters"], rand_pop))

    results, stacks = {}, {}
    for name, ts, pop in conditions:
        res, st = run_condition(ctx, args, name, ts, pop)
        results[name], stacks[name] = res, st
        log(
            f"{name:>12}: {res['time_s']:.0f}s gap={res['final_gap_mean']:.1f} "
            f"accept={res['accept_rate']:.3f} (runs with any accept {res['accept_runs']}/{args.trials}) "
            f"parent={res['mean_parent']:.1f} pop_min={res['mean_pop_min']:.1f} "
            f"child_raw={res['mean_child_raw']:.1f} child_ts={res['mean_child_ts']:.1f} "
            f"child>pop_min={res['child_beats_pop_min_rate']:.3f} "
            f"disagree={res['mean_disagree_frac']:.3f}"
        )

    meta = {
        "dataset": args.dataset, "bks": ctx.bks, "trials": args.trials,
        "generations": args.generations, "cim_budget": CIM_BUDGET,
        "ga_params": p, "cim_seeds": f"0..{args.trials * p['pop_size'] - 1}",
        "random_pop_seed": 20261016,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"meta": meta, "conditions": results}, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        out_dir / "records.npz",
        **{f"{name}__{k}": v for name, st in stacks.items() for k, v in st.items()},
    )
    plot(results, stacks, ctx.bks, out_dir,
         f"{args.dataset}: 交叉した子は集団に入れるか(子の TS 後 cut と集団の比較)")
    log(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
