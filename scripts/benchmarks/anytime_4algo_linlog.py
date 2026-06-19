"""anytime_4algo_linlog.py — CIM/SA/GA/CAC の「探索時間 × 最大カット」を 2 枚で描く。

anytime_bench.py の予算スイープ(run_algo_sweep / algo_registry)を再利用し、
4 手法(CIM, SA, GA, CAC)だけを対象に、

  - anytime_linear.png : 横軸 = 実時間(線形)
  - anytime_log.png    : 横軸 = 実時間(対数)

の 2 枚を出力する。縦軸はどちらも最大カット値(バッチ最良。破線は平均)。

実行:
  python scripts/benchmarks/anytime_4algo_linlog.py --dataset G22 --num-trials 16
出力: results/<YYYY-MM-DD>/anytime_linlog/v{N}_<desc>/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, ALGOS, load_context
from scripts.benchmarks.anytime_bench import run_algo_sweep, setup_style

EXPERIMENT_KIND = "anytime_linlog"
ALGO_ORDER = ["CIM", "SA", "GA", "CAC"]   # 対象 4 手法
MAX_BATCH_SEC_DEFAULT = 45.0


def get_out_dir(desc: str) -> Path:
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            h = p.name.split("_", 1)[0]
            if h[1:].isdigit():
                v = max(v, int(h[1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_anytime(plt, ds, data, bks, out_path, xscale):
    """探索時間 × 最大カット。xscale="linear" or "log"。"""
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=140)
    for algo_key in ALGO_ORDER:
        pts = data.get(algo_key)
        if not pts:
            continue
        t = [p["time"] for p in pts]
        cmax = [p["cut_max"] for p in pts]
        cmean = [p["cut_mean"] for p in pts]
        c = ALGOS[algo_key]["color"]
        lab = ALGOS[algo_key]["label"]
        ax.plot(t, cmax, "-o", color=c, lw=2.0, ms=5, label=f"{lab}(最良)")
        ax.plot(t, cmean, "--", color=c, lw=1.0, alpha=0.55)
    ax.axhline(bks, color="k", ls=":", lw=1.4, label=f"BKS={bks}")
    ax.set_xscale(xscale)
    axis_label = "実時間 [秒]（バッチ, 線形）" if xscale == "linear" \
        else "実時間 [秒]（バッチ, 対数）"
    ax.set_xlabel(axis_label)
    ax.set_ylabel("最大カット値")
    ax.set_title(f"{ds}: 探索時間 × 最大カット(CIM / SA / GA / CAC)"
                 f"\n実線=バッチ最良・破線=平均")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22", choices=list(DATASETS.keys()))
    ap.add_argument("--num-trials", type=int, default=16)
    ap.add_argument("--max-batch-sec", type=float, default=MAX_BATCH_SEC_DEFAULT)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    ds = args.dataset
    desc = f"{ds}_nt{args.num_trials}" + (f"_{args.tag}" if args.tag else "")
    out_dir = get_out_dir(desc)

    def log(msg):
        print(msg, flush=True)
        with open(out_dir / "run.log", "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log(f"=== {EXPERIMENT_KIND} {ds} algos={ALGO_ORDER} nt={args.num_trials} ===")
    ctx = load_context(ds)
    log(f"[{ds}] n={ctx.n} edges={len(ctx.edges)} bks={ctx.bks}")

    data = {}
    for algo_key in ALGO_ORDER:
        log(f"  --- {algo_key} ---")
        data[algo_key] = run_algo_sweep(
            ctx, algo_key, args.num_trials, args.max_batch_sec, log)
        with open(out_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump({"dataset": ds, "bks": ctx.bks,
                       "num_trials": args.num_trials,
                       "algos": ALGO_ORDER, "results": data},
                      f, ensure_ascii=False, indent=1)

    plt = setup_style()
    plot_anytime(plt, ds, data, ctx.bks, out_dir / "anytime_linear.png", "linear")
    plot_anytime(plt, ds, data, ctx.bks, out_dir / "anytime_log.png", "log")

    log(f"\nsaved → {out_dir}")
    print(f"OUT_DIR={out_dir}")


if __name__ == "__main__":
    main()
