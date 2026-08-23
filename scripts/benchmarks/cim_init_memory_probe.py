"""cim_init_memory_probe.py — 初期振幅の記憶がどこまで残るかを直接測る診断。

実験1(初期振幅のランダム化)で、予測に反して多様性が単調に増えた。
「初期条件はしきい値到達までに指数的に減衰して消えるはず」という読みが
正しいかを、振幅ノルムと初期符号との重なりの時間発展で確かめる。

測る量(stride round ごと、試行平均):
  ||c||_2                     … 振幅のノルム。しきい値前は減衰、以後は増幅。
  overlap(sign c(k), sign c(0)) … 初期符号との重なり [0,1]。0 なら記憶は消えている。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe -u scripts/benchmarks/cim_init_memory_probe.py \
      --datasets G55 G22 --num-trials 8 --tag probe
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.benchmarks.algo_registry import DATASETS, PARAMS, load_context

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "axes.labelsize": 20, "axes.titlesize": 20, "figure.titlesize": 22,
    "legend.fontsize": 20, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "font.size": 14,
})

EXPERIMENT_KIND = "cim_init_memory"

_spec = importlib.util.spec_from_file_location(
    "cim_ablation", ROOT / "modules" / "2026-08-23_CIM_ablation.py")
CIM_ABL = importlib.util.module_from_spec(_spec)
sys.modules["cim_ablation"] = CIM_ABL
_spec.loader.exec_module(CIM_ABL)

BASE_ROUNDS = {"G22": 4800, "G55": 9600, "G70": 4800, "K2000": 300}
INIT_SCALES = [0.0, 1e-5, 1e-3, 1e-1]
COLORS = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad"]


def get_out_dir(desc: str) -> Path:
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    mx = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            h = p.name.split("_", 1)[0]
            if h[1:].isdigit():
                mx = max(mx, int(h[1:]))
    out = kind_root / f"v{mx + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["G22", "G55"],
                    choices=list(DATASETS))
    ap.add_argument("--num-trials", type=int, default=8)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    desc = "_".join(["_".join(args.datasets), f"nt{args.num_trials}"]
                    + ([args.tag] if args.tag else []))
    out_dir = get_out_dir(desc)
    logf = open(out_dir / "run.log", "w", encoding="utf-8")

    def log(m):
        print(m, flush=True)
        logf.write(m + "\n")
        logf.flush()

    log(f"out_dir = {out_dir}")
    seeds = np.arange(args.num_trials, dtype=np.int64)
    payload: dict = {"meta": {"datasets": args.datasets,
                              "num_trials": args.num_trials,
                              "stride": args.stride,
                              "init_scales": INIT_SCALES}, "results": {}}

    fig, axes = plt.subplots(2, len(args.datasets),
                             figsize=(9.0 * len(args.datasets), 12.0))
    axes = np.atleast_2d(axes)
    if len(args.datasets) == 1:
        axes = axes.reshape(2, 1)

    for col, ds in enumerate(args.datasets):
        ctx = load_context(ds)
        p = PARAMS[ds]["CIM"]
        rounds = BASE_ROUNDS[ds]
        k_th = CIM_ABL.threshold_round(p["kappa"], p["L"], p["eta"], p["dP_per_round"])
        log(f"\n=== {ds} (n={ctx.n}, {rounds} rounds, 無結合しきい値 round=約{k_th:.0f}) ===")
        payload["results"][ds] = {"threshold_round": float(k_th), "rounds": rounds}

        for color, s in zip(COLORS, INIT_SCALES):
            norms, ov, rr = CIM_ABL.probe_cim_amplitude(
                ctx.n, ctx.J_cim, rounds, args.num_trials,
                p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
                p["photon_energy"], p["dP_per_round"], seeds,
                init_scale=s, stride=args.stride)
            nm, om = norms.mean(axis=0), ov.mean(axis=0)
            lab = "baseline (c(0)=0)" if s == 0.0 else f"初期振幅 {s:g}"
            axes[0, col].plot(rr, nm, color=color, linewidth=2.4, label=lab)
            axes[1, col].plot(rr, om, color=color, linewidth=2.4, label=lab)
            payload["results"][ds][f"init{s:g}"] = {
                "rounds": rr.tolist(), "norm_mean": nm.tolist(),
                "overlap_mean": om.tolist(),
                "overlap_final": float(om[-1]),
                "norm_min": float(nm.min()),
                "round_of_min": int(rr[int(np.argmin(nm))]),
            }
            log(f"  init={s:<8g} ノルム最小={nm.min():.3e}(round {rr[int(np.argmin(nm))]})"
                f"  最終ノルム={nm[-1]:.3e}  初期符号との最終重なり={om[-1]:.4f}")

        for row in (0, 1):
            ax = axes[row, col]
            ax.axvline(k_th, color="#7f8c8d", linestyle="--", linewidth=2.0)
            ax.tick_params(direction="in", which="both", top=True, right=True)
            ax.set_xlabel("round step")
        axes[0, col].set_yscale("log")
        axes[0, col].set_ylabel("振幅ノルム ||c||")
        axes[0, col].set_title(f"{ds}: 振幅の減衰と増幅")
        axes[1, col].set_ylabel("初期符号との重なり")
        axes[1, col].set_ylim(-0.02, 1.02)
        axes[1, col].set_title(f"{ds}: 初期条件の記憶")
        axes[0, col].text(k_th, axes[0, col].get_ylim()[1], " 無結合しきい値",
                          va="top", fontsize=13, color="#7f8c8d")

    axes[0, 0].legend(loc="best", framealpha=0.92, fontsize=16)
    fig.suptitle("初期振幅の記憶はどこまで残るか(破線=無結合しきい値 round)", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_dir / "init_memory.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    log(f"\nsaved -> {out_dir}")
    logf.close()


if __name__ == "__main__":
    main()
