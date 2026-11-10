"""CIM の振幅 E_i(n)(コード上の c_i)の実際の時間変化を描く(中間発表の CIM 説明スライド用)。

G22 で、実験と同じ CIM パラメータ(algo_registry.PARAMS["G22"]["CIM"])・同じ周回数(2400)で
CIM を 1 回動かし、振幅を記録する。全 2000 スピンのうちランダムに選んだ数十本を、最後の符号で
色分けして描く(赤 = +1、青 = −1)。

使い方(プロジェクトルートから):
  python scripts/plotting/plot_cim_amplitude_trace.py [--rounds 2400] [--spins 40] [--seed 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from modules.CIM import simulate_cim_with_trajectory  # noqa: E402
from scripts.benchmarks.algo_registry import PARAMS, load_context  # noqa: E402

EXPERIMENT_KIND = "cim_amplitude_trace"
RED = "#E3201B"
BLUE = "#1F4FD8"

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    "axes.labelsize": 20,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "font.size": 14,
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
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


def run(dataset: str, rounds: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    """CIM を 1 回動かし、全周回の振幅 (rounds, n) と cut を返す。"""
    ctx = load_context(dataset)
    p = PARAMS[dataset]["CIM"]
    c_hist, cut_hist, sample_rounds, best_cut, _ = simulate_cim_with_trajectory(
        ctx.n, ctx.J_cim, ctx.edges, rounds, p["kappa"], p["L"], p["gamma"], p["eta"], p["bandwidth"],
        p["photon_energy"], p["dP_per_round"], seed, num_samples=rounds, weights=ctx.weights)
    return c_hist, cut_hist, sample_rounds, best_cut, ctx.bks


def plot(c_hist: np.ndarray, rounds_axis: np.ndarray, spins: np.ndarray, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    final_sign = np.sign(c_hist[-1, spins])
    for s, sign in zip(spins, final_sign):
        ax.plot(rounds_axis + 1, c_hist[:, s], color=RED if sign > 0 else BLUE, lw=1.0, alpha=0.8)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlim(0, rounds_axis[-1] + 1)
    ax.set_xlabel("Round $n$ (pump power increases →)", fontsize=18)
    ax.set_ylabel("Amplitude $E_i(n)$", fontsize=20)
    # 最初の約 200 周回は振幅が 0 付近なので、左上・左下にラベルを置けば線と重ならない
    ax.text(0.03, 0.93, "$σ_i = +1$", transform=ax.transAxes, ha="left", va="top", color=RED, fontsize=18)
    ax.text(0.03, 0.07, "$σ_i = -1$", transform=ax.transAxes, ha="left", va="bottom", color=BLUE, fontsize=18)
    fig.tight_layout()
    fig.savefig(out_png, dpi=250, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22", choices=["G22"])
    ap.add_argument("--rounds", type=int, default=2400)
    ap.add_argument("--spins", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.rounds < 2 or args.spins < 1:
        ap.error("need rounds >= 2 and spins >= 1")

    c_hist, cut_hist, rounds_axis, best_cut, bks = run(args.dataset, args.rounds, args.seed)
    n = c_hist.shape[1]
    spins = np.sort(np.random.default_rng(args.seed).choice(n, size=min(args.spins, n), replace=False))

    desc = f"{args.dataset}_r{args.rounds}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    kind_root = get_kind_root()
    out = kind_root / f"v{next_version(kind_root)}_{desc}"
    out.mkdir(parents=True)
    np.savez_compressed(out / "trace.npz", c_spins=c_hist[:, spins], spins=spins, rounds=rounds_axis,
                        cut=cut_hist)
    plot(c_hist, rounds_axis, spins, out / "amplitude.png")
    summary = {"dataset": args.dataset, "rounds": args.rounds, "seed": args.seed, "n": int(n),
               "plotted_spins": int(spins.size), "final_cut": float(cut_hist[-1]), "best_cut": best_cut,
               "bks": bks, "final_gap_pct": (bks - float(cut_hist[-1])) / bks * 100}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print("saved:", out)


if __name__ == "__main__":
    main()
