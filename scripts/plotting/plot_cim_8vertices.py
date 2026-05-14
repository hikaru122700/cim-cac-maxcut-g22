"""CIM を 1500 ラウンド回し、8 頂点の振幅推移を個別に色分けしてプロットする。"""
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.sparse._sparsetools import csr_matvec
from modules.CIM import build_coupling_matrix, load_graph

matplotlib.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False


def next_version(out_dir: Path, base: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name.split("_")[0][1:]) for p in out_dir.iterdir()
                if p.name.startswith("v") and p.name.endswith(f"_{base}.png")
                and p.name.split("_")[0][1:].isdigit()]
    return (max(existing) if existing else 0) + 1


def run(num_rounds, seed=42):
    n, k, adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J = build_coupling_matrix(n, edges, -0.03)
    rng = np.random.default_rng(seed)
    kappa, L, gamma = 130.0, 0.05, 42.09
    eta = 10.0 ** (-11.0 / 10.0)
    bw, hbarw, dP = 1.0e9, 1.28e-19, 0.05e-3
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bw * hbarw)
    sqrt_eta = np.sqrt(eta)
    c = np.zeros(n)
    Jc = np.zeros(n)
    hist = np.zeros((num_rounds + 1, n))
    for kk in range(num_rounds):
        P_p = (kk + 1) * dP
        g0 = 2 * kappa * np.sqrt(P_p) * L
        Jc.fill(0)
        csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
        coup = sqrt_eta * c + Jc
        half_g = 0.5 * g0 * (1 - gamma * coup * coup)
        sg = np.exp(half_g)
        c = sg * coup + rng.standard_normal(n) * (noise_const * sg)
        hist[kk + 1] = c
    return hist, n


def main():
    NUM_ROUNDS = 1500
    hist, n = run(NUM_ROUNDS)
    rounds = np.arange(NUM_ROUNDS + 1)

    final = hist[-1]
    pos_idx = np.where(final > 0)[0]
    neg_idx = np.where(final < 0)[0]
    rng_sample = np.random.default_rng(3)
    sample_pos = rng_sample.choice(pos_idx, size=4, replace=False)
    sample_neg = rng_sample.choice(neg_idx, size=4, replace=False)
    sample_idx = np.concatenate([sample_pos, sample_neg])
    print("選ばれた8頂点:", sample_idx.tolist())
    for i, v in enumerate(sample_idx):
        print(f"  #{i+1} vertex {v}: c[最終]={hist[-1, v]:+.4f}")

    colors = plt.cm.tab10(np.arange(8))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    for i, v in enumerate(sample_idx):
        ax.plot(rounds, hist[:, v], lw=1.2, color=colors[i],
                label=f"頂点 {v}", alpha=0.9)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i")
    ax.set_title(f"8頂点の振幅推移(全{NUM_ROUNDS}ラウンド・線形y軸)")
    ax.legend(ncol=2, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[0, 1]
    for i, v in enumerate(sample_idx):
        ax.plot(rounds, np.abs(hist[:, v]) + 1e-12, lw=1.2,
                color=colors[i], label=f"頂点 {v}", alpha=0.9)
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("|c_i|")
    ax.set_yscale("log")
    ax.set_title("|c_i|の対数y軸表示(増幅過程が見やすい)")
    ax.legend(ncol=2, fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 0]
    zoom = slice(0, 600)
    for i, v in enumerate(sample_idx):
        ax.plot(rounds[zoom], hist[zoom, v], lw=1.2,
                color=colors[i], label=f"頂点 {v}", alpha=0.9)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i")
    ax.set_title("分岐期の拡大(0〜600ラウンド)")
    ax.legend(ncol=2, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 1]
    zoom = slice(1300, 1501)
    for i, v in enumerate(sample_idx):
        ax.plot(rounds[zoom], hist[zoom, v], lw=1.2,
                color=colors[i], label=f"頂点 {v}", alpha=0.9)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i")
    ax.set_title("飽和期の拡大(1300〜1500ラウンド)")
    ax.legend(ncol=2, fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"CIM振幅ダイナミクス — 8頂点の個別軌道(G22, seed=42)",
        fontsize=13,
    )
    fig.tight_layout()
    out_dir = ROOT / "results" / date.today().isoformat()
    base = f"cim_8vertices_{NUM_ROUNDS}rounds"
    v = next_version(out_dir, base)
    out = out_dir / f"v{v}_{base}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
