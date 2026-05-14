"""round 1300〜1500 を拡大して、何が起きているか確認する。"""
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


def run(num_rounds=1500, seed=42):
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
    edges_np = np.asarray(edges, dtype=np.int64)
    ea, eb = edges_np[:, 0], edges_np[:, 1]

    hist = np.zeros((num_rounds + 1, n))
    cuts = np.zeros(num_rounds + 1, dtype=np.int64)
    flips = np.zeros(num_rounds + 1, dtype=np.int64)
    prev_sign = c > 0

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
        s = c > 0
        cuts[kk + 1] = int((s[ea] != s[eb]).sum())
        flips[kk + 1] = int((s != prev_sign).sum())
        prev_sign = s
    return hist, cuts, flips, n


def main():
    hist, cuts, flips, n = run()
    rounds = np.arange(len(cuts))
    win = slice(1300, 1501)

    print("=== round 1300〜1500 の統計 ===")
    print(f"|c| min  range : {np.abs(hist[win]).min(axis=1).min():.4f}〜{np.abs(hist[win]).min(axis=1).max():.4f}")
    print(f"|c| mean range : {np.abs(hist[win]).mean(axis=1).min():.4f}〜{np.abs(hist[win]).mean(axis=1).max():.4f}")
    print(f"|c| max  range : {np.abs(hist[win]).max(axis=1).min():.4f}〜{np.abs(hist[win]).max(axis=1).max():.4f}")
    print(f"cut range      : {cuts[win].min()}〜{cuts[win].max()}")
    print(f"flips/round    : mean={flips[win].mean():.2f}, max={flips[win].max()}")
    print(f"\n=== round 1490〜1500 の flips の中身 ===")
    for r in range(1490, 1501):
        print(f"  round {r}: flips={flips[r]:3d}, cut={cuts[r]}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    sample = np.random.default_rng(0).choice(n, 30, replace=False)
    for i in sample:
        ax.plot(rounds[win], hist[win, i], lw=0.8, alpha=0.7)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("ラウンド"); ax.set_ylabel("振幅 c_i")
    ax.set_title("30頂点の振幅推移(1300〜1500ラウンドの拡大)")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1]
    ax.plot(rounds[win], np.abs(hist[win]).mean(axis=1), label="|c|の平均")
    ax.plot(rounds[win], np.abs(hist[win]).max(axis=1), label="|c|の最大")
    ax.plot(rounds[win], np.abs(hist[win]).min(axis=1), label="|c|の最小")
    P_p = (rounds[win]) * 5e-5
    g0 = 2 * 130 * np.sqrt(P_p) * 0.05
    c_eq = np.sqrt(np.clip((1 - (-np.log(10**(-1.1)))/g0) / (42.09 * 10**(-1.1)), 0, None))
    ax.plot(rounds[win], c_eq, "k--", lw=0.8, label="理論平衡値 c_eq(P_p)")
    ax.set_xlabel("ラウンド"); ax.set_ylabel("|c|")
    ax.set_title("集団の|c|統計と理論平衡値の比較")
    ax.legend(); ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[2]
    ax.plot(rounds[win], flips[win], lw=0.8)
    ax.set_xlabel("ラウンド"); ax.set_ylabel("符号反転した頂点数")
    ax.set_title("各ラウンドの符号反転数")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.tight_layout()
    out_dir = ROOT / "results" / date.today().isoformat()
    base = "cim_zoom_1300_1500"
    v = next_version(out_dir, base)
    out = out_dir / f"v{v}_{base}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
