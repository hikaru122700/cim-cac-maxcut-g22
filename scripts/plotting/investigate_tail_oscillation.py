"""round 1300〜1500 で本当に振幅が上下しているかを精査する。"""
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


def simulate(num_rounds=1500, seed=42):
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
    sigma_hist = np.zeros(num_rounds + 1)
    for kk in range(num_rounds):
        P_p = (kk + 1) * dP
        g0 = 2 * kappa * np.sqrt(P_p) * L
        Jc.fill(0)
        csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
        coup = sqrt_eta * c + Jc
        half_g = 0.5 * g0 * (1 - gamma * coup * coup)
        sg = np.exp(half_g)
        sigma_hist[kk + 1] = (noise_const * sg).mean()
        c = sg * coup + rng.standard_normal(n) * (noise_const * sg)
        hist[kk + 1] = c
    return hist, sigma_hist, n


def main():
    hist, sigma_hist, n = simulate()
    win = slice(1300, 1501)
    sub = hist[win]                                   # shape (201, n)

    # 各頂点の |Δc| を 1300〜1500 で評価
    delta = sub.max(axis=0) - sub.min(axis=0)          # 各頂点ごとの振幅レンジ
    print("=== 1300〜1500 区間の各頂点 c の振幅レンジ ===")
    print(f"全頂点の Δc 平均: {delta.mean():.5f}")
    print(f"全頂点の Δc 中央値: {np.median(delta):.5f}")
    print(f"全頂点の Δc 最大: {delta.max():.5f}")
    print(f"Δc > 0.05 の頂点数: {(delta > 0.05).sum()} / {n}")
    print(f"Δc > 0.01 の頂点数: {(delta > 0.01).sum()} / {n}")
    print(f"ノイズσの平均(1300-1500): {sigma_hist[win].mean():.3e}")
    print(f"σ × 5 (5σ 帯): {sigma_hist[win].mean() * 5:.3e}")

    # 一番振幅変動が大きい上位 6 頂点
    top_idx = np.argsort(delta)[-6:][::-1]
    print(f"\n最も変動が大きい頂点 top6:")
    for i, v in enumerate(top_idx):
        print(f"  #{i+1} vertex {v}: Δc={delta[v]:.4f}, "
              f"c[1300]={hist[1300, v]:+.4f}, c[1500]={hist[1500, v]:+.4f}, "
              f"sign 同じ? {(hist[1300, v] > 0) == (hist[1500, v] > 0)}")

    rounds_full = np.arange(len(hist))
    rounds_win = rounds_full[win]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, vid in zip(axes.flat, top_idx):
        ax.plot(rounds_win, sub[:, vid], lw=1.0, color="C0")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.fill_between(rounds_win,
                        sub[:, vid] - 2*sigma_hist[win],
                        sub[:, vid] + 2*sigma_hist[win],
                        color="C0", alpha=0.2, label="±2σ_noise")
        ax.set_xlabel("ラウンド")
        ax.set_ylabel("振幅 c")
        ax.set_title(f"頂点 {vid}: Δc={delta[vid]:.4f}, "
                     f"c範囲[{sub[:, vid].min():+.3f},{sub[:, vid].max():+.3f}]")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle("1300〜1500ラウンドで最も振幅変動が大きい6頂点(個別軌道)",
                 fontsize=13)
    fig.tight_layout()
    out_dir = ROOT / "results" / date.today().isoformat()
    base1 = "cim_tail_top_movers"
    v1 = next_version(out_dir, base1)
    out1 = out_dir / f"v{v1}_{base1}.png"
    fig.savefig(out1, dpi=130)
    print(f"\nsaved: {out1}")

    # ヒストグラム: 全頂点の Δc 分布
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(delta, bins=80, color="C2", edgecolor="k", lw=0.3)
    ax[0].axvline(sigma_hist[win].mean() * 5, color="r", ls="--",
                  label=f"5σ_noise = {sigma_hist[win].mean()*5:.2e}")
    ax[0].set_xlabel("Δc (1300〜1500の振幅レンジ)")
    ax[0].set_ylabel("頂点数")
    ax[0].set_title("各頂点の振幅変動 Δc の分布(全2000頂点)")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[0].set_yscale("log")
    ax[0].tick_params(direction="in", which="both", top=True, right=True)

    # |c| 平均の時系列(1300-1500)とノイズσ
    mean_abs = np.abs(sub).mean(axis=1)
    ax[1].plot(rounds_win, mean_abs, label="|c|の平均", lw=1.5)
    ax[1].plot(rounds_win, sigma_hist[win] * 5, label="5σ_noise", lw=1.0)
    ax[1].set_xlabel("ラウンド")
    ax[1].set_ylabel("値")
    ax[1].set_title("|c|平均 vs ノイズσ(両方とも単調増加)")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[1].tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()
    base2 = "cim_tail_delta_dist"
    v2 = next_version(out_dir, base2)
    out2 = out_dir / f"v{v2}_{base2}.png"
    fig.savefig(out2, dpi=130)
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
