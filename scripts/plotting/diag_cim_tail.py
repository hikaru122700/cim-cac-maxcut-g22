"""末端の振動が数値不安定なのか可視化アーティファクトなのかを切り分ける診断。"""
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
    hist = np.zeros((num_rounds + 1, n))
    g0_hist = np.zeros(num_rounds + 1)
    pp_hist = np.zeros(num_rounds + 1)
    sigma_hist = np.zeros(num_rounds + 1)
    sqrtG_hist = np.zeros(num_rounds + 1)
    for kk in range(num_rounds):
        P_p = (kk + 1) * dP
        g0 = 2 * kappa * np.sqrt(P_p) * L
        Jc.fill(0)
        csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
        coup = sqrt_eta * c + Jc
        half_g = 0.5 * g0 * (1 - gamma * coup * coup)
        sg = np.exp(half_g)
        c_new = sg * coup + rng.standard_normal(n) * (noise_const * sg)
        hist[kk + 1] = c_new
        g0_hist[kk + 1] = g0
        pp_hist[kk + 1] = P_p
        sigma_hist[kk + 1] = (noise_const * sg).mean()
        sqrtG_hist[kk + 1] = sg.mean()
        c = c_new
    return hist, pp_hist, g0_hist, sigma_hist, sqrtG_hist, n


def main():
    hist, pp, g0, sigma, sqrtG, n = run()
    win = slice(1480, 1501)

    print("=" * 70)
    print("診断1: 1480-1500 のサンプル6頂点の c[n] 数値列")
    print("=" * 70)
    sample = [498, 166, 374, 1608, 183, 71]
    print(f"  ラウンド    " + " ".join(f"v{v:>5}" for v in sample))
    for r in range(1480, 1501):
        print(f"  round {r}:  " + " ".join(f"{hist[r, v]:+.4f}" for v in sample))

    print()
    print("=" * 70)
    print("診断2: step-to-step 差 Δc[n] = c[n+1] - c[n] の符号反転率")
    print("(数値不安定なら ステップ毎に符号反転 → 反転率 ~50%)")
    print("(単調成長なら 反転率はほぼ 0%)")
    print("=" * 70)
    diff = np.diff(hist, axis=0)                          # shape (1500, n)
    diff_win = diff[1300:1500]                            # 200 step
    # 連続するΔcの符号が反転した回数 / 全ステップ
    sign_flip_rate = np.mean(np.sign(diff_win[:-1]) * np.sign(diff_win[1:]) < 0)
    print(f"  Δc[n] と Δc[n+1] が逆符号の割合: {sign_flip_rate:.3%}")
    print(f"  全頂点平均 |Δc| (1300-1500)    : {np.abs(diff_win).mean():.3e}")
    print(f"  全頂点最大 |Δc| (1300-1500)    : {np.abs(diff_win).max():.3e}")
    print(f"  ノイズσの平均 (1300-1500)      : {sigma[1300:1500].mean():.3e}")
    print(f"  |Δc| / σ_noise の中央値        : "
          f"{np.median(np.abs(diff_win)) / sigma[1300:1500].mean():.2f}")

    print()
    print("=" * 70)
    print("診断3: 安定性指標 dt × 利得")
    print("=" * 70)
    print(f"  最終ラウンド g0(1500)          = {g0[1500]:.3f}")
    print(f"  最終ラウンド √G_I(1500) 平均   = {sqrtG[1500]:.3f}")
    print(f"  最終ラウンド σ_noise           = {sigma[1500]:.3e}")
    print(f"  飽和振幅 mean|c[1500]|         = {np.abs(hist[1500]).mean():.4f}")
    print(f"  飽和振幅 max|c[1500]|          = {np.abs(hist[1500]).max():.4f}")
    print(f"  理論限界 1/√(γη)               = "
          f"{1/np.sqrt(42.09 * 10**(-1.1)):.4f}")
    print(f"  飽和度 mean|c|/c_limit         = "
          f"{np.abs(hist[1500]).mean()/(1/np.sqrt(42.09*10**(-1.1))):.2%}")

    rounds = np.arange(len(hist))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    sample = [498, 166, 374, 183, 71, 1143]
    colors = plt.cm.tab10(np.arange(len(sample)))
    win_x = rounds[1480:1501]
    for i, v in enumerate(sample):
        ax.plot(win_x, hist[1480:1501, v], "o-", lw=1.2, ms=3,
                color=colors[i], label=f"頂点 {v}")
    ax.axhline(0, color="k", lw=0.4, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("c_i (各ステップを点でマーカー)")
    ax.set_title("最終20ラウンドの c_i ステップ毎値\n(振動なら点が上下にギザギザ)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[0, 1]
    ax2 = ax.twinx()
    mean_abs = np.abs(hist).mean(axis=1)
    ax.plot(rounds, mean_abs, "C0-", lw=1.2, label="|c|の平均")
    ax2.plot(rounds, pp * 1e3, "C3--", lw=1.0, label="ポンプ P_p [mW]")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("|c|の平均", color="C0")
    ax2.set_ylabel("ポンプ P_p [mW]", color="C3")
    ax.set_title("ポンプ vs 振幅(両軸の単調性を確認)")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True)
    ax2.tick_params(direction="in", which="both")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax = axes[1, 0]
    ax.plot(rounds[1:], np.median(np.abs(diff), axis=1), lw=1.0, label="median |Δc|")
    ax.plot(rounds[1:], np.max(np.abs(diff), axis=1), lw=0.7, alpha=0.6, label="max |Δc|")
    ax.plot(rounds, sigma, "C3--", lw=0.9, label="σ_noise")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("|Δc| = |c[n+1] − c[n]|")
    ax.set_yscale("log")
    ax.set_title("ステップ毎の振幅変化量(ノイズσと同レベルなら数値不安定ではない)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 1]
    diff_win = diff[1300:1500]
    same_dir = np.mean(np.sign(diff_win[:-1]) * np.sign(diff_win[1:]) > 0, axis=0)
    opp_dir = np.mean(np.sign(diff_win[:-1]) * np.sign(diff_win[1:]) < 0, axis=0)
    ax.hist(same_dir, bins=40, alpha=0.6, label="連続するΔcが同符号の割合(per頂点)")
    ax.axvline(0.5, color="k", ls="--", lw=0.5, label="0.5 (ホワイトノイズ並み)")
    ax.set_xlabel("ステップ毎Δcが連続して同方向に動く頻度")
    ax.set_ylabel("頂点数")
    ax.set_title(
        f"per-vertex 同符号率分布(全2000頂点)\n"
        f"全体平均: {1-sign_flip_rate:.3f}(数値不安定なら 0.5 付近)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle("末端振動の診断: 数値不安定 vs 単調収束", fontsize=13)
    fig.tight_layout()
    out_dir = ROOT / "results" / date.today().isoformat()
    base = "cim_tail_diagnosis"
    v = next_version(out_dir, base)
    out = out_dir / f"v{v}_{base}.png"
    fig.savefig(out, dpi=130)
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
