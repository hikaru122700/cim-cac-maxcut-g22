"""8頂点グラフでの CIM 振幅 c_i(k) を 1000 ラウンド可視化する。"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main():
    n = 8
    num_rounds = 1000
    seed = 42

    # 8頂点のサンプルグラフ(8角形 + いくつかの対角線)
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0),
             (0, 3), (1, 4), (2, 5), (3, 6), (0, 5)]
    J = np.zeros((n, n))
    for a, b in edges:
        J[a, b] = -0.03
        J[b, a] = -0.03

    kappa = 130.0
    L = 0.05
    gamma = 42.09
    eta = 10.0 ** (-1.1)
    BW = 1.0e9
    hw = 1.28e-19
    dP = 0.05e-3

    noise_const = np.sqrt((2.0 - eta) * 0.25 * BW * hw)
    sqrt_eta = np.sqrt(eta)

    rng = np.random.default_rng(seed)
    c = np.zeros(n)
    history = np.zeros((num_rounds, n))
    cut_hist = np.zeros(num_rounds, dtype=int)
    best_hist = np.zeros(num_rounds, dtype=int)
    best_cut = 0

    for k in range(num_rounds):
        Pp = (k + 1) * dP
        g0 = 2.0 * kappa * np.sqrt(Pp) * L
        coupled = sqrt_eta * c + J @ c
        I_in = coupled * coupled
        sqrt_GI = np.exp(0.5 * g0 * (1.0 - gamma * I_in))
        N_I = rng.standard_normal(n) * (noise_const * sqrt_GI)
        c = sqrt_GI * coupled + N_I
        history[k] = c
        s = (c > 0).astype(int)
        cut = sum(1 for a, b in edges if s[a] != s[b])
        cut_hist[k] = cut
        if cut > best_cut:
            best_cut = cut
        best_hist[k] = best_cut

    final_signs = np.sign(history[-1])
    print(f"最終符号: {final_signs.astype(int)}")
    print(f"最終振幅 c_i: {history[-1]}")

    cut = sum(1 for a, b in edges if final_signs[a] != final_signs[b])
    print(f"カット数 (辺数 {len(edges)}): {cut}")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)
    colors = plt.cm.tab10(np.arange(n))
    t = np.arange(num_rounds)
    for i in range(n):
        ax.plot(t, history[:, i], color=colors[i], linewidth=1.2,
                label=f"パルス {i+1} (最終符号 {'+' if final_signs[i] > 0 else '−'})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("ラウンド $k$")
    ax.set_ylabel(r"in-phase 振幅 $c_i(k)$")
    ax.set_title(
        f"CIM 振幅推移: $N={n}$ 頂点, {num_rounds} ラウンド "
        f"(辺数 $K={len(edges)}$, カット数 = {cut})"
    )
    ax.legend(fontsize=9, loc="best", ncol=2)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    os.makedirs("results", exist_ok=True)

    def next_version(base):
        existing = [int(p.split("_")[0][1:]) for p in os.listdir("results")
                    if p.startswith("v") and p.endswith(f"_{base}.png")
                    and p.split("_")[0][1:].isdigit()]
        return (max(existing) if existing else 0) + 1

    base1 = "amplitude_8nodes_1000rounds"
    v1 = next_version(base1)
    out1 = f"results/v{v1}_{base1}.png"
    fig.savefig(out1)
    print(f"Saved: {out1}")

    # --- カット数推移のグラフ ---
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=130)
    t = np.arange(num_rounds)
    ax2.plot(t, cut_hist, color="#1f77b4", alpha=0.4, linewidth=0.9,
             label="現在のカット数 $\\mathrm{cut}(k)$")
    ax2.plot(t, best_hist, color="#d62728", linewidth=2.2,
             label="ベスト更新値 $\\mathrm{best\\_cut}$")
    ax2.axhline(len(edges), color="goldenrod", linestyle="--", linewidth=1.3,
                label=f"全辺数 $K = {len(edges)}$")
    ax2.set_xlabel("ラウンド $k$")
    ax2.set_ylabel("カット数")
    ax2.set_title(
        f"CIM カット数推移: $N={n}$ 頂点, {num_rounds} ラウンド "
        f"(最終 $\\mathrm{{best\\_cut}} = {best_cut}$)"
    )
    ax2.set_ylim(-0.5, len(edges) + 1.0)
    ax2.legend(fontsize=10, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.tick_params(direction="in", which="both", top=True, right=True)
    fig2.tight_layout()

    base2 = "cut_history_8nodes_1000rounds"
    v2 = next_version(base2)
    out2 = f"results/v{v2}_{base2}.png"
    fig2.savefig(out2)
    print(f"Saved: {out2}")
    print(f"best_cut over {num_rounds} rounds: {best_cut} / {len(edges)}")


if __name__ == "__main__":
    main()
