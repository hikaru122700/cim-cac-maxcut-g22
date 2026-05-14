"""CIM を 100 ラウンド回し、各頂点の in-phase 振幅 c_i の推移をプロットする。"""
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


def run_cim_record(n, J, num_rounds, rng, edges_np, *, kappa, L, gamma, eta,
                   bandwidth, photon_energy, dP_per_round):
    c = np.zeros(n, dtype=np.float64)
    Jc = np.zeros(n, dtype=np.float64)
    noise_const = np.sqrt((2.0 - eta) * 0.25 * bandwidth * photon_energy)
    sqrt_eta = np.sqrt(eta)

    history = np.zeros((num_rounds + 1, n), dtype=np.float64)
    cuts = np.zeros(num_rounds + 1, dtype=np.int64)
    pump = np.zeros(num_rounds + 1, dtype=np.float64)
    history[0] = c
    edge_a = edges_np[:, 0]
    edge_b = edges_np[:, 1]

    for k in range(num_rounds):
        P_p = (k + 1) * dP_per_round
        pump[k + 1] = P_p
        g0 = 2.0 * kappa * np.sqrt(P_p) * L

        Jc.fill(0.0)
        csr_matvec(n, n, J.indptr, J.indices, J.data, c, Jc)
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in
        half_g = 0.5 * g0 * (1.0 - gamma * I_in)
        sqrt_G_I = np.exp(half_g)
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)
        c = sqrt_G_I * coupled_in + N_I
        history[k + 1] = c
        signs = c > 0
        cuts[k + 1] = int((signs[edge_a] != signs[edge_b]).sum())

    return history, cuts, pump


def main():
    n, k, adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    print(f"N={n}, K={k}")

    J = build_coupling_matrix(n, edges, coupling=-0.03)
    rng = np.random.default_rng(42)

    NUM_ROUNDS = 1500
    edges_np = np.asarray(edges, dtype=np.int64)
    history, cuts, pump = run_cim_record(
        n, J, NUM_ROUNDS, rng, edges_np,
        kappa=130.0, L=0.05, gamma=42.09,
        eta=10.0 ** (-11.0 / 10.0),
        bandwidth=1.0e9, photon_energy=1.28e-19,
        dP_per_round=0.05e-3,
    )
    print(f"history shape: {history.shape}")
    print(f"final |c| range: [{np.abs(history[-1]).min():.3e}, {np.abs(history[-1]).max():.3e}]")
    print(f"final cut: {cuts[-1]},  best cut: {cuts.max()} at round {cuts.argmax()}")

    rounds = np.arange(NUM_ROUNDS + 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    sample_idx = np.random.default_rng(0).choice(n, size=30, replace=False)
    for idx in sample_idx:
        ax.plot(rounds, history[:, idx], lw=0.7, alpha=0.7)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅 c_i (in-phase)")
    ax.set_title(f"ランダムに選んだ30頂点の振幅推移({NUM_ROUNDS}ラウンド)")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[0, 1]
    mean_abs = np.mean(np.abs(history), axis=1)
    std_c = np.std(history, axis=1)
    max_abs = np.max(np.abs(history), axis=1)
    ax.plot(rounds, mean_abs, label="|c|の平均", lw=1.5)
    ax.plot(rounds, std_c, label="cの標準偏差", lw=1.5)
    ax.plot(rounds, max_abs, label="|c|の最大", lw=1.0, alpha=0.7)
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("振幅")
    ax.set_yscale("log")
    ax.set_title("集団統計(対数y軸)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 0]
    sigma_proxy = np.sqrt((2.0 - 10**(-1.1)) * 0.25 * 1e9 * 1.28e-19)
    signal_to_noise = mean_abs / (sigma_proxy + 1e-30)
    ax.plot(rounds, signal_to_noise, lw=1.5, color="C2")
    ax.axhline(1.0, color="r", lw=0.7, ls="--", label="SNR = 1")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("|c|の平均 / ノイズσ(参考値)")
    ax.set_yscale("log")
    ax.set_title("信号対雑音比(ノイズ支配→信号支配への遷移)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(direction="in", which="both", top=True, right=True)

    ax = axes[1, 1]
    ax.plot(rounds, cuts, lw=1.0, color="C3", label="cut(t)")
    ax.axhline(13359, color="k", lw=0.7, ls="--", label="既知最良解 13359")
    ax.axhline(19990 / 2, color="gray", lw=0.5, ls=":", label="ランダム期待値 K/2 = 9995")
    running_best = np.maximum.accumulate(cuts)
    ax.plot(rounds, running_best, lw=1.2, color="C4", label="running best")
    ax.set_xlabel("ラウンド")
    ax.set_ylabel("cut数")
    ax.set_title(f"カット数の推移(最終={cuts[-1]}, 最良={cuts.max()})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)

    fig.suptitle(
        f"CIM振幅ダイナミクス — {NUM_ROUNDS}ラウンド (G22, seed=42)",
        fontsize=13,
    )
    fig.tight_layout()

    out_dir = ROOT / "results" / date.today().isoformat()
    base = f"cim_amplitude_{NUM_ROUNDS}rounds"
    v = next_version(out_dir, base)
    out_path = out_dir / f"v{v}_{base}.png"
    fig.savefig(out_path, dpi=130)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
