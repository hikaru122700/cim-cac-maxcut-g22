"""強磁性 CIM が ~round 700 で不安定化(ギザギザ)する原因を診断する。

`plot_cim_gain_coupling_compare.py` の強磁性条件(J=+0.03)で、
振幅が周期倍化(period-doubling)→カオスに入る様子を可視化する。

記録: 各 round の half_g0(=½g₀)・mean(half_g)・mean|c|、および個別パルスの c(k)。
出力: results/<date>/cim_gain_coupling/v{N}_ferro_instability_diag/ferro_instability.png

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/diag_cim_ferro_instability.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402

EXPERIMENT_KIND = "cim_gain_coupling"
KAPPA, L, GAMMA = 130.0, 0.05, 42.09
ETA = 10.0 ** (-1.1)
BW, PH = 1.0e9, 1.28e-19
DP = 0.05e-3
NR = 1500
SEED = 42
TRACK = [0, 1, 2]          # 軌跡を追うパルス


def _next_version_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in root.iterdir():
        if q.is_dir() and q.name.startswith("v") and q.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(q.name.split("_", 1)[0][1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    n, _k, _a, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J = build_coupling_matrix(n, edges, +0.03)
    sqrt_eta = np.sqrt(ETA)
    nconst = np.sqrt((2 - ETA) * 0.25 * BW * PH)
    rng = np.random.default_rng(SEED)
    c = np.zeros(n)

    rounds = np.arange(1, NR + 1)
    half_g0 = np.zeros(NR)
    hg_mean = np.zeros(NR)
    mean_abs = np.zeros(NR)
    traj = np.zeros((NR, len(TRACK)))

    for k in range(NR):
        P = (k + 1) * DP
        hg0 = KAPPA * np.sqrt(P) * L
        Jc = J @ c
        cin = sqrt_eta * c + Jc
        Iin = cin * cin
        hg = hg0 * (1 - GAMMA * Iin)
        sg = np.exp(hg)
        c = sg * cin + rng.standard_normal(n) * (nconst * sg)
        half_g0[k] = hg0
        hg_mean[k] = float(hg.mean())
        mean_abs[k] = float(np.abs(c).mean())
        for j, idx in enumerate(TRACK):
            traj[k, j] = c[idx]

    # 分岐(過飽和へのオーバーシュート)開始 = mean(half_g) が初めて負になる round
    warm = 50
    neg = np.where(hg_mean[warm:] < 0)[0]
    bif = int(rounds[warm + neg[0]]) if len(neg) else NR
    print(f"分岐(mean half_g<0)開始 ≈ round {bif}  (half_g0={half_g0[bif-1]:.3f})")
    print("pulse0 c(k) rounds 770-784:",
          np.array2string(traj[769:784, 0], precision=3, max_line_width=200))

    out_dir = _next_version_dir(f"ferro_instability_diag_seed{SEED}")
    np.savez(out_dir / "data.npz", rounds=rounds, half_g0=half_g0,
             hg_mean=hg_mean, mean_abs=mean_abs, traj=traj, bif=bif)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.4), dpi=130)

    # ---- 上: ½g₀ と mean(half_g) ----
    ax1.plot(rounds, half_g0, color="#2c5f8a", lw=2.0, label=r"非飽和 $\frac{1}{2}g_0$(ポンプ)")
    ax1.plot(rounds, hg_mean, color="#8e44ad", lw=1.6,
             label=r"飽和込み $\frac{1}{2}g_0(1-\gamma I_{\rm in})$ 平均")
    ax1.axhline(0.0, color="gray", ls=":", lw=1.1)
    ax1.axvline(bif, color="black", ls="-.", lw=1.2, label=f"不安定化 ≈ round {bif}")
    ax1.set_xlim(0, NR)
    ax1.set_xlabel("rounds $k$", fontsize=12)
    ax1.set_ylabel("half_g", fontsize=12)
    ax1.set_title("① ポンプ上昇で実測ゲインが過飽和へ振れ(平均 half_g が負に)、安定点が崩れる",
                  fontsize=11.5)
    ax1.grid(alpha=0.3)
    ax1.tick_params(direction="in", which="both", top=True, right=True)
    ax1.legend(loc="upper left", fontsize=9)

    # ---- 下: 個別パルス c(k) のズーム(周期倍化) ----
    lo, hi = 600, 860
    colors = ["#c0392b", "#16a085", "#d35400"]
    for j, idx in enumerate(TRACK):
        ax2.plot(rounds[lo:hi], traj[lo:hi, j], color=colors[j], lw=1.0, marker="o",
                 ms=2.5, label=f"パルス #{idx} の振幅 $c(k)$")
    ax2.axvline(bif, color="black", ls="-.", lw=1.2, label=f"不安定化 ≈ round {bif}")
    ax2.set_xlim(lo, hi)
    ax2.set_xlabel("rounds $k$(ズーム)", fontsize=12)
    ax2.set_ylabel("振幅 $c_i(k)$", fontsize=12)
    ax2.set_title("② 各パルスの振幅が 1 周ごとに「高↔低」を交互(周期倍化 → カオス)"
                  "。符号は反転せず大きさだけ振動", fontsize=11.5)
    ax2.grid(alpha=0.3)
    ax2.tick_params(direction="in", which="both", top=True, right=True)
    ax2.legend(loc="upper left", fontsize=9)

    fig.suptitle("強磁性 CIM(J=+0.03)が ~round 700 でギザギザになる理由 — "
                 "飽和増幅器の周期倍化不安定(G22, seed 42)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "ferro_instability.png", dpi=150)
    plt.close(fig)
    print(f"saved: {out_dir / 'ferro_instability.png'}")


if __name__ == "__main__":
    main()
