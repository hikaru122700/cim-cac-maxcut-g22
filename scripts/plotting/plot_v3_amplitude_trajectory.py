"""v3 swap アブレーション実験の「振幅のラウンド変化」単独図を描く。

各レプリカ(ポンプ倍率 mult=0.8/1.0/1.3)の平均振幅 mean|c| を
ラウンド数に対してプロットする。trial 間のばらつきを帯(±1σ)で示し、
各ランプが発振しきい値 P_th=38mW を横切るラウンドを縦線で注記する。

使い方(プロジェクトルートから):
    python scripts/plotting/plot_v3_amplitude_trajectory.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

DATA = Path(
    "results/2026-06-08/cim_pt_v3_swap_ablation/"
    "v1_rounds1500_swap10_swap_ablation/data.npz"
)
OUT = DATA.parent / "amplitude_vs_rounds.png"

# 物理パラメータ(実験と同一) — しきい値交差ラウンドの算出用
P_TH = 38.0          # 発振しきい値 [mW]
DP = 0.05            # 1 ラウンドあたりのランプ増分 [mW]

COLORS = ["#d62728", "#ff7f0e", "#1f77b4"]  # replica0/1/2


def threshold_round(mult: float) -> float:
    """P_r(k)=mult*(k+1)*dP が P_th を超える(1始まりの)ラウンド数。"""
    return P_TH / (mult * DP)


def main() -> None:
    d = np.load(DATA)
    rounds = d["sample_rounds"]            # (60,)
    amp = d["v3_traj_amp"]                 # (100 trial, 60, 3 replica)
    mults = d["pump_mults"]               # [0.8, 1.0, 1.3]

    mean = amp.mean(axis=0)               # (60, 3)
    std = amp.std(axis=0)                 # (60, 3)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for r, mult in enumerate(mults):
        c = COLORS[r]
        ax.plot(rounds, mean[:, r], color=c, lw=2.2,
                label=f"replica{r} mult={mult:g}")
        ax.fill_between(rounds, mean[:, r] - std[:, r], mean[:, r] + std[:, r],
                        color=c, alpha=0.15, linewidth=0)
        # しきい値交差ラウンドを縦線で注記
        kc = threshold_round(mult)
        ax.axvline(kc, color=c, ls=":", lw=1.3, alpha=0.8)
        ax.text(kc, 0.012, f"発振しきい値通過\nround {kc:.0f}",
                color=c, fontsize=8.5, ha="center", va="bottom")

    ax.set_xlabel("ラウンド数")
    ax.set_ylabel("平均振幅 mean|c|(trial 平均 ±1σ)")
    ax.set_title("v3 各レプリカの振幅のラウンド変化(G22, 100 trial)")
    ax.set_xlim(0, rounds.max())
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved: {OUT.resolve()}")


if __name__ == "__main__":
    main()
