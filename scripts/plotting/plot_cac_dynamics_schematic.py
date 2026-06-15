"""CAC の「β_inj リセット周期 + 動的 a(t)」によるカオス探索の模式図。

実トレースではなく概念図（模式図）。β_inj のサワトゥース（成長→停滞τで0リセット）、
それに応じた a(t) の応答、cut（現在/最良）の関係を1枚で示す。
実トレースは run_cac_viz で再生可能。
"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "20260615" / "cac_dynamics_schematic.png"

alpha, rho = 3.0, 1.0          # a の基準・変調深さ
R, I, P = 180, 220, 600        # 各サイクル: 再探索 / 改善 / 停滞(≈τ)
incs = [28, 16, 9, 5, 3]       # サイクルごとの best 改善幅(逓減)
DIP = 70                       # リセット直後の cut 低下
SLOPE = 1.0                    # β_inj 成長率(模式)


def build():
    cur, beta, resets = [], [], []
    peak, t = 13180.0, 0
    for inc in incs:
        resets.append(t)
        new_peak = peak + inc
        r = np.linspace(peak - DIP, peak, R, endpoint=False)   # 再探索: 一度崩れて回復
        ii = np.linspace(peak, new_peak, I, endpoint=False)    # 改善: 新しい最良へ
        pp = np.full(P, new_peak)                              # 停滞: 最良で平坦
        cur.append(np.concatenate([r, ii, pp]))
        beta.append(SLOPE * np.arange(R + I + P))              # β_inj はサイクル頭で0→線形成長
        peak = new_peak
        t += R + I + P
    cur = np.concatenate(cur)
    beta = np.concatenate(beta)
    best = np.maximum.accumulate(cur)
    dH = 2.0 * (best - cur)                                    # ΔH = 2(best-cut)
    a = alpha + rho * np.tanh(0.03 * dH)
    return np.arange(len(cur)), cur, best, beta, a, resets


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    nu, cur, best, beta, a, resets = build()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.4), sharex=True)

    # Panel 1: cut
    ax1.plot(nu, cur, color="#7f8c8d", lw=1.0, label="現在の cut")
    ax1.plot(nu, best, color="#16a085", lw=2.2, label="最良 cut（更新で時計リセット）")
    ax1.set_ylabel("cut（大きいほど良い）")
    ax1.set_title("CAC のカオス探索：β_inj リセットと a(t) の連動（模式図）", fontsize=13)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel 2: beta_inj
    ax2.plot(nu, beta, color="#2c5f8a", lw=1.8)
    ax2.fill_between(nu, 0, beta, color="#2c5f8a", alpha=0.12)
    ax2.set_ylabel("結合スケール β_inj")
    ax2.annotate("線形に成長\n（結合が効き配置が結晶化）", (resets[0] + 360, beta[resets[0] + 360]),
                 xytext=(resets[0] + 120, beta.max() * 0.78), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="gray"))

    # Panel 3: a(t)
    ax3.plot(nu, a, color="#c0392b", lw=1.8)
    ax3.axhline(alpha, color="gray", ls=":", lw=1.0)
    ax3.axhline(alpha + rho, color="gray", ls=":", lw=1.0)
    ax3.text(nu[-1] * 0.995, alpha + 0.04, "α（基準・穏やか）", ha="right", fontsize=8, color="gray")
    ax3.text(nu[-1] * 0.995, alpha + rho - 0.12, "α+ρ（最大圧）", ha="right", fontsize=8, color="gray")
    ax3.set_ylabel("目標強度 a(t)")
    ax3.set_xlabel("外ループ step ν")
    ax3.set_ylim(alpha - 0.15, alpha + rho + 0.2)

    # リセット線（全パネル）+ 注記
    for ax in (ax1, ax2, ax3):
        for i, rt in enumerate(resets):
            ax.axvline(rt, color="red", ls="--", lw=1.1, alpha=0.7)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(alpha=0.25)
    ax2.annotate("τ ステップ改善なし\n→ β_inj=0 にリセット（再加熱）", (resets[1], 5),
                 xytext=(resets[1] + 80, beta.max() * 0.45), fontsize=8.5, color="red",
                 arrowprops=dict(arrowstyle="->", color="red"))
    ax3.annotate("リセット直後は配置が崩れ\n現在解が悪化 → a↑（強く揺さぶる）",
                 (resets[1] + 40, a[resets[1] + 40]),
                 xytext=(resets[1] + 220, alpha + rho - 0.45), fontsize=8.5, color="#c0392b",
                 arrowprops=dict(arrowstyle="->", color="#c0392b"))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
