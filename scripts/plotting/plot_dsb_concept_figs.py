"""plot_dsb_concept_figs.py — dSB 解説用の概念図を 3 点生成。

実装(modules/SB.py)に忠実な式で描く:
  (1) 復元係数ランプ a0 - a(t) の時間変化
  (2) 傾いた二重井戸ポテンシャル V(x)=1/2 (a0-a) x^2 - h x (|x|<=1 壁)
  (3) 小インスタンスで dSB を実走させた x_i(t) 軌道(分岐)＋カット推移

出力: docs/dSB_figs/{ramp,potential,trajectory}_v1.png
使い方: python scripts/plotting/plot_dsb_concept_figs.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "dSB_figs"
OUT.mkdir(parents=True, exist_ok=True)

A0 = 1.0
DT = 0.5


# ============ (1) 復元係数ランプ ============
def fig_ramp():
    steps = 1000
    k = np.arange(1, steps + 1)
    a_t = k / steps * A0          # a(t): 0 -> a0
    a_diff = A0 - a_t             # (a0 - a(t)): a0 -> 0
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(k / steps, a_t, color="#2980b9", lw=2.2, label="ポンプ a(t)（0→a0）")
    ax.plot(k / steps, a_diff, color="#e74c3c", lw=2.2,
            label="自己復元係数 a0−a(t)（a0→0）")
    ax.set_xlabel("規格化時間 t / T（ステップ進行度）")
    ax.set_ylabel("係数の大きさ")
    ax.set_title("dSB の時間スケジュール：自己復元を徐々に弱める")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "ramp_v1.png", dpi=130)
    plt.close(fig)


# ============ (2) 傾いた二重井戸ポテンシャル ============
def fig_potential():
    x = np.linspace(-1.1, 1.1, 400)
    h = 0.25  # 局所場 c0*(J sign(x))_i の一例（正なら +1 側へ傾く）
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for a_diff, color, lab in [(0.9, "#9b59b6", "序盤 a0−a≈0.9（深い単一井戸）"),
                               (0.35, "#16a085", "中盤 a0−a≈0.35"),
                               (0.03, "#e74c3c", "終盤 a0−a≈0.03（平坦＋傾き）")]:
        V = 0.5 * a_diff * x ** 2 - h * x
        ax.plot(x, V, color=color, lw=2.2, label=lab)
    ax.axvline(1.0, color="gray", ls="--", lw=1.0)
    ax.axvline(-1.0, color="gray", ls="--", lw=1.0)
    ax.text(1.02, ax.get_ylim()[1] * 0.7, "壁 x=+1", color="gray", fontsize=9)
    ax.set_xlabel("振動子の変位 x_i")
    ax.set_ylabel("実効ポテンシャル V(x)")
    ax.set_title("局所場 h>0 のときの実効ポテンシャル：\n復元が弱まると最小が +1 側の壁へ移動")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=9, loc="upper center")
    fig.tight_layout()
    fig.savefig(OUT / "potential_v1.png", dpi=130)
    plt.close(fig)


# ============ (3) 実 dSB 軌道（小インスタンス） ============
def run_dsb_traj(n, edges, num_steps, seed=0, dt=DT):
    """modules/SB.py の dSB を忠実に再現し x の全軌道を記録(小 n 用, plain numpy)。"""
    rng = np.random.default_rng(seed)
    # CSR 相当の隣接(対称)。J_ij = -1（repo の coupling=-1.0, +1 重み）
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    # c0 = 0.5 sqrt((n-1)/ sum J^2),  sum J^2 = 2E（各辺2エントリ, 値^2=1）
    sum_J2 = 2.0 * len(edges)
    c0 = 0.5 * np.sqrt((n - 1) / sum_J2)

    x = (rng.random(n) - 0.5) * 2.0 * 0.1
    y = (rng.random(n) - 0.5) * 2.0 * 0.1
    xs = np.zeros((num_steps + 1, n))
    cuts = np.zeros(num_steps + 1)
    xs[0] = x

    def cut_of(xv):
        c = 0
        for a, b in edges:
            if (xv[a] > 0) != (xv[b] > 0):
                c += 1
        return c
    cuts[0] = cut_of(x)

    for k in range(num_steps):
        a_t = (k + 1) / num_steps * A0
        a_diff = A0 - a_t
        Jx = np.empty(n)
        for i in range(n):
            acc = 0.0
            for j in adj[i]:
                acc += -1.0 * (1.0 if x[j] > 0 else -1.0)  # J_ij=-1 * sign(x_j)
            Jx[i] = acc
        y = y + dt * (-a_diff * x + c0 * Jx)      # dSB: Kerr なし
        x = x + dt * A0 * y                        # symplectic Euler
        # 壁拘束
        over = x > 1.0
        under = x < -1.0
        x[over] = 1.0; y[over] = 0.0
        x[under] = -1.0; y[under] = 0.0
        xs[k + 1] = x
        cuts[k + 1] = cut_of(x)
    return xs, cuts


def fig_trajectory():
    # 疎な偶数サイクル C_n（二部グラフ＝最大カットは全辺）で分岐をクリアに見せる
    n = 14
    edges = sorted({(i, (i + 1) % n) if i < (i + 1) % n else ((i + 1) % n, i)
                    for i in range(n)})
    num_steps = 1200
    dt = 0.35  # 教育用にやや小さめ（より断熱的）。実装既定は 0.5
    xs, cuts = run_dsb_traj(n, edges, num_steps, seed=2, dt=dt)
    t = np.arange(num_steps + 1) / num_steps

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    # 最終符号で色分け（+1=赤系, -1=青系）
    for i in range(n):
        col = "#e74c3c" if xs[-1, i] > 0 else "#2980b9"
        ax1.plot(t, xs[:, i], lw=1.4, alpha=0.85, color=col)
    ax1.axhline(1.0, color="gray", ls="--", lw=1.0)
    ax1.axhline(-1.0, color="gray", ls="--", lw=1.0)
    ax1.axhline(0.0, color="black", ls=":", lw=0.8)
    ax1.text(0.02, 1.04, "+1 側（赤）", color="#e74c3c", fontsize=9)
    ax1.text(0.02, -1.12, "−1 側（青）", color="#2980b9", fontsize=9)
    ax1.set_xlabel("規格化時間 t / T")
    ax1.set_ylabel("各振動子の変位 x_i(t)")
    ax1.set_title("dSB の分岐：x=0 付近から ±1 へ枝分かれ（偶数サイクル C14）")
    ax1.set_ylim(-1.25, 1.25)
    ax1.grid(True, ls=":", alpha=0.4)

    ax2.plot(t, cuts, color="#16a085", lw=2.0)
    ax2.axhline(n, color="gray", ls="--", lw=1.0)
    ax2.text(0.02, n - 1.3, f"最大カット={n}（C{n} は二部グラフ）", color="gray", fontsize=9)
    ax2.set_xlabel("規格化時間 t / T")
    ax2.set_ylabel("カット数（その時刻の sign(x)）")
    ax2.set_title(f"カット数の推移（最終到達={int(cuts.max())}）")
    ax2.grid(True, ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_v1.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig_ramp()
    fig_potential()
    fig_trajectory()
    print("[OK] wrote", OUT / "ramp_v1.png")
    print("[OK] wrote", OUT / "potential_v1.png")
    print("[OK] wrote", OUT / "trajectory_v1.png")
