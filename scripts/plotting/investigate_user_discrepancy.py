"""ユーザー提供スクリプトと本リポジトリ実装の「結果が違う」原因を調査する。

ユーザー版の特徴(本リポジトリ/私の図との差分):
  (1) 結合 = 1次元リング最近接のみ J[i,i±1]=±0.01(次数2)  ← 私は G22/密 次数20, ±0.03
  (2) ポンプ ΔP = 0.1 mW/step                            ← 私は 0.05 mW
  (3) 縦軸 = g_0(1−γI_in)(=強度ゲイン G_I の指数, 全量)  ← 私は ½g_0(1−γI_in)(振幅ゲインの指数)
  (4) steps = 1000                                        ← 私は 1500

本スクリプトは
  A: ユーザー版を忠実に再現(リング±0.01, ΔP=0.1, 全量 arg)→ 振幅図・arg図
  B: ユーザー枠組みのまま「結合だけ」密 次数20 ±0.03 に差し替え → 強磁性の挙動が変わるか
を出して、差の主因(=結合の実効ゲイン)を切り分ける。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/investigate_user_discrepancy.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---- ユーザー版と同一の物理定数 ----
N = 32
STEPS = 1000
ETA = 10 ** (-11.0 / 10.0)
KAPPA = 130.0
L = 0.05
GAMMA = 42.09
DELTA_P = 0.1 * 1e-3          # ユーザー: 0.1 mW/step
B = 1e9
H = 6.626e-34
C_SPEED = 3.0e8
LAM = 1550e-9
H_NU = H * C_SPEED / LAM
SEED = 0


def ring_J(coupling):
    J = np.zeros((N, N))
    for i in range(N):
        J[i, (i + 1) % N] = coupling
        J[(i + 1) % N, i] = coupling
    return J


def dense_J(coupling, deg=20, seed=7):
    """次数≈deg の ER ランダムグラフ(G22 相当の連結性)。"""
    rng = np.random.default_rng(seed)
    p = deg / (N - 1)
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p:
                J[i, j] = J[j, i] = coupling
    return J


def simulate(J, delta_P=DELTA_P, steps=STEPS, seed=SEED):
    """ユーザー版と同一の更新則。arg=g_0(1−γI_in)(全量)を記録。"""
    rng = np.random.default_rng(seed)
    c = np.zeros(N)
    c_hist = np.zeros((steps, N))
    arg_hist = np.zeros((steps, N))
    for k in range(1, steps):
        P_p0 = k * delta_P
        g_0 = 2.0 * KAPPA * np.sqrt(P_p0) * L
        c_iii = np.sqrt(ETA) * c + J @ c
        I_in = c_iii ** 2
        arg_val = g_0 * (1.0 - GAMMA * I_in)         # 全量(=G_I の指数)
        G_I = np.exp(arg_val)
        sigma_I = np.sqrt(0.25 * (2.0 - ETA) * G_I * H_NU * B)
        c = np.sqrt(G_I) * c_iii + rng.standard_normal(N) * sigma_I
        c_hist[k] = c
        arg_hist[k] = arg_val
    return c_hist, arg_hist


def lam_minmax(J, coupling):
    """J = coupling * A から A の固有値端と実効ゲインを返す。"""
    if coupling == 0:
        return 0.0, 0.0, np.sqrt(ETA)
    A = J / coupling
    ev = np.linalg.eigvalsh(A)
    lam_max, lam_min = float(ev[-1]), float(ev[0])
    # 整列モード(ferro)/反整列モード(anti)の実効振幅ゲイン端
    eff = np.sqrt(ETA) + max(coupling * lam_max, coupling * lam_min,
                             -coupling * lam_min, -coupling * lam_max)
    return lam_max, lam_min, eff


def _next_version_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / "cim_user_discrepancy"
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in root.iterdir():
        if q.is_dir() and q.name.startswith("v") and q.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(q.name.split("_", 1)[0][1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    half_g_th = 0.5 * np.log(1.0 / ETA)
    g_th = np.log(1.0 / ETA)            # 全量での発振しきい
    print(f"eta={ETA:.4f}  h_nu={H_NU:.3e}  発振しきい: 全量 ln(1/η)={g_th:.3f} / 半量={half_g_th:.3f}")
    print(f"P_th(全量しきい)到達 step ≈ {(g_th/(2*KAPPA*L))**2 / DELTA_P:.0f}  (ΔP={DELTA_P*1e3}mW)")

    # ---- ユーザー版: リング ±0.01 ----
    J_none = ring_J(0.0)
    J_anti = ring_J(-0.01)
    J_ferro = ring_J(+0.01)
    user_conditions = [
        ("相互作用なし $J=0$", J_none, 0.0),
        ("反強磁性(リング) $J=-0.01$", J_anti, -0.01),
        ("強磁性(リング) $J=+0.01$", J_ferro, +0.01),
    ]
    print("\n=== ユーザー版(リング最近接, 次数2) ===")
    user_c, user_arg = [], []
    for label, J, coup in user_conditions:
        ch, ah = simulate(J)
        user_c.append(ch); user_arg.append(ah)
        lmax, lmin, eff = lam_minmax(J, coup)
        std_late = float(np.std(ah[STEPS // 2:].mean(axis=1)))
        print(f"  {label:<26} λmax(A)={lmax:.2f} λmin(A)={lmin:.2f} 実効ゲイン={eff:.3f} "
              f"後半std={std_late:.3f}")

    # ---- 比較: 同じ枠組みで結合だけ密 次数20 ±0.03 ----
    Jf_dense = dense_J(+0.03)
    Ja_dense = dense_J(-0.03)
    cf_dense, af_dense = simulate(Jf_dense)
    ca_dense, aa_dense = simulate(Ja_dense)
    lmaxd, lmind, effd = lam_minmax(Jf_dense, 0.03)
    print("\n=== 比較: 密グラフ(次数20) ±0.03(私の設定相当) ===")
    print(f"  強磁性密 λmax(A)={lmaxd:.2f} 実効ゲイン={effd:.3f} "
          f"後半std={np.std(af_dense[STEPS//2:].mean(axis=1)):.3f}")

    out_dir = _next_version_dir("ring001_vs_dense003")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    k_plot = np.arange(STEPS)

    # ===== Fig A: ユーザー版 振幅発展(3条件) =====
    figA, axA = plt.subplots(1, 3, figsize=(18, 5.6))
    for idx, (label, _J, _c) in enumerate(user_conditions):
        ax = axA[idx]
        for i in range(N):
            ax.plot(k_plot, user_c[idx][:, i], marker="o", ms=3, markerfacecolor="none",
                    markeredgecolor="black", linestyle="none", alpha=0.3)
        ax.set_xlabel("$k$"); ax.set_ylabel("$c_i$")
        ax.set_ylim(-0.6, 0.6)
        ax.set_title(label, fontsize=13)
        ax.tick_params(direction="in", which="both", top=True, right=True)
    figA.suptitle("【ユーザー版を再現】振幅 $c_i(k)$ の発展（リング最近接 ±0.01, ΔP=0.1mW）"
                  "— 3条件とも似た挙動・強磁性も安定", fontsize=14)
    figA.tight_layout(rect=(0, 0, 1, 0.95))
    figA.savefig(out_dir / "A_user_amplitude.png", dpi=140)
    plt.close(figA)

    # ===== Fig B: ユーザー版 arg 中身(3条件) =====
    figB, axB = plt.subplots(1, 3, figsize=(18, 5.6))
    for idx, (label, _J, _c) in enumerate(user_conditions):
        ax = axB[idx]
        for i in range(N):
            ax.plot(k_plot, user_arg[idx][:, i], color="black", lw=0.8)
        mx = float(np.max(user_arg[idx]))
        ax.axhline(g_th, color="goldenrod", ls="--", lw=2,
                   label=f"発振しきい(全量) ln(1/η)={g_th:.2f}")
        ax.set_xlabel("$k$"); ax.set_ylabel(r"$g_0(1-\gamma I_{\rm in})$")
        ax.set_ylim(0, max(mx * 1.2, g_th * 1.3))
        ax.set_title(f"{label}\nMax={mx:.3f}", fontsize=12)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(fontsize=9, loc="lower right")
    figB.suptitle("【ユーザー版を再現】exp の中身 $g_0(1-\\gamma I_{\\rm in})$（全量）"
                  "— 強磁性も周期倍化せず安定", fontsize=14)
    figB.tight_layout(rect=(0, 0, 1, 0.95))
    figB.savefig(out_dir / "B_user_argcontent.png", dpi=140)
    plt.close(figB)

    # ===== Fig C: 強磁性 arg中身 リング±0.01 vs 密±0.03(主因の切り分け) =====
    figC, axC = plt.subplots(1, 2, figsize=(15, 5.8), sharey=False)
    # 左: ユーザーのリング ferro
    for i in range(N):
        axC[0].plot(k_plot, user_arg[2][:, i], color="black", lw=0.8)
    axC[0].axhline(g_th, color="goldenrod", ls="--", lw=2, label=f"しきい={g_th:.2f}")
    axC[0].set_title(f"強磁性・リング ±0.01（次数2, 実効ゲイン{lam_minmax(J_ferro,0.01)[2]:.2f}）\n"
                     "→ 安定（ユーザーの結果）", fontsize=12)
    axC[0].set_xlabel("$k$"); axC[0].set_ylabel(r"$g_0(1-\gamma I_{\rm in})$")
    axC[0].set_ylim(0, g_th * 1.4)
    axC[0].tick_params(direction="in", which="both", top=True, right=True)
    axC[0].legend(fontsize=9)
    # 右: 密 ferro ±0.03
    for i in range(N):
        axC[1].plot(k_plot, af_dense[:, i], color="#8e44ad", lw=0.7)
    axC[1].axhline(g_th, color="goldenrod", ls="--", lw=2, label=f"しきい={g_th:.2f}")
    axC[1].set_title(f"強磁性・密 次数20 ±0.03（実効ゲイン{effd:.2f}）\n"
                     "→ 周期倍化で不安定（私の結果）", fontsize=12)
    axC[1].set_xlabel("$k$"); axC[1].set_ylabel(r"$g_0(1-\gamma I_{\rm in})$")
    axC[1].tick_params(direction="in", which="both", top=True, right=True)
    axC[1].legend(fontsize=9)
    figC.suptitle("【主因の切り分け】同じ更新則・同じ ΔP・同じ縦軸（全量）で「結合だけ」変えた比較\n"
                  "結合がリング(次数2)では実効ゲインが小さく安定／密(次数20)では大きく周期倍化",
                  fontsize=13)
    figC.tight_layout(rect=(0, 0, 1, 0.92))
    figC.savefig(out_dir / "C_ferro_ring_vs_dense.png", dpi=140)
    plt.close(figC)

    # ===== Fig D: J 行列のイメージ(成分の可視化) =====
    def _draw_mat(ax, M, title, vmax):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("列 $j$"); ax.set_ylabel("行 $i$")
        ax.set_xticks([0, 8, 16, 24, 31]); ax.set_yticks([0, 8, 16, 24, 31])
        ax.tick_params(direction="out", length=3)
        import matplotlib.pyplot as _plt
        cb = _plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("$J_{ij}$")

    nnz_ring = int((J_ferro != 0).sum()); deg_ring = (J_ferro != 0).sum(1).mean()
    nnz_dense = int((Jf_dense != 0).sum()); deg_dense = (Jf_dense != 0).sum(1).mean()
    figD, axD = plt.subplots(1, 3, figsize=(17, 5.4))
    _draw_mat(axD[0], J_ferro,
              f"ユーザー: 強磁性リング +0.01\n各行2成分(i±1) 辺{nnz_ring//2} λmax={lam_minmax(J_ferro,0.01)[0]:.0f}",
              0.03)
    _draw_mat(axD[1], J_anti,
              f"ユーザー: 反強磁性リング −0.01\n同じ位置・符号だけ反転 辺{nnz_ring//2}",
              0.03)
    _draw_mat(axD[2], Jf_dense,
              f"私: 強磁性 密(次数≈20) +0.03\n各行≈{deg_dense:.0f}成分 辺{nnz_dense//2} λmax={lmaxd:.0f}",
              0.03)
    figD.suptitle("J 行列のイメージ(32×32, 赤=正/青=負/白=0) — リングは2本の対角線＋角だけ・"
                  "密は全体に散らばる", fontsize=13)
    figD.tight_layout(rect=(0, 0, 1, 0.94))
    figD.savefig(out_dir / "D_Jmatrices.png", dpi=140)
    plt.close(figD)

    # 成分の数値抜粋(左上 6×6)
    np.set_printoptions(precision=2, suppress=True)
    print("\n--- 強磁性リング J(左上6×6) ---")
    print(J_ferro[:6, :6])
    print("--- 強磁性 密 J(左上6×6) ---")
    print(Jf_dense[:6, :6])
    print(f"リング: 各行の非ゼロ数(次数)= {(J_ferro!=0).sum(1)[:8]} ...(全行2)")
    print(f"密    : 各行の非ゼロ数(次数)= {(Jf_dense!=0).sum(1)[:8]} ...(平均{deg_dense:.1f})")

    print(f"\nsaved: {out_dir/'A_user_amplitude.png'}")
    print(f"saved: {out_dir/'B_user_argcontent.png'}")
    print(f"saved: {out_dir/'C_ferro_ring_vs_dense.png'}")
    print(f"saved: {out_dir/'D_Jmatrices.png'}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
