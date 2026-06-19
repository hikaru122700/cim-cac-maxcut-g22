"""パルス数 N=32 で CIM ゲイン特性・強磁性不安定を再検証する。

G22 の先頭 32 ノード誘導部分グラフは辺 6 本(平均次数 0.38)と疎すぎて、結合±0.03 では
ほぼ無結合になり強磁性不安定を検証できない。そこで **G22 と同等の連結性**
(平均次数≈20, λ_max(A)≈20 → 強磁性の実効ゲイン √η+0.03·λ_max ≈ 0.9 が G22 と一致)
を持つ 32 ノード ER ランダムグラフ(seed 固定)を使い、結合±0.03 で同じ実験を回す。

出力(2 枚):
  panels_32.png     : 1×3 パネル(無結合/強磁性/反強磁性)— plot_cim_gain_coupling_compare と同形式
  ferro_traj_32.png : 強磁性の全 32 パルス軌跡(周期倍化を直接表示)

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/verify_cim_gain_32pulses.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix  # noqa: E402

EXPERIMENT_KIND = "cim_gain_32pulses"
KAPPA, L, GAMMA = 130.0, 0.05, 42.09
ETA = 10.0 ** (-1.1)
BW, PH = 1.0e9, 1.28e-19
DP = 0.05e-3
NR = 1500
SIM_SEED = 42          # シミュレーション雑音の seed
GRAPH_SEED = 7         # 32 ノードグラフ生成の seed
N = 32
TARGET_DEG = 20        # G22 の平均次数に合わせる

CONDITIONS = [
    ("none",  "相互作用なし $J=0$",            0.0,   "#7f8c8d", True,  "各パルス独立"),
    ("ferro", "強磁性 $J=+0.03$",              +0.03, "#8e44ad", False, "周期倍化で不安定?"),
    ("anti",  "反強磁性 $J=-0.03$",            -0.03, "#c0392b", True,  "しきい下プラトー?"),
]


def make_graph():
    """N=32, 平均次数≈TARGET_DEG の ER ランダムグラフ(seed 固定)。"""
    rng = np.random.default_rng(GRAPH_SEED)
    p = TARGET_DEG / (N - 1)
    edges = [(i, j) for i in range(N) for j in range(i + 1, N) if rng.random() < p]
    A = np.zeros((N, N))
    for a, b in edges:
        A[a, b] = A[b, a] = 1.0
    lam = np.linalg.eigvalsh(A)
    return edges, A.sum(1), lam


def run(edges, coupling, track_all=False):
    J = build_coupling_matrix(N, edges, coupling)
    sqrt_eta = np.sqrt(ETA)
    nconst = np.sqrt((2 - ETA) * 0.25 * BW * PH)
    rng = np.random.default_rng(SIM_SEED)
    c = np.zeros(N)
    half_g0 = np.zeros(NR); hg_mean = np.zeros(NR)
    hg_p10 = np.zeros(NR); hg_p90 = np.zeros(NR)
    traj = np.zeros((NR, N)) if track_all else None
    for k in range(NR):
        hg0 = KAPPA * np.sqrt((k + 1) * DP) * L
        cin = sqrt_eta * c + J @ c
        Iin = cin * cin
        hg = hg0 * (1 - GAMMA * Iin)
        c = np.exp(hg) * cin + rng.standard_normal(N) * (nconst * np.exp(hg))
        half_g0[k] = hg0; hg_mean[k] = hg.mean()
        hg_p10[k] = np.percentile(hg, 10); hg_p90[k] = np.percentile(hg, 90)
        if track_all:
            traj[k] = c
    return dict(half_g0=half_g0, hg_mean=hg_mean, hg_p10=hg_p10, hg_p90=hg_p90, traj=traj)


def _onset_plateau(rounds, hg0, hgm):
    rel = (hg0 - hgm) / np.maximum(hg0, 1e-9)
    onset = int(rounds[int(np.argmax(rel > 0.02))]) if np.any(rel > 0.02) else 0
    mid = (rounds >= 400) & (rounds <= 1200)
    return onset, float(np.median(hgm[mid])), float(np.std(hgm[rounds >= 700]))


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
    edges, deg, lam = make_graph()
    sqrt_eta = np.sqrt(ETA)
    half_g_th = 0.5 * np.log(1.0 / ETA)
    rounds = np.arange(1, NR + 1)
    print(f"32ノードグラフ: E={len(edges)} avg_deg={deg.mean():.2f} "
          f"λmax(A)={lam[-1]:.3f} λmin(A)={lam[0]:.3f}")
    print(f"  強磁性 実効ゲイン √η+0.03·λmax = {sqrt_eta + 0.03*lam[-1]:.3f}  "
          f"(G22 は 0.914)")

    recs = {}
    for key, label, coupling, color, stable, note in CONDITIONS:
        r = run(edges, coupling, track_all=(key == "ferro"))
        onset, plateau, std_late = _onset_plateau(rounds, r["half_g0"], r["hg_mean"])
        r.update(onset=onset, plateau=plateau, std_late=std_late,
                 label=label, color=color, stable=stable, note=note)
        recs[key] = r
        print(f"{label:<22} 立ち上がり≈round {onset:>4}  プラトー≈{plateau:.3f}  "
              f"後半std={std_late:.3f}")
    print(f"非結合しきい ½ln(1/η)={half_g_th:.4f}")

    # 強磁性の分岐 round(mean half_g が初めて負)
    fm = recs["ferro"]["hg_mean"]
    neg = np.where(fm[50:] < 0)[0]
    bif = int(rounds[50 + neg[0]]) if len(neg) else NR
    print(f"強磁性の分岐(mean half_g<0)≈ round {bif}")

    out_dir = _next_version_dir(f"deg{TARGET_DEG}_seed{SIM_SEED}_{NR}rounds")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    keys = [c[0] for c in CONDITIONS]
    hg0 = recs["none"]["half_g0"]

    # ===== Fig1: 1×3 パネル =====
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), sharex=True)
    for ax, key in zip(axes, keys):
        r = recs[key]
        ax.plot(rounds, hg0, color="#2c5f8a", lw=2.0, label=r"非飽和 $\frac{1}{2}g_0$(共通)")
        ax.fill_between(rounds, r["hg_p10"], r["hg_p90"], color=r["color"], alpha=0.18,
                        label="実測 p10–p90")
        ax.plot(rounds, r["hg_mean"], color=r["color"], lw=1.6,
                label=r"飽和込み平均 $\frac{1}{2}g_0(1-\gamma I_{\rm in})$")
        ax.axhline(half_g_th, color="goldenrod", ls="--", lw=1.4,
                   label=fr"非結合しきい={half_g_th:.2f}")
        if r["stable"]:
            ax.axhline(r["plateau"], color=r["color"], ls=":", lw=1.4,
                       label=f"プラトー≈{r['plateau']:.2f}")
        else:
            ax.text(0.5, 0.06, "発振が不安定（周期倍化）", transform=ax.transAxes,
                    ha="center", fontsize=10, color=r["color"], style="italic")
        ax.axvline(r["onset"], color="black", ls="-.", lw=1.1,
                   label=f"立ち上がり≈{r['onset']}")
        ax.set_title(f"{r['label']}（後半std={r['std_late']:.2f}）", fontsize=11.5)
        ax.set_xlabel("rounds $k$", fontsize=12); ax.set_xlim(0, NR)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(loc="lower left" if not r["stable"] else "upper left", fontsize=8)
    axes[0].set_ylabel(r"exp() の中身 half_g $=\frac{1}{2}g_0(1-\gamma I_{\rm in})$", fontsize=12)
    fig.suptitle(f"【N=32 で再検証】CIM ゲインの rounds 特性 — 結合の符号で比較"
                 f"(32 パルス, 平均次数{TARGET_DEG}, seed {SIM_SEED})  ※縦軸個別スケール",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "panels_32.png", dpi=150)
    plt.close(fig)

    # ===== Fig2: 強磁性 全32パルス軌跡(周期倍化) =====
    traj = recs["ferro"]["traj"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8.4), dpi=130)
    a1.plot(rounds, hg0, color="#2c5f8a", lw=2.0, label=r"非飽和 $\frac{1}{2}g_0$")
    a1.plot(rounds, fm, color="#8e44ad", lw=1.4, label="飽和込み平均 half_g")
    a1.axhline(0, color="gray", ls=":", lw=1.0)
    a1.axvline(bif, color="black", ls="-.", lw=1.2, label=f"不安定化≈round {bif}")
    a1.set_xlim(0, NR); a1.set_xlabel("rounds $k$", fontsize=12); a1.set_ylabel("half_g", fontsize=12)
    a1.set_title("① 平均 half_g が負へ＝過飽和オーバーシュート開始", fontsize=11.5)
    a1.grid(alpha=0.3); a1.tick_params(direction="in", which="both", top=True, right=True)
    a1.legend(loc="upper left", fontsize=9)

    lo, hi = 600, 860
    for i in range(N):
        a2.plot(rounds[lo:hi], traj[lo:hi, i], lw=0.8, alpha=0.8)
    a2.axvline(bif, color="black", ls="-.", lw=1.2, label=f"不安定化≈round {bif}")
    a2.set_xlim(lo, hi); a2.set_xlabel("rounds $k$（ズーム）", fontsize=12)
    a2.set_ylabel("振幅 $c_i(k)$（全32パルス）", fontsize=12)
    a2.set_title("② 全32パルスの振幅が1周ごとに高↔低を交互（周期倍化→カオス）", fontsize=11.5)
    a2.grid(alpha=0.3); a2.tick_params(direction="in", which="both", top=True, right=True)
    a2.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"【N=32 で再検証】強磁性 CIM の周期倍化不安定（32 パルス, seed {SIM_SEED}）",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / "ferro_traj_32.png", dpi=150)
    plt.close(fig)

    np.savez(out_dir / "data.npz", rounds=rounds, edges=np.array(edges),
             lam=lam, half_g_th=half_g_th, bif=bif,
             **{f"{k}_{q}": recs[k][q] for k in recs
                for q in ("half_g0", "hg_mean", "hg_p10", "hg_p90")})
    print(f"\nsaved: {out_dir/'panels_32.png'}")
    print(f"saved: {out_dir/'ferro_traj_32.png'}")


if __name__ == "__main__":
    main()
