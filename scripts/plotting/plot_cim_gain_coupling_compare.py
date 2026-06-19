"""CIM のゲイン(exp の中身)の rounds 特性を、結合の符号で比較する。

`plot_cim_gain_vs_rounds.py` を 3 つの結合条件に拡張:
  - 相互作用なし   J_ij = 0      (各パルスが独立)
  - 強磁性        J_ij = +0.03  (整列を好む)
  - 反強磁性      J_ij = -0.03  (反整列を好む = G22/MAX-CUT の標準)

縦軸 half_g = ½g₀(1−γI_in)(=`sqrt_G_I=exp(half_g)` の中身)、横軸 rounds。
非飽和 ½g₀(k)=κL√((k+1)ΔP) はポンプだけで決まるので **3 条件共通**。
変わるのは飽和込み実測ゲインの「立ち上がり round」と「その後の挙動」:

  - 無結合   : 各パルスが独立に発振しきいで飽和 → プラトー ≈ ½ln(1/η)(しきいちょうど)
  - 反強磁性 : 結合が実効利得を足し、しきい下のプラトー + 終盤に振幅不均一(帯が開く)
  - 強磁性   : 整列モードの固有値が大きく最速で立ち上がるが、暴走↔崩壊で発振が不安定

出力 2 枚:
  gain_coupling_panels.png  : 1×3 パネル(各条件を個別スケールで詳細表示)
  gain_coupling_overlay.png : 3 条件の実測平均(移動平均)を重ねた直接比較

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_cim_gain_coupling_compare.py
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

# ---- 物理パラメータ(modules/CIM.py main と同一) ----
KAPPA, L, GAMMA = 130.0, 0.05, 42.09
LOSS_DB = 11.0
ETA = 10.0 ** (-LOSS_DB / 10.0)
BANDWIDTH, PHOTON_ENERGY = 1.0e9, 1.28e-19
DP = 0.05e-3
NUM_ROUNDS = 1500
SEED = 42

# 比較する 3 条件: (key, ラベル, 結合係数, 色, 安定か, 短い説明)
CONDITIONS = [
    ("none",  "相互作用なし $J=0$",            0.0,   "#7f8c8d", True,  "各パルス独立 → しきいで飽和"),
    ("ferro", "強磁性 $J=+0.03$",              +0.03, "#8e44ad", False, "最速で立つが発振が不安定"),
    ("anti",  "反強磁性 $J=-0.03$(MAX-CUT)",  -0.03, "#c0392b", True,  "しきい下プラトー+終盤に振幅不均一"),
]


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


def run_and_record(n, edges, coupling):
    """単一 trial の CIM(指定結合)を回し、各 round の half_g 統計を記録。"""
    J = build_coupling_matrix(n, edges, coupling)
    sqrt_eta = np.sqrt(ETA)
    noise_const = np.sqrt((2.0 - ETA) * 0.25 * BANDWIDTH * PHOTON_ENERGY)
    rng = np.random.default_rng(SEED)
    c = np.zeros(n)

    half_g0 = np.zeros(NUM_ROUNDS)
    hg_mean = np.zeros(NUM_ROUNDS)
    hg_p10 = np.zeros(NUM_ROUNDS)
    hg_p90 = np.zeros(NUM_ROUNDS)

    for k in range(NUM_ROUNDS):
        P_p = (k + 1) * DP
        hg0 = KAPPA * np.sqrt(P_p) * L          # = ½g₀
        Jc = J @ c
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in
        half_g = hg0 * (1.0 - GAMMA * I_in)

        half_g0[k] = hg0
        hg_mean[k] = float(half_g.mean())
        hg_p10[k] = float(np.percentile(half_g, 10))
        hg_p90[k] = float(np.percentile(half_g, 90))

        sqrt_G_I = np.exp(half_g)
        c = sqrt_G_I * coupled_in + rng.standard_normal(n) * (noise_const * sqrt_G_I)

    return dict(half_g0=half_g0, hg_mean=hg_mean, hg_p10=hg_p10, hg_p90=hg_p90)


def _roll(a, w=21):
    """移動平均(可視化用)。端は実際に重なった要素数で正規化し、境界の落ち込みを防ぐ。"""
    if w <= 1:
        return a
    ker = np.ones(w)
    num = np.convolve(a, ker, mode="same")
    den = np.convolve(np.ones_like(a), ker, mode="same")   # 端での有効サンプル数
    return num / den


def _onset_plateau(rounds, hg0, hgm):
    rel_dev = (hg0 - hgm) / np.maximum(hg0, 1e-9)
    onset = int(rounds[int(np.argmax(rel_dev > 0.02))]) if np.any(rel_dev > 0.02) else 0
    mid = (rounds >= 400) & (rounds <= 1200)
    plateau = float(np.median(hgm[mid]))
    std_late = float(np.std(hgm[rounds >= 700]))   # 後半の振れ(不安定さの指標)
    return onset, plateau, std_late


def main() -> None:
    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    rounds = np.arange(1, NUM_ROUNDS + 1)
    half_g_th = 0.5 * np.log(1.0 / ETA)     # 非結合系の発振しきい

    recs = {}
    for key, label, coupling, color, stable, note in CONDITIONS:
        rec = run_and_record(n, edges, coupling)
        onset, plateau, std_late = _onset_plateau(rounds, rec["half_g0"], rec["hg_mean"])
        rec.update(onset=onset, plateau=plateau, std_late=std_late,
                   label=label, color=color, stable=stable, note=note)
        recs[key] = rec
        tag = "安定" if stable else "不安定"
        print(f"{label:<28} 立ち上がり≈round {onset:>4}  プラトー≈{plateau:.3f}  "
              f"後半std={std_late:.3f}({tag})")
    print(f"非結合系の発振しきい ½ln(1/η) = {half_g_th:.4f}")

    out_dir = _next_version_dir(f"3coupling_seed{SEED}_{NUM_ROUNDS}rounds")
    np.savez(out_dir / "data.npz", rounds=rounds, half_g_th=half_g_th,
             **{f"{k}_{q}": recs[k][q] for k in recs
                for q in ("half_g0", "hg_mean", "hg_p10", "hg_p90")})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    keys = [c[0] for c in CONDITIONS]
    hg0_shared = recs["none"]["half_g0"]   # 非飽和 ½g₀ は全条件共通

    # ===== Fig1: 1×3 パネル(個別スケール = sharey なし) =====
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), sharex=True)
    for ax, key in zip(axes, keys):
        r = recs[key]
        ax.plot(rounds, hg0_shared, color="#2c5f8a", lw=2.0,
                label=r"非飽和 $\frac{1}{2}g_0$(共通)")
        ax.fill_between(rounds, r["hg_p10"], r["hg_p90"], color=r["color"], alpha=0.18,
                        label="実測 p10–p90")
        ax.plot(rounds, r["hg_mean"], color=r["color"], lw=1.8,
                label=r"飽和込み $\frac{1}{2}g_0(1-\gamma I_{\rm in})$ 平均")
        ax.axhline(half_g_th, color="goldenrod", ls="--", lw=1.4,
                   label=fr"非結合しきい $\frac{{1}}{{2}}\ln(1/\eta)={half_g_th:.2f}$")
        if r["stable"]:
            ax.axhline(r["plateau"], color=r["color"], ls=":", lw=1.4,
                       label=f"プラトー ≈ {r['plateau']:.2f}")
        else:
            ax.text(0.5, 0.06, "発振が不安定（暴走↔崩壊を振動）", transform=ax.transAxes,
                    ha="center", fontsize=10, color=r["color"], style="italic")
        ax.axvline(r["onset"], color="black", ls="-.", lw=1.1,
                   label=f"立ち上がり ≈ round {r['onset']}")
        ax.set_title(f"{r['label']}\n{r['note']}", fontsize=11.5)
        ax.set_xlabel("rounds $k$", fontsize=12)
        ax.set_xlim(0, NUM_ROUNDS)
        ax.grid(alpha=0.3)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(loc="lower left" if not r["stable"] else "upper left", fontsize=8)
    axes[0].set_ylabel(r"exp() の中身  half_g $=\frac{1}{2}g_0(1-\gamma I_{\rm in})$", fontsize=12)
    fig.suptitle("CIM ゲイン(exp の中身)の rounds 特性 — 結合の符号で比較(G22, seed "
                 f"{SEED})  ※縦軸は各パネル個別スケール", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "gain_coupling_panels.png", dpi=150)
    plt.close(fig)

    # ===== Fig2: 重ね描き(実測平均の移動平均で直接比較, y クリップ) =====
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)
    ax.plot(rounds, hg0_shared, color="#2c5f8a", lw=2.4,
            label=r"非飽和 $\frac{1}{2}g_0$(ポンプが決める・3 条件共通)")
    for key in keys:
        r = recs[key]
        extra = "（移動平均・実際は不安定）" if not r["stable"] else ""
        ax.plot(rounds, _roll(r["hg_mean"], 25), color=r["color"], lw=2.2,
                label=f"{r['label']}：平均{extra}（立ち上がり≈{r['onset']}）")
    ax.axhline(half_g_th, color="goldenrod", ls="--", lw=1.5,
               label=fr"非結合系の発振しきい $\frac{{1}}{{2}}\ln(1/\eta)={half_g_th:.2f}$")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0, label=r"PSA 中立 (half_g=0)")
    ax.set_xlabel("rounds(周回数 $k$)", fontsize=13)
    ax.set_ylabel(r"exp() の中身  half_g $=\frac{1}{2}g_0(1-\gamma I_{\rm in})$", fontsize=13)
    ax.set_ylim(-1.0, 2.0)
    ax.set_title("CIM 実測ゲインの rounds 特性 — 結合の符号で比較(G22, seed "
                 f"{SEED})\n結合が強い/正なほど早く発振が立つ。無結合=しきいちょうど・"
                 "反強磁性=しきい下・強磁性=不安定（移動平均で表示, y は ±クリップ）", fontsize=11)
    ax.set_xlim(0, NUM_ROUNDS)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "gain_coupling_overlay.png", dpi=150)
    plt.close(fig)

    print(f"\nsaved: {out_dir / 'gain_coupling_panels.png'}")
    print(f"saved: {out_dir / 'gain_coupling_overlay.png'}")
    print(f"saved: {out_dir / 'data.npz'}")


if __name__ == "__main__":
    main()
