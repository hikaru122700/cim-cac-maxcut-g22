"""CIM のゲイン(exp の中身)の rounds 特性を描く。

`modules/CIM.py` の振幅更新は

    half_g  = 0.5 * g0 * (1 - gamma * I_in)      # = exp() の中身
    sqrt_G_I = exp(half_g)
    c = sqrt_G_I * coupled_in + noise

で、`sqrt_G_I = exp(half_g)` の **half_g が「exp() の中身」**(振幅ゲインの指数)。
本図は横軸 rounds・縦軸 half_g として、次の 2 本を比較する。

  - 非飽和    ½g₀(k) = κL√((k+1)ΔP)           … ポンプが決める利得(I_in=0)
  - 飽和込み  ½g₀(1−γI_in)(実測, 全パルス平均)  … 実際に効くゲイン

g₀(k)=2κL√(P(k)), P(k)=(k+1)ΔP。発振しきいは 1 周の正味振幅倍率
√η·exp(½g₀)=1、すなわち half_g = ½ln(1/η)。飽和は実測 half_g をこのしきい値
付近にクランプする(定常では √η·exp(half_g)=1)。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_cim_gain_vs_rounds.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modules.CIM import build_coupling_matrix, load_graph  # noqa: E402

EXPERIMENT_KIND = "cim_gain_rounds"

# ---- 物理パラメータ(modules/CIM.py main と同一) ----
KAPPA, L, GAMMA = 130.0, 0.05, 42.09
LOSS_DB = 11.0
ETA = 10.0 ** (-LOSS_DB / 10.0)
BANDWIDTH, PHOTON_ENERGY = 1.0e9, 1.28e-19
DP = 0.05e-3            # W/round
COUPLING = -0.03
NUM_ROUNDS = 1500
SEED = 42


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


def run_and_record():
    """単一 trial の CIM を回し、各 round の half_g 統計を記録する。"""
    n, k_edges, _adj, edges = load_graph(str(ROOT / "input" / "G22.txt"))
    J = build_coupling_matrix(n, edges, COUPLING)
    sqrt_eta = np.sqrt(ETA)
    noise_const = np.sqrt((2.0 - ETA) * 0.25 * BANDWIDTH * PHOTON_ENERGY)
    rng = np.random.default_rng(SEED)
    c = np.zeros(n)

    rounds = np.arange(1, NUM_ROUNDS + 1)
    half_g0 = np.zeros(NUM_ROUNDS)     # 非飽和 ½g₀(k)
    hg_mean = np.zeros(NUM_ROUNDS)     # 飽和込み ½g₀(1−γI_in) 平均
    hg_p10 = np.zeros(NUM_ROUNDS)
    hg_p90 = np.zeros(NUM_ROUNDS)
    mean_Iin = np.zeros(NUM_ROUNDS)
    mean_abs_c = np.zeros(NUM_ROUNDS)

    for k in range(NUM_ROUNDS):
        P_p = (k + 1) * DP
        g0 = 2.0 * KAPPA * np.sqrt(P_p) * L
        hg0 = 0.5 * g0

        Jc = J @ c
        coupled_in = sqrt_eta * c + Jc
        I_in = coupled_in * coupled_in
        half_g = hg0 * (1.0 - GAMMA * I_in)   # ← exp() の中身(パルスごと)

        half_g0[k] = hg0
        hg_mean[k] = float(half_g.mean())
        hg_p10[k] = float(np.percentile(half_g, 10))
        hg_p90[k] = float(np.percentile(half_g, 90))
        mean_Iin[k] = float(I_in.mean())
        mean_abs_c[k] = float(np.abs(c).mean())

        sqrt_G_I = np.exp(half_g)
        N_I = rng.standard_normal(n) * (noise_const * sqrt_G_I)
        c = sqrt_G_I * coupled_in + N_I

    return dict(n=n, k_edges=k_edges, rounds=rounds, half_g0=half_g0,
                hg_mean=hg_mean, hg_p10=hg_p10, hg_p90=hg_p90,
                mean_Iin=mean_Iin, mean_abs_c=mean_abs_c)


def main() -> None:
    rec = run_and_record()
    rounds = rec["rounds"]

    # 非結合系の発振しきい値: √η·exp(½g₀)=1 → half_g = ½ln(1/η)
    half_g_th = 0.5 * np.log(1.0 / ETA)
    P_th = (np.log(1.0 / ETA) / (2.0 * KAPPA * L)) ** 2          # W
    cross_round = (half_g_th / (KAPPA * L)) ** 2 / DP           # 非飽和がしきい超え

    # 飽和の効き始め: 実測平均が非飽和から 2% 以上ずれる最初の round
    hg0, hgm = rec["half_g0"], rec["hg_mean"]
    rel_dev = (hg0 - hgm) / np.maximum(hg0, 1e-9)
    onset_idx = int(np.argmax(rel_dev > 0.02)) if np.any(rel_dev > 0.02) else 0
    onset_round = int(rounds[onset_idx])
    # 飽和プラトー: 中盤(round 400–1200)の実測平均の中央値
    mid = (rounds >= 400) & (rounds <= 1200)
    plateau = float(np.median(hgm[mid]))

    print(f"eta={ETA:.4f}  非結合しきい ½ln(1/η)={half_g_th:.4f}  P_th={P_th*1e3:.2f} mW")
    print(f"非飽和 ½g₀: round1={hg0[0]:.4f} → round{NUM_ROUNDS}={hg0[-1]:.4f}"
          f" / 非飽和がしきい超え ≈ round {cross_round:.0f}")
    print(f"飽和の効き始め ≈ round {onset_round}  /  飽和プラトー ≈ {plateau:.3f}"
          f"(非結合しきい {half_g_th:.3f} より下)")
    print(f"飽和込み 実測平均: round{NUM_ROUNDS}={hgm[-1]:.4f}")

    out_dir = _next_version_dir(f"seed{SEED}_{NUM_ROUNDS}rounds")
    np.savez(out_dir / "data.npz", **{k: v for k, v in rec.items()
                                       if isinstance(v, np.ndarray)},
             half_g_th=half_g_th, P_th=P_th, cross_round=cross_round)

    # ---- 描画 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=130)

    # 非飽和 ½g₀(ポンプが決める利得)
    ax.plot(rounds, rec["half_g0"], color="#2c5f8a", linewidth=2.2,
            label=r"非飽和 $\frac{1}{2}g_0(k)=\kappa L\sqrt{(k+1)\Delta P}$(ポンプが決める)")
    # 飽和込み実測(全パルス平均)＋ p10–p90 帯
    ax.fill_between(rounds, rec["hg_p10"], rec["hg_p90"], color="#c0392b", alpha=0.18,
                    label="飽和込み 実測 p10–p90(全パルス)")
    ax.plot(rounds, rec["hg_mean"], color="#c0392b", linewidth=2.2,
            label=r"飽和込み $\frac{1}{2}g_0(1-\gamma I_{\rm in})$ 平均(実際に効く)")

    # 非結合しきい線・飽和プラトー線・中立線・飽和の効き始め round
    ax.axhline(half_g_th, color="goldenrod", linestyle="--", linewidth=1.5,
               label=fr"非結合系の発振しきい $\frac{{1}}{{2}}\ln(1/\eta)={half_g_th:.2f}$")
    ax.axhline(plateau, color="#c0392b", linestyle=":", linewidth=1.4,
               label=f"飽和プラトー ≈ {plateau:.2f}(実測の頭打ち)")
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0,
               label=r"PSA 中立 $G_I=1$(half_g=0)")
    ax.axvline(onset_round, color="black", linestyle="-.", linewidth=1.2,
               label=f"飽和の効き始め ≈ round {onset_round}")

    ax.set_xlabel("rounds(周回数 $k$)", fontsize=13)
    ax.set_ylabel(r"exp() の中身  half_g $=\frac{1}{2}g_0(1-\gamma I_{\rm in})$", fontsize=13)
    ax.set_title(
        "CIM のゲイン(exp の中身)の rounds 特性 — G22(seed "
        f"{SEED}, {NUM_ROUNDS} rounds)\n"
        "ポンプは非飽和 ½g₀ を単調に上げるが、発振後は飽和が実測ゲインをプラトーに抑える"
        "(I_in 増大で頭打ち)。終盤の帯の広がり = 振幅不均一",
        fontsize=11.5)
    ax.set_xlim(0, NUM_ROUNDS)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out_dir / "gain_vs_rounds.png")
    plt.close(fig)

    print(f"\nsaved: {out_dir / 'gain_vs_rounds.png'}")
    print(f"saved: {out_dir / 'data.npz'}")


if __name__ == "__main__":
    main()
