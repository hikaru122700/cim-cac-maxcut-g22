"""各方式が設計するポンプ電力スケジュール P(k) と、正準更新式②の
ポンプ項 αp_n = α·p(n) = κL√P(n)(= g0(n)/2)の n 依存性を 2 パネルで描く。

対象実験: results/2026-06-14/pumpbench_real_cim_g22/v1_5cond_100trial_real/
  - 開ループ 4 方式(線形電力 / べき乗早上げ p=0.5 / シグモイド / 線形利得)は
    run_pumpbench_real_cim_g22.py と同一の make_P_sched・端点で生成。
  - CAC(閉ループ)は開ループのポンプ電力波形を「設計」しない(レバーは結合ゲイン e_i)
    ため、本図(開ループのポンプ設計図)には曲線として載せない。注記のみ。

正準更新式②: E(n) = F(n)·exp[ αp_n·(1 − β|F(n)|²) ],  αp_n = α·p(n) = κL√P(n) = g0(n)/2
発振しきい値: 2·αp_th = ln(1/η) = g0_th  ⇒  αp_th = ½·ln(1/η)

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_pump_schedule_and_alphap.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cim_pumpsched", ROOT / "modules" / "2026-06-08_CIM_pumpsched.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

# ---- 物理パラメータ(実験と同一: run_pumpbench_real_cim_g22.py) ----
kappa, L, gamma = 130.0, 0.05, 42.09
eta = 10.0 ** (-1.1)
dP = 0.05e-3
K = 1500
P_th = (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2     # 発振しきい電力 [W]
g0_th = np.log(1.0 / eta)                                # 発振しきい利得
alphap_th = 0.5 * g0_th                                  # αp 項のしきい値 = ½ ln(1/η)
rP_lo, rP_hi = 1.0 * dP / P_th, K * dP / P_th
u_lo, u_hi = np.sqrt(rP_lo), np.sqrt(rP_hi)

# 色は run_pumpbench_real_cim_g22.py と統一
SCHEDULES = [
    ("線形利得ランプ(最良開ループ)", "#16a085",
     ps.make_P_sched(K, "g0", "linear", u_lo, u_hi, P_th, kappa, L, g0_th)),
    ("線形電力ランプ(現行)", "#7f8c8d", (np.arange(K) + 1) * dP),
    ("シグモイド", "#2c5f8a",
     ps.make_P_sched(K, "P", "sigmoid", rP_lo, rP_hi, P_th, kappa, L, g0_th)),
    ("べき乗 早上げ p=0.5", "#d35400",
     ps.make_P_sched(K, "P", "power", rP_lo, rP_hi, P_th, kappa, L, g0_th, p=0.5)),
]

OUT_DIR = (ROOT / "results" / "2026-06-14" / "pumpbench_real_cim_g22"
           / "v1_5cond_100trial_real")


def cross_round(P_sched: np.ndarray) -> int | None:
    """P/P_th が初めて 1 を超えるラウンド(1-indexed)。"""
    r = P_sched / P_th
    idx = int(np.argmax(r >= 1.0))
    return idx + 1 if r[idx] >= 1.0 else None


def alphap(P_sched: np.ndarray) -> np.ndarray:
    """ポンプ項 αp_n = κL√P(n) = g0(n)/2。"""
    return kappa * L * np.sqrt(P_sched)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    rounds = np.arange(K) + 1
    print(f"P_th={P_th*1e3:.2f}mW  g0_th={g0_th:.4f}  αp_th=½ln(1/η)={alphap_th:.4f}")
    print("しきい超えラウンド(P/P_th=1):")
    for name, _, P in SCHEDULES:
        print(f"  {name:<22} round {cross_round(P)}")

    fig, (axP, axA) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    for name, color, P in SCHEDULES:
        cr = cross_round(P)
        axP.plot(rounds, P / P_th, color=color, lw=2.0,
                 label=f"{name}(しきい超え≈{cr})")
        axA.plot(rounds, alphap(P), color=color, lw=2.0, label=name)
        if cr:
            axP.plot(cr, 1.0, "o", color=color, ms=6, zorder=5)
            axA.plot(cr, alphap_th, "o", color=color, ms=6, zorder=5)

    # 左: ポンプ電力スケジュール P(k)/P_th
    axP.axhline(1.0, color="red", ls="--", lw=1.3, label="発振しきい値 P_th")
    axP.set_xlabel("ラウンド n", fontsize=12)
    axP.set_ylabel("ポンプ電力 P(n) / P_th", fontsize=12)
    axP.set_title("各方式が設計するポンプ電力スケジュール P(n)", fontsize=12)
    axP.tick_params(direction="in", which="both", top=True, right=True)
    axP.grid(alpha=0.3)
    axP.legend(fontsize=8, loc="upper left")

    # 右: αp_n 項 = κL√P(n)
    axA.axhline(alphap_th, color="red", ls="--", lw=1.3,
                label=f"しきい値 αp_th=½ln(1/η)={alphap_th:.3f}")
    axA.set_xlabel("ラウンド n", fontsize=12)
    axA.set_ylabel(r"ポンプ項 $\alpha p_n=\kappa L\sqrt{P(n)}=g_0(n)/2$", fontsize=12)
    axA.set_title(r"$\alpha p_n$ 項の n 依存性(式②の増幅指数)", fontsize=12)
    axA.tick_params(direction="in", which="both", top=True, right=True)
    axA.grid(alpha=0.3)
    axA.legend(fontsize=8, loc="upper left")
    # 右軸に正規化値 αp_n/αp_th(= g0/g0_th = √(P/P_th))を併記
    axA2 = axA.twinx()
    axA2.set_ylim(axA.get_ylim()[0] / alphap_th, axA.get_ylim()[1] / alphap_th)
    axA2.set_ylabel(r"$\alpha p_n/\alpha p_{th}=g_0/g_{0,th}=\sqrt{P/P_{th}}$",
                    fontsize=11)
    axA2.tick_params(direction="in")

    fig.suptitle("ポンプ電力スケジュールと αp_n 項の n 依存性 — G22 開ループ4方式"
                 "(CAC は閉ループのため別軸・本図には非掲載)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "pump_schedule_alphap_n.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
