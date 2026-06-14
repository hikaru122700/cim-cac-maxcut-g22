"""実験で採用した4つのポンプ関数を、実際の生成関数からグラフ化して比較する。

run_pumpbench_real_cim_g22.py と同一の make_P_sched / 端点で
 線形電力ランプ / べき乗早上げ p=0.5 / シグモイド / 線形利得ランプ
を作り、P/P_th と g0/g0_th の2軸で描く。しきい超えラウンドも算出・注記する。

実行: .venv/Scripts/python.exe scripts/plotting/plot_pump_schedules.py
"""
import importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "cim_pumpsched", ROOT / "modules" / "2026-06-08_CIM_pumpsched.py")
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)

# 物理パラメータ(実験と同一)
kappa, L, gamma = 130.0, 0.05, 42.09
eta = 10.0 ** (-1.1)
dP = 0.05e-3
K = 1500
P_th = (np.log(1.0 / eta) / (2.0 * kappa * L)) ** 2
g0_th = np.log(1.0 / eta)
rP_lo, rP_hi = 1.0 * dP / P_th, K * dP / P_th
u_lo, u_hi = np.sqrt(rP_lo), np.sqrt(rP_hi)

SCHEDULES = [
    ("線形電力ランプ(現行)", "#7f8c8d", (np.arange(K) + 1) * dP),
    ("べき乗 早上げ p=0.5", "#d35400",
     ps.make_P_sched(K, "P", "power", rP_lo, rP_hi, P_th, kappa, L, g0_th, p=0.5)),
    ("シグモイド", "#2c5f8a",
     ps.make_P_sched(K, "P", "sigmoid", rP_lo, rP_hi, P_th, kappa, L, g0_th)),
    ("線形利得ランプ(最良)", "#16a085",
     ps.make_P_sched(K, "g0", "linear", u_lo, u_hi, P_th, kappa, L, g0_th)),
]


def cross_round(P_sched):
    """P/P_th が 1 を初めて超えるラウンド(0-indexed +1)。"""
    r = P_sched / P_th
    idx = np.argmax(r >= 1.0)
    return int(idx + 1) if r[idx] >= 1.0 else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    rounds = np.arange(K) + 1
    print("しきい超えラウンド(P/P_th=1):")
    for name, _, P in SCHEDULES:
        print(f"  {name:<20} round {cross_round(P)}")

    fig, (axP, axG) = plt.subplots(1, 2, figsize=(13, 5.4))

    for name, color, P in SCHEDULES:
        rP = P / P_th
        rg = 2.0 * kappa * L * np.sqrt(P) / g0_th  # g0/g0_th = sqrt(P/P_th)
        cr = cross_round(P)
        axP.plot(rounds, rP, color=color, lw=2.0, label=f"{name}（しきい超え≈{cr}）")
        axG.plot(rounds, rg, color=color, lw=2.0, label=name)
        if cr:
            axP.plot(cr, 1.0, "o", color=color, ms=6, zorder=5)
            axG.plot(cr, 1.0, "o", color=color, ms=6, zorder=5)

    for ax, ylab, ttl in [(axP, "ポンプ電力 P / P_th", "電力スケジュール P(k)"),
                          (axG, "利得 g0 / g0_th", "利得スケジュール g0(k)=2κL√P")]:
        ax.axhline(1.0, color="red", ls="--", lw=1.3, label="発振しきい値")
        ax.set_xlabel("ラウンド k", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.set_title(ttl, fontsize=12)
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.grid(alpha=0.3)
    axP.legend(fontsize=8, loc="upper left")

    fig.suptitle("採用した4つのポンプ関数の比較 — 同じ端点・同じ1500ラウンド、形だけが違う", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "docs" / "20260614" / "pump_schedules.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
