"""CIM のゲイン定義（g0 → G_I → √G_I）と「½ / 2」の関係の模式図。

ゲイン解説 PDF（docs/まとめ）用の 1 枚図。ポンプ→利得係数→強度ゲイン/振幅ゲイン→
振幅更新、の連鎖と、係数 2 と ½ がどこに現れるかを示す。

実行(プロジェクトルートから):
  .venv/Scripts/python.exe scripts/plotting/plot_gain_definition_schematic.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _next_version_dir(desc: str) -> Path:
    root = ROOT / "results" / date.today().isoformat() / "cim_gain_definition"
    root.mkdir(parents=True, exist_ok=True)
    v = 0
    for q in root.iterdir():
        if q.is_dir() and q.name.startswith("v") and q.name.split("_", 1)[0][1:].isdigit():
            v = max(v, int(q.name.split("_", 1)[0][1:]))
    out = root / f"v{v + 1}_{desc}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(14, 6.6), dpi=140)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    def box(x, y, text, fc, w=2.0, h=1.05, fs=12, ec="#333"):
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=fc, edgecolor=ec, lw=1.4))

    def arrow(x1, y1, x2, y2, color="#333"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0))

    ymid = 4.5
    box(1.3, ymid, "ポンプ電力\n$P_p(k)=k\\,\\Delta P$", "#ecf0f1")
    box(4.0, ymid, "非飽和 利得係数\n$g_0 = \\mathbf{2}\\,\\kappa L\\sqrt{P_p}$\n（★ここに「2」）", "#fdebd0")
    box(6.7, ymid, "飽和込み\n$g = g_0(1-\\gamma I_{\\mathrm{in}})$", "#d6eaf8")

    arrow(2.3, ymid, 3.0, ymid)
    arrow(5.0, ymid, 5.7, ymid)

    # 分岐: 強度ゲイン(上) と 振幅ゲイン(下)
    box(9.7, 5.7, "強度（パワー）ゲイン\n$G_I = e^{\\,g}$\n（全量・論文 Eq.14）", "#fadbd8", fs=11.5)
    box(9.7, 3.3, "振幅ゲイン\n$\\sqrt{G_I}=e^{\\,g/2}$\n（半量・★「½」）", "#d5f5e3", fs=11.5)
    arrow(7.7, ymid + 0.3, 8.5, 5.5)
    arrow(7.7, ymid - 0.3, 8.5, 3.5)

    box(9.7, 1.2, "振幅更新（c に効く）\n$c(k{+}1)=\\sqrt{G_I}\\,(\\sqrt{\\eta}\\,c+Jc)+N_I$\n論文 Eq.3", "#eaeded", fs=11)
    arrow(9.7, 2.75, 9.7, 1.75)

    # 下部の要点ボックス
    note = ("振幅 = √(強度)  →  パワーが $G_I$ 倍なら振幅は $\\sqrt{G_I}=e^{g/2}$ 倍。\n"
            "係数 2 を「強度側」に置けば振幅指数は $\\frac{1}{2}g_0$（＝「½」, $g_0=2\\kappa L\\sqrt{P}$ 基準）／"
            "「振幅側」に置けば $G_I=e^{2r}$（＝「2」, $r=\\kappa L\\sqrt{P}$ 基準）。\n"
            "実際に振幅へ効く指数はどちらでも $\\kappa L\\sqrt{P_p}\\,(1-\\gamma I_{\\mathrm{in}})$（2 と ½ は相殺）。同じ物理。")
    ax.text(6.0, 0.05, note, ha="center", va="bottom", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#fcf3cf", edgecolor="#b7950b", lw=1.4))

    ax.set_title("CIM のゲイン連鎖と「½ / 2」の出どころ —  $g_0\\!\\to\\! G_I\\!\\to\\!\\sqrt{G_I}$",
                 fontsize=15, pad=12)

    out_dir = _next_version_dir("summary")
    fig.tight_layout()
    fig.savefig(out_dir / "gain_chain_schematic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_dir / 'gain_chain_schematic.png'}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
