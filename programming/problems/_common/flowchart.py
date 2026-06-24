"""処理フロー図の描画ユーティリティ(日本語対応・matplotlib)。

各問題の解説で使う「処理の流れ図」を、同じ見た目で簡単に作るためのヘルパー。
縦に積んだ角丸ボックスを下向き矢印でつなぎ、必要なら右側に補足注記を添える。
分岐(条件)はひし形ノードで表す。

使い方の例:
    from flowchart import draw_flowchart, Step, Branch

    draw_flowchart(
        [
            Step("PRBS を 1 周期生成", note="LFSR を 32767 回まわす"),
            Step("双極性に変換 (0->-1, 1->+1)"),
            Branch("矩形パルスにするか?", yes="np.repeat で整形", no="そのまま FFT"),
            Step("FFT してパワー |X|^2 を計算"),
            Step("3 段の図を spectrum.png に保存"),
        ],
        out_path="flow.png",
        title="PRBS15 強度スペクトルの処理フロー",
    )

Windows 上の Yu Gothic / Meiryo / MS Gothic を自動で拾う。見つからなければ
既定フォントにフォールバックする(その場合、日本語が豆腐になることがある)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as patches
import matplotlib.pyplot as plt


# --- 日本語フォントの設定 ---------------------------------------------------
_JP_CANDIDATES = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "IPAGothic"]


def _set_japanese_font() -> str:
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _JP_CANDIDATES:
        if name in available:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return plt.rcParams.get("font.family", ["sans-serif"])[0]


# --- ノード定義 -------------------------------------------------------------
@dataclass(frozen=True)
class Step:
    """通常の処理ボックス(1 ステップ)。"""

    text: str
    note: str = ""


@dataclass(frozen=True)
class Branch:
    """条件分岐(ひし形)。yes / no それぞれの行き先を短く書く。"""

    text: str
    yes: str = ""
    no: str = ""


# --- 色 ---------------------------------------------------------------------
_BOX_FACE = "#eef3f8"
_BOX_EDGE = "#2c5f8a"
_BRANCH_FACE = "#fdf3e3"
_BRANCH_EDGE = "#c8862b"
_NOTE_COLOR = "#555555"
_ARROW = dict(arrowstyle="-|>", color="#2c5f8a", linewidth=1.6, mutation_scale=18)


def draw_flowchart(
    nodes: list,
    out_path: str,
    title: str = "",
    box_width: float = 4.2,
    box_height: float = 0.9,
    v_gap: float = 0.7,
) -> str:
    """ノード列を縦型フロー図として描き、PNG に保存する。

    Args:
        nodes: Step / Branch の並び(上から下へ実行される順)。
        out_path: 出力 PNG パス。
        title: 図のタイトル(空なら付けない)。
        box_width, box_height: 各ボックスの大きさ(データ座標)。
        v_gap: ボックス間の縦の隙間(矢印が通る)。

    Returns:
        保存した PNG の絶対パス相当(渡した out_path をそのまま返す)。
    """
    _set_japanese_font()

    n = len(nodes)
    cx = 0.0  # 中心 x
    total_h = n * box_height + (n - 1) * v_gap
    top = total_h / 2.0

    fig_h = max(3.0, total_h + (1.2 if title else 0.6))
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-top - 0.6, top + (1.0 if title else 0.4))
    ax.axis("off")

    if title:
        ax.text(cx, top + 0.55, title, ha="center", va="center",
                fontsize=14, fontweight="bold", color="#1a1a1a")

    centers: list[float] = []
    y = top - box_height / 2.0
    for _ in range(n):
        centers.append(y)
        y -= box_height + v_gap

    for i, node in enumerate(nodes):
        ycen = centers[i]
        if isinstance(node, Branch):
            _draw_diamond(ax, cx, ycen, box_width, box_height, node.text)
            if node.yes:
                ax.text(cx + box_width / 2.0 + 0.15, ycen + 0.12, f"はい: {node.yes}",
                        ha="left", va="center", fontsize=9, color=_BRANCH_EDGE)
            if node.no:
                ax.text(cx + box_width / 2.0 + 0.15, ycen - 0.20, f"いいえ: {node.no}",
                        ha="left", va="center", fontsize=9, color=_NOTE_COLOR)
        else:
            _draw_box(ax, cx, ycen, box_width, box_height, node.text)
            if node.note:
                ax.annotate(
                    node.note,
                    xy=(cx + box_width / 2.0, ycen),
                    xytext=(cx + box_width / 2.0 + 0.25, ycen),
                    ha="left", va="center", fontsize=9, color=_NOTE_COLOR,
                    arrowprops=dict(arrowstyle="-", color="#bbbbbb", linewidth=0.8),
                )

        # 下のノードへ矢印
        if i < n - 1:
            y_from = ycen - box_height / 2.0
            y_to = centers[i + 1] + box_height / 2.0
            ax.annotate("", xy=(cx, y_to), xytext=(cx, y_from), arrowprops=_ARROW)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _draw_box(ax, cx, cy, w, h, text) -> None:
    box = patches.FancyBboxPatch(
        (cx - w / 2.0, cy - h / 2.0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=_BOX_EDGE, facecolor=_BOX_FACE,
    )
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10.5, color="#1a1a1a", wrap=True)


def _draw_diamond(ax, cx, cy, w, h, text) -> None:
    hw, hh = w / 2.0 * 0.85, h / 2.0 * 1.25
    diamond = patches.Polygon(
        [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
        closed=True, linewidth=1.6, edgecolor=_BRANCH_EDGE, facecolor=_BRANCH_FACE,
    )
    ax.add_patch(diamond)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10, color="#1a1a1a")


if __name__ == "__main__":
    # 動作確認(このファイル単体で実行するとサンプル図を出す)
    draw_flowchart(
        [
            Step("PRBS を 1 周期生成", note="LFSR を 32767 回まわす"),
            Step("双極性に変換 (0->-1, 1->+1)"),
            Branch("矩形パルスにするか?", yes="np.repeat で整形", no="そのまま FFT"),
            Step("FFT してパワー |X|^2 を計算"),
            Step("3 段の図を保存"),
        ],
        out_path="_flow_sample.png",
        title="サンプル: 処理フロー",
    )
    print("saved _flow_sample.png")
