"""反強磁性モデルでCIM種評価の仮説を説明する。配置は手作りで実験結果ではない。"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_KIND = "cim_seed_research_plan"
plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12


def next_version(root: Path) -> int:
    """実験種別ごとに未使用の版を採番する。"""
    versions = [int(p.name.split("_", 1)[0][1:]) for p in root.iterdir()
                if p.is_dir() and re.match(r"^v\d+_", p.name)]
    return max(versions, default=0) + 1


def cut(s: np.ndarray, edges: list[tuple[int, int]]) -> int:
    """正の単位重みの辺を直接数える。"""
    return sum(int(s[i] != s[j]) for i, j in edges)


def metrics(s: np.ndarray, edges: list[tuple[int, int]]) -> dict:
    """全頂点の単独反転を直接評価し、局所利得の式とも照合する。"""
    before = cut(s, edges)
    gains = []
    for i in range(len(s)):
        flipped = s.copy()
        flipped[i] *= -1
        gain = cut(flipped, edges) - before
        formula = sum(int(s[u] * s[v]) for u, v in edges if i in (u, v))
        assert gain == formula
        gains.append(gain)
    return {"cut": before, "edges": len(edges), "unsatisfied": len(edges) - before,
            "single_flip_gains": gains, "spins": s.tolist()}


def draw(ax, coords, edges, spins, title, note, highlight=()):
    """同じ座標と色で配置と不満足辺を示す。"""
    for i, j in edges:
        bad = spins[i] == spins[j]
        ax.plot([coords[i][0], coords[j][0]], [coords[i][1], coords[j][1]],
                color="#D84738" if bad else "#C5CCD3", lw=4.5 if bad else 2,
                zorder=1)
    xy = np.asarray(coords)
    ax.scatter(xy[:, 0], xy[:, 1], s=640,
               c=["#176B9A" if s > 0 else "#E4A62C" for s in spins],
               edgecolors="white", linewidths=1.6, zorder=2)
    for i, (x, y) in enumerate(coords):
        ax.text(x, y, "+" if spins[i] > 0 else "−", ha="center", va="center",
                color="white" if spins[i] > 0 else "#382B15", fontsize=20,
                weight="bold", zorder=3)
    if highlight:
        ax.scatter(xy[list(highlight), 0], xy[list(highlight), 1], s=960,
                   facecolors="none", edgecolors="#273746", linewidths=1.5,
                   linestyle="--", zorder=4)
    ax.set_title(title, fontsize=15, weight="bold", pad=17)
    ax.text(.5, -.12, note, transform=ax.transAxes, ha="center", va="top",
            fontsize=12, linespacing=1.65)
    ax.set_aspect("equal")
    ax.set_xlim(-.55, 3.55)
    ax.set_ylim(-.5, 3.5)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.axis("off")


def finish(fig, path, title, subtitle):
    """図の位置を固定して保存する。"""
    fig.suptitle(title, fontsize=21, weight="bold", y=.97)
    fig.text(.5, .885, subtitle, ha="center", fontsize=12)
    fig.legend(handles=[
        Line2D([0], [0], color="#C5CCD3", lw=3, label="満足辺：隣同士が逆符号"),
        Line2D([0], [0], color="#D84738", lw=4, label="不満足辺：隣同士が同符号"),
    ], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, .01))
    fig.subplots_adjust(top=.76, bottom=.25, left=.035, right=.965, wspace=.24)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    """格子と三角形の検算済み模式図を、新しい保存先へ作成する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["grid4x4"], default="grid4x4")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    if args.tag and not re.fullmatch(r"[A-Za-z0-9_]+", args.tag):
        parser.error("tagは半角英数字とアンダースコアのみ")
    kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    kind_root.mkdir(parents=True, exist_ok=True)
    description = args.example + ("_" + args.tag if args.tag else "")
    out = kind_root / f"v{next_version(kind_root)}_{description}"
    out.mkdir(exist_ok=False)

    coords = [(c, 3-r) for r in range(4) for c in range(4)]
    edges = [(4*r+c, 4*r+c+1) for r in range(4) for c in range(3)]
    edges += [(4*r+c, 4*(r+1)+c) for r in range(3) for c in range(4)]
    ground = np.array([(-1)**(r+c) for r in range(4) for c in range(4)])
    point = ground.copy()
    point[5] *= -1
    wall = ground.copy()
    wall[[4*r+c for r in range(4) for c in range(2)]] *= -1
    other = ground.copy()
    other[10] *= -1
    tri_edges = [(0, 1), (1, 2), (2, 0)]
    tri = np.array([1, -1, 1])
    data = {"provenance": "手作り配置。CIM/SA/GAを実行した結果ではない。",
            "boundary": "格子は開放境界。全辺の重みは+1、外場なし。",
            "grid_edges": edges, "triangle_edges": tri_edges,
            "point": metrics(point, edges), "wall": metrics(wall, edges),
            "other_parent": metrics(other, edges), "child": metrics(ground, edges),
            "triangle": metrics(tri, tri_edges)}
    assert data["point"]["cut"] == data["wall"]["cut"] == 20
    assert max(data["point"]["single_flip_gains"]) == 4
    assert max(data["wall"]["single_flip_gains"]) == -1
    assert cut(ground, edges) == 24 and cut(tri, tri_edges) == 2
    assert not any(i in (5, 10) and j in (5, 10) for i, j in edges)
    # 親の不一致点5と10は直接結合しないため、良い側を独立に選べる。
    child = point.copy()
    child[5] = other[5]
    assert np.array_equal(child, ground)
    (out / "examples.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    draw(axes[0], coords, edges, point, "A：1スピンの反転で直せる",
         "カット値 20 / 24・不満足辺 4本\n丸囲みを反転 → 24 / 24", highlight=(5,))
    draw(axes[1], coords, edges, wall, "B：逆位相の領域が隣り合う",
         "カット値 20 / 24・不満足辺 4本\nどの1スピン反転も、まず悪化する")
    draw(axes[2], [(0, .35), (1.5, 2.95), (3, .35)], tri_edges, tri,
         "C：三角形では1本残るのが最適",
         "カット値 2 / 3・不満足辺 1本\n全てを同時に満たせない（フラストレーション）")
    finish(fig, out / "defect_examples.png", "同じカット値でも、改善に必要な操作は違う",
           "反強磁性・単位重み・外場なし。AとBは同じ4×4格子（開放境界）。手作りの説明用配置。")

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    draw(axes[0], coords, edges, point, "親A：左上寄りに欠陥",
         "カット値 20 / 24\n親Bとは2頂点だけが異なる", highlight=(5,))
    draw(axes[1], coords, edges, other, "親B：右下寄りに欠陥",
         "カット値 20 / 24\nそれぞれ別の場所が良い", highlight=(10,))
    draw(axes[2], coords, edges, child, "良い部分を選べた子",
         "カット値 24 / 24\n相補性を説明する配置。現行GAの実測ではない。")
    finish(fig, out / "complementary_parents.png", "GAでは「どれだけ違うか」に加えて「良い部分を補い合えるか」",
           "不一致の2頂点は直接結合していないため、親の良い側を独立に選べる。初期局所探索前の模式例。")
    print(out)
    print("検算: A=20, max gain=4; B=20, max gain=-1; triangle=2; child=24")


if __name__ == "__main__":
    main()
