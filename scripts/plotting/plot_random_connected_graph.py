"""ランダムな連結グラフを生成して可視化するスクリプト。

CLAUDE.md の命名規則に従い
``results/<YYYY-MM-DD>/random_graph/v{N}_n{N}_m{M}/graph.png`` に保存する。
"""

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

EXPERIMENT_KIND = "random_graph"


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def generate_connected_graph(n: int, m: int, seed: int) -> nx.Graph:
    if m < n - 1:
        raise ValueError(f"m={m} は連結に必要な最小辺数 n-1={n - 1} 未満です")
    max_m = n * (n - 1) // 2
    if m > max_m:
        raise ValueError(f"m={m} は単純グラフの最大辺数 {max_m} を超えています")

    rng = random.Random(seed)
    nodes = list(range(n))
    rng.shuffle(nodes)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    # まずランダム全域木で連結性を確保
    for i in range(1, n):
        u = nodes[i]
        v = nodes[rng.randint(0, i - 1)]
        g.add_edge(u, v)

    # 残り辺数をランダムに追加
    while g.number_of_edges() < m:
        u, v = rng.sample(range(n), 2)
        if not g.has_edge(u, v):
            g.add_edge(u, v)

    assert nx.is_connected(g)
    assert g.number_of_edges() == m
    return g


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="頂点数")
    parser.add_argument("--m", type=int, default=20, help="辺数")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    plt.rcParams["font.family"] = "Yu Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    g = generate_connected_graph(args.n, args.m, args.seed)

    kind_root = get_kind_root()
    v = next_version(kind_root)
    parts = [f"n{args.n}", f"m{args.m}"]
    if args.tag:
        parts.append(args.tag)
    out_dir = kind_root / f"v{v}_{'_'.join(parts)}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pos = nx.spring_layout(g, seed=args.seed)
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#666", width=1.4)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color="#4c78a8", node_size=620, edgecolors="black", linewidths=1.0
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_color="white", font_size=11)

    ax.set_title(f"ランダム連結グラフ (頂点 {args.n}, 辺 {args.m}, seed={args.seed})")
    ax.set_axis_off()
    fig.tight_layout()

    png_path = out_dir / "graph.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    edge_list_path = out_dir / "edges.txt"
    with edge_list_path.open("w", encoding="utf-8") as f:
        f.write(f"# n={args.n} m={args.m} seed={args.seed}\n")
        for u, v_ in sorted(g.edges()):
            f.write(f"{u} {v_}\n")

    print(f"saved: {png_path}")
    print(f"saved: {edge_list_path}")


if __name__ == "__main__":
    main()
