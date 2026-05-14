"""辺を入力順に見て、できるだけ異色になるよう貪欲に頂点を 2 色塗り分ける。

ルール:
  辺 (a, b) を順に処理する。
    1. 両方未割当 → a=0, b=1 でこの辺をカット
    2. 片方だけ割当済 → もう片方を反対色に置いて、この辺をカット
    3. 両方割当済   → 既に決まっているので何もしない(カットされる/されないは確定)

最適化や山登り、ルックアヘッドは一切使わない素直な 1 パス処理。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
from pathlib import Path

from modules.CIM import load_graph
from modules.verify import compute_cut_from_edges


def rulebase_maxcut(n: int, edges: list[tuple[int, int]]) -> tuple[list[int], int]:
    """edges を入力順に走査するだけの 1 パス貪欲。"""
    assignment: list[int | None] = [None] * n
    for a, b in edges:
        sa, sb = assignment[a], assignment[b]
        if sa is None and sb is None:
            assignment[a] = 0
            assignment[b] = 1
        elif sa is None:
            assignment[a] = 1 - sb
        elif sb is None:
            assignment[b] = 1 - sa
        # else: 両方割当済 → 触らない

    # 孤立頂点(辺を持たない頂点)は仮に 0
    isolated = sum(1 for x in assignment if x is None)
    final = [0 if x is None else x for x in assignment]

    cut = compute_cut_from_edges(final, edges)
    return final, cut, isolated


def main():
    targets = [
        "input/G22.txt",
        "input/G1.txt",
        "input/G14.txt",
        "input/G15.txt",
        "input/G23.txt",
        "input/G32.txt",
        "input/G39.txt",
        "input/G55.txt",
        "input/G70.txt",
        "input/G77.txt",
        "input/G81.txt",
    ]
    # 既知最良値(便宜的に手元で確認した値)
    known_best = {
        "G1": 11624, "G14": 3064, "G15": 3050, "G22": 13359, "G23": 13344,
        "G32": 1410, "G39": 2408, "G55": 10299, "G70": 9591,
        "G77": 9926, "G81": 14030,
    }

    print(f"{'graph':<6} {'N':>5} {'K':>7} {'cut':>7} {'ratio':>8} {'isolated':>9} {'time[ms]':>9}")
    print("-" * 60)

    for path in targets:
        name = Path(path).stem
        n, k, _, edges = load_graph(path)
        t0 = time.perf_counter()
        assign, cut, isolated = rulebase_maxcut(n, edges)
        ms = (time.perf_counter() - t0) * 1000
        kb = known_best.get(name)
        ratio = cut / kb if kb else 0.0
        print(f"{name:<6} {n:>5} {k:>7} {cut:>7} {ratio:>7.4f}  {isolated:>9} {ms:>9.2f}")


if __name__ == "__main__":
    main()
