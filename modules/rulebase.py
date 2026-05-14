"""1パス貪欲法 — 辺を入力順に見て、できるだけ異色になるよう貪欲に2分割。

ルール:
  辺 (a, b) を順に処理する。
    1. 両方未割当 → a=0, b=1 でこの辺をカット
    2. 片方だけ割当済 → もう片方を反対色に置いて、この辺をカット
    3. 両方割当済   → 既に決まっているので何もしない(カットされる/されないは確定)

最適化や山登り、ルックアヘッドは一切使わない素直な 1 パス処理。
"""
from .verify import compute_cut_from_edges


def rulebase_maxcut(n: int, edges: list[tuple[int, int]]) -> tuple[list[int], int, int]:
    """edges を入力順に走査するだけの 1 パス貪欲。

    Returns:
        assignment: 0/1 ラベル(長さ N)
        cut:        カット数
        isolated:   辺を持たず 0 にデフォルト割当した頂点数
    """
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

    isolated = sum(1 for x in assignment if x is None)
    final = [0 if x is None else x for x in assignment]
    cut = compute_cut_from_edges(final, edges)
    return final, cut, isolated
