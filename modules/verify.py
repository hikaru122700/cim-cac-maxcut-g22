"""
MAX-CUT 検算モジュール

SA内部の差分計算とは独立した方法でカット数を検証する。
チェック項目:
  - 辺リストからの愚直なカット数計算
  - 隣接リストからの愚直なカット数計算(二重カウント検出)
  - グラフの整合性(自己ループ、多重辺、インデックス範囲)
  - 解の妥当性(全要素が0or1、長さがN)
"""


def verify_graph(n: int, k: int, adj: list[list[int]], edges: list[tuple[int, int]]):
    """グラフの整合性を検証"""
    errors = []

    # 辺数チェック
    if len(edges) != k:
        errors.append(f"辺数不一致: edges={len(edges)}, K={k}")

    # 隣接リスト合計 = 辺数*2 (無向グラフ)
    total_adj = sum(len(a) for a in adj)
    if total_adj != 2 * len(edges):
        errors.append(f"隣接リスト合計不一致: adj合計={total_adj}, 辺数*2={2*len(edges)}")

    # 自己ループチェック
    for a, b in edges:
        if a == b:
            errors.append(f"自己ループ検出: ({a}, {b})")

    # インデックス範囲チェック
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n):
            errors.append(f"インデックス範囲外: ({a}, {b}), N={n}")

    # 多重辺チェック
    edge_set = set()
    for a, b in edges:
        key = (min(a, b), max(a, b))
        if key in edge_set:
            errors.append(f"多重辺検出: ({a}, {b})")
        edge_set.add(key)

    if errors:
        for e in errors:
            print(f"[GRAPH ERROR] {e}")
        return False

    print("[GRAPH OK] 自己ループなし, 多重辺なし, インデックス範囲正常, 辺数一致")
    return True


def verify_solution(x: list[int], n: int):
    """解の妥当性を検証"""
    errors = []

    if len(x) != n:
        errors.append(f"解の長さ不一致: len(x)={len(x)}, N={n}")

    invalid = [i for i, xi in enumerate(x) if xi not in (0, 1)]
    if invalid:
        errors.append(f"0/1以外の値: indices={invalid[:10]}...")

    if errors:
        for e in errors:
            print(f"[SOLUTION ERROR] {e}")
        return False

    count_0 = x.count(0)
    count_1 = x.count(1)
    print(f"[SOLUTION OK] len={len(x)}, 集合A={count_0}頂点, 集合B={count_1}頂点")
    return True


def compute_cut_from_edges(x: list[int], edges: list[tuple[int, int]]) -> int:
    """辺リストから愚直にカット数を計算(検算用)"""
    return sum(1 for a, b in edges if x[a] != x[b])


def compute_cut_from_adj(x: list[int], adj: list[list[int]]) -> int:
    """隣接リストから愚直にカット数を計算(二重カウント検出用)
    無向グラフなので各辺は2回数えられる → 2で割る"""
    count = 0
    for v in range(len(adj)):
        for u in adj[v]:
            if x[v] != x[u]:
                count += 1
    assert count % 2 == 0, f"隣接リストカウントが奇数: {count}"
    return count // 2


def verify_cut(
    x: list[int],
    edges: list[tuple[int, int]],
    adj: list[list[int]],
    reported_cut: int,
) -> bool:
    """3つの方法でカット数を比較検証"""
    cut_edges = compute_cut_from_edges(x, edges)
    cut_adj = compute_cut_from_adj(x, adj)

    all_ok = True

    if cut_edges != cut_adj:
        print(f"[CUT MISMATCH] 辺リスト={cut_edges}, 隣接リスト={cut_adj}")
        all_ok = False

    if cut_edges != reported_cut:
        print(f"[CUT MISMATCH] SA報告値={reported_cut}, 辺リスト検算={cut_edges}")
        all_ok = False

    if all_ok:
        print(f"[CUT OK] 検算一致: cut={cut_edges} (SA報告値={reported_cut})")

    return all_ok


def run_all_checks(
    x: list[int],
    n: int,
    k: int,
    adj: list[list[int]],
    edges: list[tuple[int, int]],
    reported_cut: int,
) -> bool:
    """全検証を実行"""
    print("=" * 50)
    print("検算開始")
    print("=" * 50)

    ok = True
    ok &= verify_graph(n, k, adj, edges)
    ok &= verify_solution(x, n)
    ok &= verify_cut(x, edges, adj, reported_cut)

    print("=" * 50)
    if ok:
        print("全検証パス")
    else:
        print("検証失敗あり")
    print("=" * 50)

    return ok
