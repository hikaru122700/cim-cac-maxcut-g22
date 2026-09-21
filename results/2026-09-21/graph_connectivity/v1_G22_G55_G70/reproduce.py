"""Read-only graph audit; run from the repository root. Writes a new JSON only."""
from collections import Counter, deque
from hashlib import sha256
from pathlib import Path
import json

output = Path(__file__).with_name("summary.json")
records = []
for name in ("G22", "G55", "G70"):
    path = Path("input") / f"{name}.txt"
    payload = path.read_bytes()
    lines = payload.decode("utf-8-sig").splitlines()
    n, declared_m = map(int, lines[0].split())
    adjacency = [[] for _ in range(n)]
    edges = set()
    weights = Counter()
    for line in lines[1:]:
        if not line.strip():
            continue
        u, v, w = map(int, line.split())
        u -= 1
        v -= 1
        assert 0 <= u < n and 0 <= v < n and u != v
        edge = (min(u, v), max(u, v))
        assert edge not in edges
        edges.add(edge)
        adjacency[u].append(v)
        adjacency[v].append(u)
        weights[w] += 1
    assert len(edges) == declared_m
    seen = set()
    sizes = []
    for start in range(n):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        size = 0
        while queue:
            u = queue.popleft()
            size += 1
            for v in adjacency[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        sizes.append(size)
    degree = [len(neighbors) for neighbors in adjacency]
    records.append({
        "graph": name, "input": path.as_posix(), "sha256": sha256(payload).hexdigest(),
        "vertices": n, "edges": len(edges), "mean_degree": sum(degree) / n,
        "connected_components": len(sizes), "largest_component": max(sizes),
        "isolated_vertices": degree.count(0), "degree_one_vertices": degree.count(1),
        "max_degree": max(degree), "edge_weight_counts": dict(weights),
        "component_size_histogram": dict(sorted(Counter(sizes).items())),
    })
with output.open("x", encoding="utf-8") as stream:
    json.dump({"method": "BFS on supplied undirected input graphs", "graphs": records}, stream, indent=2)
for item in records:
    print(item["graph"], item["connected_components"], item["isolated_vertices"], item["largest_component"])
