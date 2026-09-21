"""Audit graph structure without changing inputs; run from the repository root."""
from collections import Counter
from datetime import date
from hashlib import sha256
from pathlib import Path
import json
import networkx as nx


def main():
    records = []
    for name in ("G22", "G55", "G70"):
        path = Path("input") / f"{name}.txt"
        payload = path.read_bytes()
        lines = payload.decode("utf-8-sig").splitlines()
        n, m = map(int, lines[0].split())
        graph = nx.Graph()
        graph.add_nodes_from(range(1, n + 1))
        for line in lines[1:]:
            if not line.strip():
                continue
            u, v, w = map(int, line.split())
            assert 1 <= u <= n and 1 <= v <= n and u != v
            assert not graph.has_edge(u, v)
            graph.add_edge(u, v, weight=w)
        assert graph.number_of_edges() == m
        components = list(nx.connected_components(graph))
        cores = nx.core_number(graph)
        core2 = graph.subgraph([u for u, k in cores.items() if k >= 2])
        blocks = list(nx.biconnected_components(graph))
        degrees = [d for _, d in graph.degree()]
        records.append({
            "graph": name, "input": path.as_posix(),
            "sha256": sha256(payload).hexdigest(), "vertices": n, "edges": m,
            "mean_degree": 2 * m / n,
            "connected_components": len(components),
            "largest_component": max(map(len, components)),
            "isolated_vertices": degrees.count(0),
            "degree_one_vertices": degrees.count(1),
            "bridges": sum(1 for _ in nx.bridges(graph)),
            "articulation_points": sum(1 for _ in nx.articulation_points(graph)),
            "biconnected_blocks_including_bridges": len(blocks),
            "largest_biconnected_block_vertices": max(map(len, blocks), default=0),
            "core_number_histogram": dict(sorted(Counter(cores.values()).items())),
            "two_core_vertices": core2.number_of_nodes(),
            "two_core_edges": core2.number_of_edges(),
            "cycle_rank": m - n + len(components),
            "triangles": sum(nx.triangles(graph).values()) // 3,
            "average_clustering_including_degree_zero_one": nx.average_clustering(graph),
            "edge_weight_counts": dict(Counter(d["weight"] for _, _, d in graph.edges(data=True))),
        })
    root = Path("results") / date.today().isoformat() / "graph_connectivity"
    root.mkdir(parents=True, exist_ok=True)
    versions = [int(p.name.split("_", 1)[0][1:]) for p in root.glob("v*_*")
                if p.name.split("_", 1)[0][1:].isdigit()]
    output = root / f"v{max(versions, default=0) + 1}_structure_G22_G55_G70"
    output.mkdir()
    result = {"networkx_version": nx.__version__, "graphs": records}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
