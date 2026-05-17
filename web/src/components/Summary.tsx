import type { CutStats, Graph } from "../types";

interface Props {
  graph: Graph;
  stats: CutStats;
  bks?: number;
}

/** サマリパネル: N/K/cut/balance/局所最適距離など。 */
export function Summary({ graph, stats, bks }: Props) {
  const items: { label: string; value: string; highlight?: boolean }[] = [
    { label: "N (vertices)", value: graph.n.toLocaleString() },
    { label: "K (edges)", value: graph.k.toLocaleString() },
    {
      label: "Cut value",
      value: stats.cut.toLocaleString(),
      highlight: true,
    },
    {
      label: "Cut ratio",
      value: `${(stats.cutRatio * 100).toFixed(2)} %`,
    },
    {
      label: "+1 / −1",
      value: `${stats.numPositive.toLocaleString()} / ${stats.numNegative.toLocaleString()}`,
    },
    {
      label: "Balance |Δ|",
      value: Math.abs(stats.numPositive - stats.numNegative).toLocaleString(),
    },
    {
      label: "Local improvable",
      value: `${stats.localImprovable.toLocaleString()} / ${graph.n.toLocaleString()}`,
    },
  ];
  if (bks !== undefined) {
    items.push({
      label: "Gap to BKS",
      value: `${bks - stats.cut}  (${((stats.cut / bks) * 100).toFixed(2)} %)`,
    });
  }

  return (
    <div className="summary">
      {items.map((it) => (
        <div key={it.label} className={`stat${it.highlight ? " hl" : ""}`}>
          <div className="stat-label">{it.label}</div>
          <div className="stat-value">{it.value}</div>
        </div>
      ))}
    </div>
  );
}
