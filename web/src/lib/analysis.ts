/**
 * グラフ + 割当からの集計 (pure functions, テスト容易)。
 */

import type { Assignment, CutStats, Graph } from "../types";

export function computeCutStats(graph: Graph, assignment: Assignment): CutStats {
  const { n, edges } = graph;
  const { side } = assignment;

  if (side.length !== n) {
    throw new Error(
      `assignment length ${side.length} != graph N ${n}`,
    );
  }

  // per-vertex 次数と cut 次数を計算
  const degrees = new Array<number>(n).fill(0);
  const cutDegrees = new Array<number>(n).fill(0);

  let cut = 0;
  let totalWeight = 0;
  for (let e = 0; e < edges.length; e++) {
    const [u, v, w] = edges[e];
    totalWeight += w;
    degrees[u] += 1;
    degrees[v] += 1;
    if (side[u] !== side[v]) {
      cut += 1;
      cutDegrees[u] += 1;
      cutDegrees[v] += 1;
    }
  }

  let numPositive = 0;
  for (let i = 0; i < n; i++) if (side[i]) numPositive += 1;

  // 局所最適性: 頂点 i をフリップすると cut は
  //   (今 cut でない辺 = degrees[i] - cutDegrees[i])
  //   を cut に変え、
  //   (今 cut の辺 = cutDegrees[i])
  //   を cut でなくする。
  // つまり delta = (degrees[i] - cutDegrees[i]) - cutDegrees[i]
  //              = degrees[i] - 2 * cutDegrees[i]
  // delta > 0 ならフリップで改善 (= 局所最適でない頂点)。
  let localImprovable = 0;
  for (let i = 0; i < n; i++) {
    if (degrees[i] - 2 * cutDegrees[i] > 0) localImprovable += 1;
  }

  return {
    cut,
    totalWeight,
    cutRatio: totalWeight > 0 ? cut / totalWeight : 0,
    numPositive,
    numNegative: n - numPositive,
    localImprovable,
    cutDegrees,
    degrees,
  };
}

/**
 * cut degree のヒストグラムを返す (bin index → count)。
 * bin は 0..maxDeg を num_bins 等分 (既定 = maxDeg + 1 の離散 bin)。
 */
export function cutDegreeHistogram(
  cutDegrees: ReadonlyArray<number>,
): { counts: number[]; max: number } {
  if (cutDegrees.length === 0) return { counts: [], max: 0 };
  let max = 0;
  for (const d of cutDegrees) if (d > max) max = d;
  const counts = new Array<number>(max + 1).fill(0);
  for (const d of cutDegrees) counts[d] += 1;
  return { counts, max };
}
