/**
 * ドメイン型定義。すべて不変 (readonly) で扱う。
 */

export interface Graph {
  readonly n: number;
  readonly k: number;
  /** 0-indexed 辺リスト (a, b, weight) */
  readonly edges: ReadonlyArray<readonly [number, number, number]>;
}

export interface Assignment {
  /** 長さ N の bool 配列。true = partition 1 (+1 side), false = partition 0 (-1 side) */
  readonly side: ReadonlyArray<boolean>;
}

export interface CutStats {
  readonly cut: number;
  readonly totalWeight: number;
  readonly cutRatio: number;
  readonly numPositive: number;
  readonly numNegative: number;
  /** 1 回のフリップで cut が増える頂点の数 (局所最適からの距離の目安) */
  readonly localImprovable: number;
  /** 各頂点について (自分の辺のうち相手側へ向かう数, 自分の辺総数) */
  readonly cutDegrees: ReadonlyArray<number>;
  readonly degrees: ReadonlyArray<number>;
}
