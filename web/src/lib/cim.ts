/**
 * Coherent Ising Machine (CIM) の測定フィードバック型シミュレータ。
 *
 * 1 ラウンド = 光パルスがファイバーループを 1 周する過程:
 *   1. パルスを投げる              … 前ラウンドの振幅 c
 *   2. 減衰 (ループ損失)          … √η · c
 *   3. 結合 (MFB, J_ij を掛ける)  … + Σ_j J_ij c_j
 *   4. 利得 (縮退パラメトリック増幅 + 飽和) … × exp[½g0(1 − γ|F|²)]
 *   5. ノイズを足す                … + ξ
 *   6. 次のパルス c' を生成して再び投げる
 *
 * 物理定数は本リポジトリの modules/CIM.py に合わせる:
 *   g0 = 2κ√P·L,  発振しきい値 P_th = (ln(1/η)/(2κL))²。
 * ポンプ電力 P をラウンドごとに上げる (ランプ=焼きなまし)。
 * ノイズだけは可視化用に振幅をスライダーで調整できるようにしてある。
 */

// --- リポジトリ準拠の物理定数 ---
const KAPPA = 130.0; // 結合係数
const L = 0.05; // PSA 媒質長 [m]
const GAMMA = 42.09; // 飽和係数
const ETA = Math.pow(10, -1.1); // ループ損失 (透過率)
const SQRT_ETA = Math.sqrt(ETA);
const TWO_KAPPA_L = 2.0 * KAPPA * L;

/** 発振しきい値ポンプ P_th [W] = (ln(1/η)/(2κL))²。 */
export const P_TH_W = Math.pow(Math.log(1.0 / ETA) / TWO_KAPPA_L, 2);
export const P_TH_MW = P_TH_W * 1e3;

export type GraphKind = "ring" | "random" | "complete" | "grid";

export interface CimConfig {
  readonly n: number;
  readonly rounds: number;
  /** ポンプ・ランプ率 [mW/round]。0 にすると固定ポンプ。 */
  readonly dPumpMW: number;
  /** 固定ポンプ時 / ランプ初期オフセットのベース倍率 (P_th 単位)。 */
  readonly pumpMult: number;
  /** ノイズ振幅 (可視化用)。 */
  readonly noise: number;
  /** 反強磁性結合の強さ |J| スケール。 */
  readonly coupling: number;
  readonly graph: GraphKind;
  /** random グラフの辺確率。 */
  readonly edgeProb: number;
  readonly seed: number;
}

export const DEFAULT_CONFIG: CimConfig = {
  n: 24,
  rounds: 1500,
  dPumpMW: 0.05,
  pumpMult: 1.0,
  noise: 0.0015,
  coupling: 0.03,
  graph: "random",
  edgeProb: 0.5,
  seed: 1,
};

/** 0-indexed 重みなし辺リスト。 */
export type Edge = readonly [number, number];

// --- seedable RNG (mulberry32) ---
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller で標準正規乱数を返すジェネレータ。 */
function gaussianFactory(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    const mag = Math.sqrt(-2.0 * Math.log(u));
    spare = mag * Math.sin(2.0 * Math.PI * v);
    return mag * Math.cos(2.0 * Math.PI * v);
  };
}

/** グラフ種別から辺リストを生成する。 */
export function buildEdges(
  kind: GraphKind,
  n: number,
  edgeProb: number,
  seed: number,
): Edge[] {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const edges: Edge[] = [];
  if (kind === "ring") {
    for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
  } else if (kind === "complete") {
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) edges.push([i, j]);
  } else if (kind === "grid") {
    const side = Math.max(2, Math.round(Math.sqrt(n)));
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const v = r * side + c;
        if (v >= n) continue;
        if (c + 1 < side && v + 1 < n) edges.push([v, v + 1]);
        if (r + 1 < side && v + side < n) edges.push([v, v + side]);
      }
    }
  } else {
    // random (Erdős–Rényi)、連結性確保のためまずリングを張る
    for (let i = 0; i < n; i++) edges.push([i, (i + 1) % n]);
    for (let i = 0; i < n; i++)
      for (let j = i + 2; j < n; j++)
        if (rand() < edgeProb) edges.push([i, j]);
  }
  return edges;
}

/** 隣接リスト (CSR 風) を作る。 */
function buildAdjacency(
  n: number,
  edges: readonly Edge[],
): { indptr: Int32Array; indices: Int32Array } {
  const deg = new Int32Array(n);
  for (const [a, b] of edges) {
    deg[a]++;
    deg[b]++;
  }
  const indptr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) indptr[i + 1] = indptr[i] + deg[i];
  const indices = new Int32Array(indptr[n]);
  const cursor = indptr.slice(0, n);
  for (const [a, b] of edges) {
    indices[cursor[a]++] = b;
    indices[cursor[b]++] = a;
  }
  return { indptr, indices };
}

export interface CimSnapshot {
  readonly round: number;
  readonly pumpMW: number;
  readonly pumpRatio: number; // P / P_th
  readonly cut: number;
  readonly meanAbsAmp: number;
  /** 現在の振幅 c (コピー)。 */
  readonly amps: Float64Array;
}

/**
 * CIM シミュレータ本体。物理積分のホットループのため内部では
 * TypedArray を破壊的に更新する (React 側へは snapshot() でコピーを渡す)。
 */
export class CimSim {
  readonly n: number;
  readonly edges: readonly Edge[];
  private readonly cfg: CimConfig;
  private readonly indptr: Int32Array;
  private readonly indices: Int32Array;
  private readonly c: Float64Array;
  private readonly jcoef: number; // 反強磁性: 負
  private readonly gauss: () => number;
  private k = 0;

  constructor(cfg: CimConfig, edges?: readonly Edge[]) {
    this.cfg = cfg;
    this.n = cfg.n;
    this.edges = edges ?? buildEdges(cfg.graph, cfg.n, cfg.edgeProb, cfg.seed);
    const adj = buildAdjacency(cfg.n, this.edges);
    this.indptr = adj.indptr;
    this.indices = adj.indices;
    this.c = new Float64Array(cfg.n);
    this.jcoef = -Math.abs(cfg.coupling); // MAX-CUT は反強磁性 (J<0)
    this.gauss = gaussianFactory(mulberry32(cfg.seed));
  }

  get round(): number {
    return this.k;
  }
  get done(): boolean {
    return this.k >= this.cfg.rounds;
  }

  /** round k でのポンプ電力 [W]。 */
  private pumpW(): number {
    const ramp = this.cfg.dPumpMW > 0 ? (this.k + 1) * this.cfg.dPumpMW : P_TH_MW;
    return (this.cfg.pumpMult * ramp) / 1e3;
  }

  /** 1 ラウンド (パルス 1 周) を進める。 */
  step(): void {
    if (this.done) return;
    const { n, c, indptr, indices, jcoef } = this;
    const P = this.pumpW();
    const g0 = TWO_KAPPA_L * Math.sqrt(P);
    const halfG0 = 0.5 * g0;
    const negHalfG0Gamma = -0.5 * g0 * GAMMA;
    const noiseScale = this.cfg.noise;

    const next = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      // 3. 結合 (J_ij c_j の総和 = MFB)
      let jc = 0.0;
      for (let p = indptr[i]; p < indptr[i + 1]; p++) jc += c[indices[p]];
      jc *= jcoef;
      // 2. 減衰 + 3. 結合 → 場 F
      const coupledIn = SQRT_ETA * c[i] + jc;
      const Iin = coupledIn * coupledIn;
      // 4. 利得 (飽和つき) exp[½g0(1 − γ|F|²)]
      const halfG = halfG0 + negHalfG0Gamma * Iin;
      const sqrtGI = Math.exp(halfG);
      // 5. ノイズ (利得に比例)
      const noise = this.gauss() * noiseScale * sqrtGI;
      // 6. 次のパルス
      next[i] = sqrtGI * coupledIn + noise;
    }
    c.set(next);
    this.k++;
  }

  /** 現在のカット値 (符号が異なる辺の本数)。 */
  cut(): number {
    let cut = 0;
    const { c } = this;
    for (const [a, b] of this.edges) {
      if (c[a] > 0 !== c[b] > 0) cut++;
    }
    return cut;
  }

  meanAbsAmp(): number {
    let s = 0;
    for (let i = 0; i < this.n; i++) s += Math.abs(this.c[i]);
    return s / this.n;
  }

  snapshot(): CimSnapshot {
    const P = this.pumpW();
    return {
      round: this.k,
      pumpMW: P * 1e3,
      pumpRatio: P / P_TH_W,
      cut: this.cut(),
      meanAbsAmp: this.meanAbsAmp(),
      amps: this.c.slice(),
    };
  }
}
