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

/** 合成グラフ種別 + 実ベンチマーク (Gset/K2000)。 */
export type GraphKind =
  | "ring"
  | "random"
  | "complete"
  | "grid"
  | "G22"
  | "K2000";

/** 内部表現: 辺を 3 本の TypedArray で持つ (大規模グラフ対応)。 */
export interface GraphData {
  readonly n: number;
  readonly m: number;
  readonly ea: Int32Array;
  readonly eb: Int32Array;
  readonly ew: Float64Array;
  readonly weighted: boolean;
}

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

/** ファイルから読み込む実グラフの種別と公開パス。 */
export const FILE_GRAPHS: Record<string, { path: string; n: number; label: string }> = {
  G22: { path: "graphs/G22.txt", n: 2000, label: "G22 (Gset, N=2000)" },
  K2000: { path: "graphs/K2000.txt", n: 2000, label: "K2000 (SK ±1, N=2000)" },
};

export function isFileGraph(kind: GraphKind): boolean {
  return kind === "G22" || kind === "K2000";
}

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

/** 合成グラフを生成する (重みは全て 1)。 */
export function buildSyntheticGraph(
  kind: GraphKind,
  n: number,
  edgeProb: number,
  seed: number,
): GraphData {
  const rand = mulberry32(seed ^ 0x9e3779b9);
  const a: number[] = [];
  const b: number[] = [];
  const push = (u: number, v: number) => {
    a.push(u);
    b.push(v);
  };
  if (kind === "ring") {
    for (let i = 0; i < n; i++) push(i, (i + 1) % n);
  } else if (kind === "complete") {
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) push(i, j);
  } else if (kind === "grid") {
    const side = Math.max(2, Math.round(Math.sqrt(n)));
    for (let r = 0; r < side; r++) {
      for (let c = 0; c < side; c++) {
        const v = r * side + c;
        if (v >= n) continue;
        if (c + 1 < side && v + 1 < n) push(v, v + 1);
        if (r + 1 < side && v + side < n) push(v, v + side);
      }
    }
  } else {
    for (let i = 0; i < n; i++) push(i, (i + 1) % n);
    for (let i = 0; i < n; i++)
      for (let j = i + 2; j < n; j++) if (rand() < edgeProb) push(i, j);
  }
  const m = a.length;
  const ew = new Float64Array(m).fill(1);
  return {
    n,
    m,
    ea: Int32Array.from(a),
    eb: Int32Array.from(b),
    ew,
    weighted: false,
  };
}

/**
 * Gset 形式テキストをパースする。
 * 1 行目 "N K", 続く K 行 "u v w" (1-indexed)。大規模ファイル向けに
 * 文字列を 1 パスでスキャンして整数を直接取り出す。
 */
export function parseGsetText(text: string): GraphData {
  const len = text.length;
  let pos = 0;
  let sign = 1;
  // 整数を 1 つ読む
  const nextInt = (): number => {
    while (pos < len) {
      const ch = text.charCodeAt(pos);
      if (ch === 45) {
        // '-'
        sign = -1;
        pos++;
        break;
      }
      if (ch >= 48 && ch <= 57) break;
      pos++;
    }
    let val = 0;
    let any = false;
    while (pos < len) {
      const ch = text.charCodeAt(pos);
      if (ch < 48 || ch > 57) break;
      val = val * 10 + (ch - 48);
      pos++;
      any = true;
    }
    const out = any ? sign * val : NaN;
    sign = 1;
    return out;
  };

  const n = nextInt();
  const k = nextInt();
  if (!Number.isFinite(n) || !Number.isFinite(k)) {
    throw new Error("Gset ヘッダ (N K) を読めません");
  }
  const ea = new Int32Array(k);
  const eb = new Int32Array(k);
  const ew = new Float64Array(k);
  let weighted = false;
  let idx = 0;
  for (let e = 0; e < k; e++) {
    const u = nextInt();
    const v = nextInt();
    const w = nextInt();
    if (!Number.isFinite(u) || !Number.isFinite(v)) break;
    ea[idx] = u - 1;
    eb[idx] = v - 1;
    ew[idx] = Number.isFinite(w) ? w : 1;
    if (ew[idx] !== 1) weighted = true;
    idx++;
  }
  return {
    n,
    m: idx,
    ea: ea.subarray(0, idx),
    eb: eb.subarray(0, idx),
    ew: ew.subarray(0, idx),
    weighted,
  };
}

/** 重み付き隣接リスト (CSR)。 */
function buildAdjacency(g: GraphData): {
  indptr: Int32Array;
  indices: Int32Array;
  wdata: Float64Array;
} {
  const { n, m, ea, eb, ew } = g;
  const deg = new Int32Array(n);
  for (let e = 0; e < m; e++) {
    deg[ea[e]]++;
    deg[eb[e]]++;
  }
  const indptr = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) indptr[i + 1] = indptr[i] + deg[i];
  const total = indptr[n];
  const indices = new Int32Array(total);
  const wdata = new Float64Array(total);
  const cursor = indptr.slice(0, n);
  for (let e = 0; e < m; e++) {
    const a = ea[e];
    const b = eb[e];
    const w = ew[e];
    indices[cursor[a]] = b;
    wdata[cursor[a]] = w;
    cursor[a]++;
    indices[cursor[b]] = a;
    wdata[cursor[b]] = w;
    cursor[b]++;
  }
  return { indptr, indices, wdata };
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
  readonly graph: GraphData;
  private readonly cfg: CimConfig;
  private readonly indptr: Int32Array;
  private readonly indices: Int32Array;
  private readonly wdata: Float64Array;
  private readonly c: Float64Array;
  private readonly next: Float64Array;
  private readonly jcoef: number; // 反強磁性: 負
  private readonly gauss: () => number;
  private k = 0;

  constructor(cfg: CimConfig, graph: GraphData) {
    this.cfg = cfg;
    this.graph = graph;
    this.n = graph.n;
    const adj = buildAdjacency(graph);
    this.indptr = adj.indptr;
    this.indices = adj.indices;
    this.wdata = adj.wdata;
    this.c = new Float64Array(graph.n);
    this.next = new Float64Array(graph.n);
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
    const ramp =
      this.cfg.dPumpMW > 0 ? (this.k + 1) * this.cfg.dPumpMW : P_TH_MW;
    return (this.cfg.pumpMult * ramp) / 1e3;
  }

  /** 1 ラウンド (パルス 1 周) を進める。 */
  step(): void {
    if (this.done) return;
    const { n, c, next, indptr, indices, wdata, jcoef } = this;
    const P = this.pumpW();
    const g0 = TWO_KAPPA_L * Math.sqrt(P);
    const halfG0 = 0.5 * g0;
    const negHalfG0Gamma = -0.5 * g0 * GAMMA;
    const noiseScale = this.cfg.noise;

    for (let i = 0; i < n; i++) {
      // 3. 結合 (Σ_j J_ij c_j = MFB)
      let jc = 0.0;
      const end = indptr[i + 1];
      for (let p = indptr[i]; p < end; p++) jc += wdata[p] * c[indices[p]];
      jc *= jcoef;
      // 2. 減衰 + 3. 結合 → 場 F
      const coupledIn = SQRT_ETA * c[i] + jc;
      const Iin = coupledIn * coupledIn;
      // 4. 利得 (飽和つき) exp[½g0(1 − γ|F|²)]
      const sqrtGI = Math.exp(halfG0 + negHalfG0Gamma * Iin);
      // 5. ノイズ (利得に比例) → 6. 次のパルス
      next[i] = sqrtGI * coupledIn + this.gauss() * noiseScale * sqrtGI;
    }
    c.set(next);
    this.k++;
  }

  /** 現在のカット値 (符号が異なる辺の重み総和; 無重みなら本数)。 */
  cut(): number {
    let cut = 0;
    const { c, graph } = this;
    const { ea, eb, ew, m } = graph;
    for (let e = 0; e < m; e++) {
      if (c[ea[e]] > 0 !== c[eb[e]] > 0) cut += ew[e];
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
