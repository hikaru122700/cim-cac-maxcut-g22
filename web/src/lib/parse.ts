/**
 * Gset グラフファイルと 0/1 割当ファイルのパーサ。
 */

import type { Assignment, Graph } from "../types";

export class ParseError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ParseError";
  }
}

/**
 * Gset 形式のグラフテキストをパース。
 *
 * 期待形式:
 *   N K
 *   u1 v1 w1
 *   u2 v2 w2
 *   ...
 *
 * u, v は 1-indexed (Gset 慣習)。内部では 0-indexed に変換。
 * 空行・空白のみの行は無視。
 */
export function parseGraph(text: string): Graph {
  const lines = text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0 && !l.startsWith("#"));

  if (lines.length === 0) {
    throw new ParseError("graph file is empty");
  }

  const header = lines[0].split(/\s+/);
  if (header.length < 2) {
    throw new ParseError(
      `graph header must be "N K", got: "${lines[0]}"`,
    );
  }
  const n = parseInt(header[0], 10);
  const k = parseInt(header[1], 10);
  if (!Number.isFinite(n) || !Number.isFinite(k) || n <= 0 || k < 0) {
    throw new ParseError(`invalid N K: ${lines[0]}`);
  }

  const edges: [number, number, number][] = [];
  for (let i = 1; i < lines.length; i++) {
    const parts = lines[i].split(/\s+/);
    if (parts.length < 2) {
      throw new ParseError(
        `line ${i + 1}: expected "u v [w]", got "${lines[i]}"`,
      );
    }
    const u = parseInt(parts[0], 10) - 1;
    const v = parseInt(parts[1], 10) - 1;
    const w = parts.length >= 3 ? parseFloat(parts[2]) : 1;
    if (!Number.isInteger(u) || !Number.isInteger(v)) {
      throw new ParseError(`line ${i + 1}: non-integer vertex`);
    }
    if (u < 0 || u >= n || v < 0 || v >= n) {
      throw new ParseError(
        `line ${i + 1}: vertex out of range (N=${n}): u=${u + 1}, v=${v + 1}`,
      );
    }
    if (u === v) {
      // 自己ループは無視 (MAX-CUT では cut に寄与しない)
      continue;
    }
    edges.push([u, v, Number.isFinite(w) ? w : 1]);
  }

  if (edges.length !== k) {
    // 警告レベル: header の K と実際の辺数が食い違う場合も読めたら続行
    console.warn(
      `graph header says K=${k} but parsed ${edges.length} edges`,
    );
  }

  return { n, k: edges.length, edges };
}

/**
 * 0/1 割当テキストをパース。
 *
 * 期待形式: N 行, 各行が "0" または "1" (0 = partition 0, 1 = partition 1)。
 * 複数値が 1 行に並ぶケース (空白/カンマ区切り) にも対応。
 * 空行・`#` 始まりのコメント行は無視。
 */
export function parseAssignment(text: string, expectedN?: number): Assignment {
  const tokens: string[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (line.length === 0 || line.startsWith("#")) continue;
    for (const t of line.split(/[\s,]+/)) {
      if (t.length > 0) tokens.push(t);
    }
  }

  if (tokens.length === 0) {
    throw new ParseError("assignment file is empty");
  }

  const side: boolean[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i];
    let v: boolean;
    if (t === "0" || t === "-1") v = false;
    else if (t === "1" || t === "+1") v = true;
    else {
      throw new ParseError(
        `assignment token ${i + 1}: expected 0/1 (or ±1), got "${t}"`,
      );
    }
    side.push(v);
  }

  if (expectedN !== undefined && side.length !== expectedN) {
    throw new ParseError(
      `assignment length mismatch: got ${side.length}, expected ${expectedN}`,
    );
  }

  return { side };
}
