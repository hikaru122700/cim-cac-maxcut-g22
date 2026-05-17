import { useEffect, useMemo, useRef, useState } from "react";

import type { Assignment, CutStats, Graph } from "../types";

interface Props {
  graph: Graph;
  assignment: Assignment;
  stats: CutStats;
}

type ColorMode = "partition" | "cutdeg" | "improvable";

/**
 * スピン状態を 2D グリッドキャンバスで描画。
 *
 * - partition: 単純に 0/1 の色分け (blue / red)
 * - cutdeg:    cut degree / degree の割合で色相 (多いほど蛍光)
 * - improvable: フリップで cut が増える頂点だけ黄色でハイライト (局所最適からの距離)
 */
export function SpinGrid({ graph, assignment, stats }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<ColorMode>("partition");

  const { cols, rows } = useMemo(() => gridShape(graph.n), [graph.n]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = cols;
    canvas.height = rows;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = ctx.createImageData(cols, rows);
    const { side } = assignment;
    const { cutDegrees, degrees } = stats;
    for (let i = 0; i < graph.n; i++) {
      const base = i * 4;
      let r = 0, g = 0, b = 0;
      if (mode === "partition") {
        if (side[i]) {
          r = 60; g = 130; b = 250;
        } else {
          r = 240; g = 80; b = 80;
        }
      } else if (mode === "cutdeg") {
        const frac = degrees[i] > 0 ? cutDegrees[i] / degrees[i] : 0;
        // base color from partition, lightness from frac
        if (side[i]) {
          r = Math.round(40 + 180 * frac);
          g = Math.round(80 + 140 * frac);
          b = Math.round(200 + 55 * frac);
        } else {
          r = Math.round(200 + 55 * frac);
          g = Math.round(60 + 120 * frac);
          b = Math.round(60 + 120 * frac);
        }
      } else {
        // improvable
        const delta = degrees[i] - 2 * cutDegrees[i];
        if (delta > 0) {
          // フリップで +delta 改善する: 黄色、強度 ~ delta / degree
          const a = degrees[i] > 0 ? delta / degrees[i] : 0;
          r = Math.round(180 + 60 * a);
          g = Math.round(180 + 60 * a);
          b = Math.round(40 + 40 * a);
        } else {
          // 局所最適 or 中立: 暗いグレー、partition の色相だけ残す
          if (side[i]) {
            r = 25; g = 40; b = 70;
          } else {
            r = 70; g = 25; b = 25;
          }
        }
      }
      img.data[base + 0] = r;
      img.data[base + 1] = g;
      img.data[base + 2] = b;
      img.data[base + 3] = 255;
    }
    // 余りセル
    for (let i = graph.n; i < cols * rows; i++) {
      const base = i * 4;
      img.data[base + 0] = 10;
      img.data[base + 1] = 12;
      img.data[base + 2] = 20;
      img.data[base + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [graph, assignment, stats, mode, cols, rows]);

  return (
    <section className="card">
      <div className="card-header">
        <h2>Spin Grid (index order)</h2>
        <div className="mode-tabs">
          <button
            className={mode === "partition" ? "active" : ""}
            onClick={() => setMode("partition")}
          >
            partition
          </button>
          <button
            className={mode === "cutdeg" ? "active" : ""}
            onClick={() => setMode("cutdeg")}
          >
            cut-degree
          </button>
          <button
            className={mode === "improvable" ? "active" : ""}
            onClick={() => setMode("improvable")}
          >
            improvable
          </button>
        </div>
      </div>
      <div className="legend">
        {mode === "partition" && (
          <span>
            <b style={{ color: "#60a5fa" }}>■ +1 side</b> /{" "}
            <b style={{ color: "#ef4444" }}>■ −1 side</b>
          </span>
        )}
        {mode === "cutdeg" &&
          "明度 = cut_degree / degree (相手側に伸びる辺の割合)"}
        {mode === "improvable" &&
          "黄 = フリップで cut が増える頂点 (局所最適でない頂点)"}
      </div>
      <canvas
        ref={canvasRef}
        style={{
          width: "100%",
          aspectRatio: `${cols} / ${rows}`,
          imageRendering: "pixelated",
          background: "#030712",
          borderRadius: 4,
          display: "block",
        }}
      />
    </section>
  );
}

function gridShape(n: number): { cols: number; rows: number } {
  if (n <= 0) return { cols: 1, rows: 1 };
  const cols = Math.max(1, Math.round(Math.sqrt(n * 1.25)));
  const rows = Math.ceil(n / cols);
  return { cols, rows };
}
