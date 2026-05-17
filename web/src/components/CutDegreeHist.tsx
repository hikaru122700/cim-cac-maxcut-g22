import { useEffect, useRef } from "react";

import { cutDegreeHistogram } from "../lib/analysis";
import type { CutStats } from "../types";

interface Props {
  stats: CutStats;
}

/**
 * cut-degree ヒストグラム (離散 bin)。
 * x = 各頂点の cut 次数, y = 頂点数。
 */
export function CutDegreeHist({ stats }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 600;
    const cssH = 220;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { counts, max } = cutDegreeHistogram(stats.cutDegrees);
    const W = cssW, H = cssH;

    // 背景
    ctx.fillStyle = "#030712";
    ctx.fillRect(0, 0, W, H);

    if (counts.length === 0) return;

    const padL = 40, padR = 12, padT = 10, padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const yMax = Math.max(...counts, 1);

    // bars
    const binW = plotW / counts.length;
    for (let i = 0; i < counts.length; i++) {
      const h = (counts[i] / yMax) * plotH;
      const x = padL + i * binW;
      const y = padT + plotH - h;
      ctx.fillStyle = "#60a5fa";
      ctx.fillRect(x + 0.5, y, Math.max(1, binW - 1), h);
    }

    // axes
    ctx.strokeStyle = "#374151";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // labels
    ctx.fillStyle = "#9ca3af";
    ctx.font = "11px monospace";
    ctx.textAlign = "right";
    ctx.fillText(String(yMax), padL - 4, padT + 10);
    ctx.fillText("0", padL - 4, padT + plotH);
    ctx.textAlign = "center";
    ctx.fillText("0", padL, padT + plotH + 14);
    ctx.fillText(String(max), padL + plotW, padT + plotH + 14);
    ctx.textAlign = "left";
    ctx.fillText("cut degree per vertex", padL, H - 4);
  }, [stats]);

  return (
    <section className="card">
      <div className="card-header">
        <h2>Cut-degree distribution</h2>
      </div>
      <div className="legend">
        各頂点が相手パーティションへ伸ばす辺の本数の分布。
        右に裾が長いほど「境界」の頂点が多く、左に寄るほど内側で閉じている。
      </div>
      <canvas ref={canvasRef} style={{ width: "100%", height: 220 }} />
    </section>
  );
}
