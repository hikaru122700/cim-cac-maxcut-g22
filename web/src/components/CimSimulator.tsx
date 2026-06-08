import { useCallback, useEffect, useRef, useState } from "react";

import {
  CimSim,
  DEFAULT_CONFIG,
  FILE_GRAPHS,
  P_TH_MW,
  buildSyntheticGraph,
  isFileGraph,
  parseGsetText,
  type CimConfig,
  type CimSnapshot,
  type GraphData,
  type GraphKind,
} from "../lib/cim";

interface HistPoint {
  round: number;
  cut: number;
  meanAmp: number;
}

const GRAPH_LABELS: Record<GraphKind, string> = {
  random: "ランダム (ER)",
  ring: "リング",
  complete: "完全グラフ",
  grid: "格子",
  G22: "G22 (Gset)",
  K2000: "K2000 (SK ±1)",
};

const POS = "#60a5fa";
const NEG = "#fb923c";

/**
 * CIM 物理シミュレーションのインタラクティブ可視化ページ。
 * パルス→利得→減衰→J結合→ノイズ→次パルス、のループを 1 ラウンドずつ回し、
 * 振幅・ポンプランプ・カット値・凍結曲線をリアルタイム描画する。
 */
export function CimSimulator() {
  const [cfg, setCfg] = useState<CimConfig>(DEFAULT_CONFIG);
  const [running, setRunning] = useState(false);
  const [snap, setSnap] = useState<CimSnapshot | null>(null);
  const [bestCut, setBestCut] = useState(0);
  const [speed, setSpeed] = useState(8);
  const [graphData, setGraphData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const simRef = useRef<CimSim | null>(null);
  const histRef = useRef<HistPoint[]>([]);
  const rafRef = useRef<number | null>(null);

  // --- グラフの読み込み (合成は同期 / ファイルは非同期 fetch) ---
  useEffect(() => {
    let ignore = false;
    setRunning(false);
    if (isFileGraph(cfg.graph)) {
      setLoading(true);
      setLoadError(null);
      const meta = FILE_GRAPHS[cfg.graph];
      fetch(meta.path)
        .then((r) => {
          if (!r.ok) throw new Error(`${meta.path} を取得できません (${r.status})`);
          return r.text();
        })
        .then((text) => {
          if (ignore) return;
          const g = parseGsetText(text);
          setGraphData(g);
          setLoading(false);
        })
        .catch((e: unknown) => {
          if (ignore) return;
          setLoadError(e instanceof Error ? e.message : String(e));
          setGraphData(null);
          setLoading(false);
        });
    } else {
      setLoadError(null);
      setLoading(false);
      setGraphData(
        buildSyntheticGraph(cfg.graph, cfg.n, cfg.edgeProb, cfg.seed),
      );
    }
    return () => {
      ignore = true;
    };
  }, [cfg.graph, cfg.n, cfg.edgeProb, cfg.seed]);

  // --- graphData / 物理パラメータが変わったら sim を作り直す ---
  const reset = useCallback(() => {
    if (!graphData) return;
    setRunning(false);
    const sim = new CimSim(cfg, graphData);
    simRef.current = sim;
    histRef.current = [{ round: 0, cut: sim.cut(), meanAmp: 0 }];
    setSnap(sim.snapshot());
    setBestCut(sim.cut());
  }, [cfg, graphData]);

  useEffect(() => {
    reset();
  }, [reset]);

  // --- アニメーションループ ---
  useEffect(() => {
    if (!running) return;
    const tick = () => {
      const sim = simRef.current;
      if (!sim) return;
      for (let s = 0; s < speed && !sim.done; s++) sim.step();
      const snapshot = sim.snapshot();
      histRef.current.push({
        round: snapshot.round,
        cut: snapshot.cut,
        meanAmp: snapshot.meanAbsAmp,
      });
      setSnap(snapshot);
      setBestCut((b) => Math.max(b, snapshot.cut));
      if (sim.done) {
        setRunning(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [running, speed]);

  const stepOnce = () => {
    const sim = simRef.current;
    if (!sim || sim.done) return;
    for (let s = 0; s < speed; s++) sim.step();
    const snapshot = sim.snapshot();
    histRef.current.push({
      round: snapshot.round,
      cut: snapshot.cut,
      meanAmp: snapshot.meanAbsAmp,
    });
    setSnap(snapshot);
    setBestCut((b) => Math.max(b, snapshot.cut));
  };

  const ampCanvas = useRef<HTMLCanvasElement>(null);
  const pumpCanvas = useRef<HTMLCanvasElement>(null);
  const cutCanvas = useRef<HTMLCanvasElement>(null);
  const freezeCanvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    drawAmps(ampCanvas.current, snap);
  }, [snap]);
  useEffect(() => {
    drawPump(pumpCanvas.current, cfg, snap);
  }, [snap, cfg]);
  useEffect(() => {
    drawSeries(cutCanvas.current, histRef.current, cfg.rounds, "cut", {
      color: POS,
      label: "カット値",
      asInt: !graphData?.weighted,
    });
  }, [snap, cfg.rounds, graphData]);
  useEffect(() => {
    drawSeries(freezeCanvas.current, histRef.current, cfg.rounds, "meanAmp", {
      color: "#fbbf24",
      label: "平均振幅 mean|c|",
      asInt: false,
    });
  }, [snap, cfg.rounds, graphData]);

  const set = <K extends keyof CimConfig>(key: K, value: CimConfig[K]) =>
    setCfg((c) => ({ ...c, [key]: value }));

  const fileGraph = isFileGraph(cfg.graph);
  const nDisplay = graphData?.n ?? cfg.n;
  const pumpRatio = snap?.pumpRatio ?? 0;
  const regime =
    pumpRatio < 0.97
      ? "しきい値未満 (探索)"
      : pumpRatio <= 1.03
        ? "臨界 (分岐)"
        : "しきい値超 (凍結)";

  return (
    <div className="cim">
      <section className="card">
        <div className="card-header">
          <h2>物理ループ — 1 ラウンドの流れ</h2>
        </div>
        <div className="cim-pipeline">
          <span>① パルス c</span>
          <span className="arrow">→</span>
          <span>② 減衰 √η·c</span>
          <span className="arrow">→</span>
          <span>③ 結合 +ΣJ_ij c_j</span>
          <span className="arrow">→</span>
          <span>④ 利得 ×exp[½g0(1−γ|F|²)]</span>
          <span className="arrow">→</span>
          <span>⑤ +ノイズ</span>
          <span className="arrow">→</span>
          <span>⑥ 次パルス c′</span>
          <span className="arrow">↺</span>
        </div>
      </section>

      <section className="cim-main">
        {/* 左: コントロール */}
        <div className="card cim-controls">
          <div className="card-header">
            <h2>パラメータ</h2>
          </div>

          <label className="ctrl">
            <span>グラフ</span>
            <select
              value={cfg.graph}
              onChange={(e) => set("graph", e.target.value as GraphKind)}
            >
              {(Object.keys(GRAPH_LABELS) as GraphKind[]).map((g) => (
                <option key={g} value={g}>
                  {GRAPH_LABELS[g]}
                </option>
              ))}
            </select>
          </label>

          {cfg.graph === "K2000" && (
            <div className="cim-note">
              ⚠ K2000 は 2000 頂点・約 200 万辺(±1, 22MB)。読込・1ラウンドが重いので
              速度を落として実行してください。
            </div>
          )}

          <Slider
            label="スピン数 N"
            value={nDisplay}
            min={4}
            max={64}
            step={1}
            disabled={fileGraph}
            onChange={(v) => set("n", v)}
          />
          {cfg.graph === "random" && (
            <Slider
              label="辺確率"
              value={cfg.edgeProb}
              min={0.1}
              max={1}
              step={0.05}
              fmt={(v) => v.toFixed(2)}
              onChange={(v) => set("edgeProb", v)}
            />
          )}
          <Slider
            label="ラウンド数"
            value={cfg.rounds}
            min={200}
            max={3000}
            step={100}
            onChange={(v) => set("rounds", v)}
          />
          <Slider
            label="ポンプ・ランプ率 [mW/round]"
            value={cfg.dPumpMW}
            min={0}
            max={0.15}
            step={0.005}
            fmt={(v) => (v === 0 ? "固定" : v.toFixed(3))}
            onChange={(v) => set("dPumpMW", v)}
          />
          <Slider
            label="ポンプ倍率 (×ランプ)"
            value={cfg.pumpMult}
            min={0.5}
            max={2}
            step={0.05}
            fmt={(v) => v.toFixed(2)}
            onChange={(v) => set("pumpMult", v)}
          />
          <Slider
            label="ノイズ振幅"
            value={cfg.noise}
            min={0}
            max={0.01}
            step={0.0005}
            fmt={(v) => v.toFixed(4)}
            onChange={(v) => set("noise", v)}
          />
          <Slider
            label="結合強度 |J|"
            value={cfg.coupling}
            min={0.005}
            max={0.1}
            step={0.005}
            fmt={(v) => v.toFixed(3)}
            onChange={(v) => set("coupling", v)}
          />
          <Slider
            label="seed"
            value={cfg.seed}
            min={1}
            max={50}
            step={1}
            onChange={(v) => set("seed", v)}
          />
          <Slider
            label="速度 (steps/frame)"
            value={speed}
            min={1}
            max={40}
            step={1}
            onChange={setSpeed}
          />

          <div className="cim-buttons">
            <button
              className={running ? "btn warn" : "btn primary"}
              onClick={() => setRunning((r) => !r)}
              disabled={loading || !graphData || (simRef.current?.done && !running)}
            >
              {running ? "⏸ 一時停止" : "▶ 実行"}
            </button>
            <button
              className="btn"
              onClick={stepOnce}
              disabled={running || loading || !graphData}
            >
              ⏭ ステップ
            </button>
            <button className="btn" onClick={reset} disabled={loading}>
              ⟲ リセット
            </button>
          </div>
        </div>

        {/* 右: 可視化 */}
        <div className="cim-viz">
          <div className="card">
            {loading && (
              <div className="cim-loading">
                ⏳ {FILE_GRAPHS[cfg.graph]?.label ?? cfg.graph} を読み込み中…
              </div>
            )}
            {loadError && <div className="error-banner">⚠ {loadError}</div>}
            <div className="cim-stats">
              <Stat label="round" value={`${snap?.round ?? 0} / ${cfg.rounds}`} />
              <Stat
                label="pump P/P_th"
                value={pumpRatio.toFixed(2)}
                sub={`${(snap?.pumpMW ?? 0).toFixed(1)} mW`}
              />
              <Stat label="状態" value={regime} small />
              <Stat label="cut" value={fmtNum(snap?.cut ?? 0)} hl />
              <Stat label="best cut" value={fmtNum(bestCut)} hl />
              <Stat
                label="mean |c|"
                value={(snap?.meanAbsAmp ?? 0).toFixed(3)}
              />
              <Stat label="N / edges" value={`${nDisplay} / ${fmtNum(graphData?.m ?? 0)}`} />
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2>In-phase 振幅 c_i (符号 = スピン ±)</h2>
            </div>
            <div className="legend">
              中心線から上=正(青)/下=負(橙)。発振すると ± に分かれてロックする。
            </div>
            <canvas ref={ampCanvas} style={{ width: "100%", height: 200 }} />
          </div>

          <div className="cim-row2">
            <div className="card">
              <div className="card-header">
                <h2>ポンプ・ランプ (時間発展)</h2>
              </div>
              <canvas ref={pumpCanvas} style={{ width: "100%", height: 180 }} />
            </div>
            <div className="card">
              <div className="card-header">
                <h2>カット値の推移</h2>
              </div>
              <canvas ref={cutCanvas} style={{ width: "100%", height: 180 }} />
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2>平均振幅 mean|c| (凍結曲線)</h2>
            </div>
            <div className="legend">
              0 から立ち上がって頭打ちになるほど、全スピンが発振・凍結したことを表す。
            </div>
            <canvas ref={freezeCanvas} style={{ width: "100%", height: 180 }} />
          </div>
        </div>
      </section>
    </div>
  );
}

// ---------- 小物コンポーネント ----------

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  fmt?: (v: number) => string;
  disabled?: boolean;
  onChange: (v: number) => void;
}
function Slider({
  label,
  value,
  min,
  max,
  step,
  fmt,
  disabled,
  onChange,
}: SliderProps) {
  return (
    <label className={"ctrl" + (disabled ? " disabled" : "")}>
      <span>
        {label} <b>{fmt ? fmt(value) : value}</b>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

interface StatProps {
  label: string;
  value: string;
  sub?: string;
  hl?: boolean;
  small?: boolean;
}
function Stat({ label, value, sub, hl, small }: StatProps) {
  return (
    <div className={"stat" + (hl ? " hl" : "")}>
      <span className="stat-label">{label}</span>
      <span className="stat-value" style={small ? { fontSize: 13 } : undefined}>
        {value}
      </span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

function fmtNum(v: number): string {
  return Number.isInteger(v) ? v.toLocaleString("en-US") : v.toFixed(1);
}

// ---------- 描画ヘルパ ----------

function setupCanvas(
  canvas: HTMLCanvasElement | null,
  cssH: number,
): { ctx: CanvasRenderingContext2D; W: number; H: number } | null {
  if (!canvas) return null;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 600;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = "#030712";
  ctx.fillRect(0, 0, cssW, cssH);
  return { ctx, W: cssW, H: cssH };
}

function drawAmps(canvas: HTMLCanvasElement | null, snap: CimSnapshot | null) {
  const s = setupCanvas(canvas, 200);
  if (!s || !snap) return;
  const { ctx, W, H } = s;
  const amps = snap.amps;
  const n = amps.length;
  let amax = 1e-9;
  for (let i = 0; i < n; i++) amax = Math.max(amax, Math.abs(amps[i]));
  const mid = H / 2;
  const padX = 8;
  const bw = (W - 2 * padX) / n;

  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, mid);
  ctx.lineTo(W - padX, mid);
  ctx.stroke();

  for (let i = 0; i < n; i++) {
    const v = amps[i];
    const h = (Math.abs(v) / amax) * (mid - 12);
    const x = padX + i * bw;
    ctx.fillStyle = v >= 0 ? POS : NEG;
    if (v >= 0) ctx.fillRect(x, mid - h, Math.max(1, bw - 1), h);
    else ctx.fillRect(x, mid, Math.max(1, bw - 1), h);
  }
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`|c|max = ${amax.toFixed(3)}  (N=${n})`, padX, 12);
}

function drawPump(
  canvas: HTMLCanvasElement | null,
  cfg: CimConfig,
  snap: CimSnapshot | null,
) {
  const s = setupCanvas(canvas, 180);
  if (!s) return;
  const { ctx, W, H } = s;
  const padL = 38;
  const padR = 10;
  const padT = 12;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const rounds = cfg.rounds;
  const pumpAt = (k: number) =>
    cfg.pumpMult * (cfg.dPumpMW > 0 ? (k + 1) * cfg.dPumpMW : P_TH_MW);
  const pMaxMW = Math.max(pumpAt(rounds - 1), P_TH_MW * 1.2);

  const x = (k: number) => padL + (k / rounds) * plotW;
  const y = (mw: number) => padT + plotH - (mw / pMaxMW) * plotH;

  ctx.strokeStyle = "#6b7280";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(padL, y(P_TH_MW));
  ctx.lineTo(padL + plotW, y(P_TH_MW));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#9ca3af";
  ctx.font = "10px monospace";
  ctx.textAlign = "left";
  ctx.fillText(`P_th=${P_TH_MW.toFixed(0)}mW`, padL + 2, y(P_TH_MW) - 3);

  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let k = 0; k <= rounds; k += Math.max(1, Math.floor(rounds / 200))) {
    const px = x(k);
    const py = y(pumpAt(k));
    if (k === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  if (snap) {
    const px = x(snap.round);
    ctx.strokeStyle = "#f9fafb";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, padT);
    ctx.lineTo(px, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = "#f9fafb";
    ctx.beginPath();
    ctx.arc(px, y(snap.pumpMW), 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = "#374151";
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "right";
  ctx.fillText(`${pMaxMW.toFixed(0)}`, padL - 3, padT + 8);
  ctx.fillText("0", padL - 3, padT + plotH);
  ctx.textAlign = "center";
  ctx.fillText("round", padL + plotW / 2, H - 5);
}

interface SeriesOpts {
  color: string;
  label: string;
  asInt: boolean;
}
function drawSeries(
  canvas: HTMLCanvasElement | null,
  hist: readonly HistPoint[],
  rounds: number,
  key: "cut" | "meanAmp",
  opts: SeriesOpts,
) {
  const s = setupCanvas(canvas, 180);
  if (!s || hist.length < 2) return;
  const { ctx, W, H } = s;
  const padL = 52;
  const padR = 12;
  const padT = 12;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  let vMin = Infinity;
  let vMax = -Infinity;
  for (const p of hist) {
    const v = p[key];
    vMin = Math.min(vMin, v);
    vMax = Math.max(vMax, v);
  }
  if (key === "meanAmp") vMin = 0;
  if (vMax - vMin < 1e-9) vMax = vMin + 1;

  const x = (k: number) => padL + (k / rounds) * plotW;
  const y = (v: number) => padT + plotH - ((v - vMin) / (vMax - vMin)) * plotH;

  ctx.strokeStyle = opts.color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  hist.forEach((p, i) => {
    const px = x(p.round);
    const py = y(p[key]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  ctx.strokeStyle = "#374151";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, padT);
  ctx.lineTo(padL, padT + plotH);
  ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();

  const fmt = (v: number) =>
    opts.asInt ? Math.round(v).toLocaleString("en-US") : v.toFixed(3);
  ctx.font = "10px monospace";
  ctx.fillStyle = opts.color;
  ctx.textAlign = "right";
  ctx.fillText(fmt(vMax), padL - 3, padT + 8);
  ctx.fillText(fmt(vMin), padL - 3, padT + plotH);
  ctx.fillStyle = "#9ca3af";
  ctx.textAlign = "center";
  ctx.fillText(opts.label, padL + plotW / 2, H - 5);
}
