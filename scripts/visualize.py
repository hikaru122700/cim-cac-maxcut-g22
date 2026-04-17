"""
イジングマシン最適化の AHC 風インタラクティブビジュアライザ。

RunRecord (シミュレーション履歴 + 集計) を受け取って、Plotly.js +
Canvas2D ベースの自己完結 HTML ファイルを生成する。

設計方針:
  - 純粋ロジック (データ変換、ヒストグラム生成、HTML レンダリング) は
    ファイル I/O やブラウザ無しで単体テスト可能
  - HTML は自己完結 (Plotly.js だけ CDN 参照)
  - データ型は frozen dataclass (不変)
  - AHC 風プレーヤー: 再生/停止/seek バー/キーボード/速度変更
  - スピン空間ビュー: 2000 スピンをグリッドに並べ符号 × |x| で描画

使い方:

    from scripts.visualize import (
        RunRecord, TrajectorySnapshot, SpinFrame, write_html
    )
    record = RunRecord(
        method="CAC", graph_name="G22", n_spins=2000, n_edges=19990,
        num_trials=100, num_outer_steps=50000, config={...},
        final_cuts=(...), trajectory=(...), spin_frames=(...),
        wall_time_sec=12.3, timestamp="2026-04-18T12:34:56",
        target_cut=13359,
    )
    path = write_html(record, Path("results/viz/cac_run.html"))
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


# ============================================================
#  Data model
# ============================================================
@dataclass(frozen=True)
class TrajectorySnapshot:
    """1 外ループステップ時点の代表 trial の集計状態スナップショット。"""

    step: int
    cut: int
    best_cut: int
    mean_abs_x: float
    std_abs_x: float
    mean_e: float
    std_e: float
    beta_inj: float
    a_t: float
    num_positive: int
    beta_reset: bool
    improvement: bool


@dataclass(frozen=True)
class SpinFrame:
    """AHC 風プレーヤー用の per-spin 状態フレーム。

    x_q: 量子化された per-spin 振幅 (int8, -127..127)
    scale: 逆量子化係数 (x_real ≒ x_q * scale)
    cut: この時点の cut 値 (チャート同期用)
    """

    step: int
    cut: int
    scale: float
    x_q: Sequence[int]


@dataclass(frozen=True)
class RunRecord:
    """1 回の実行全体の記録 (HTML レンダラへの入力)。"""

    method: str
    graph_name: str
    n_spins: int
    n_edges: int
    num_trials: int
    num_outer_steps: int
    config: Mapping[str, Any]
    final_cuts: Sequence[int]
    trajectory: Sequence[TrajectorySnapshot]
    wall_time_sec: float
    timestamp: str
    target_cut: int = 13359
    spin_frames: Sequence[SpinFrame] = field(default_factory=tuple)


# ============================================================
#  Pure logic (テスト対象)
# ============================================================
def snapshots_to_arrays(
    snaps: Sequence[TrajectorySnapshot],
) -> dict[str, list]:
    """スナップショット列を Plotly 向けのカラム配列辞書に転置する。"""
    return {
        "step": [s.step for s in snaps],
        "cut": [s.cut for s in snaps],
        "best_cut": [s.best_cut for s in snaps],
        "mean_abs_x": [s.mean_abs_x for s in snaps],
        "std_abs_x": [s.std_abs_x for s in snaps],
        "mean_e": [s.mean_e for s in snaps],
        "std_e": [s.std_e for s in snaps],
        "beta_inj": [s.beta_inj for s in snaps],
        "a_t": [s.a_t for s in snaps],
        "num_positive": [s.num_positive for s in snaps],
        "reset_steps": [s.step for s in snaps if s.beta_reset],
        "improve_steps": [s.step for s in snaps if s.improvement],
        "improve_cuts": [s.best_cut for s in snaps if s.improvement],
    }


def spin_frames_to_arrays(
    frames: Sequence[SpinFrame],
) -> dict[str, list]:
    """SpinFrame 列を JSON シリアライズしやすい形に転置する。"""
    return {
        "step": [f.step for f in frames],
        "cut": [f.cut for f in frames],
        "scale": [f.scale for f in frames],
        "x_q": [list(f.x_q) for f in frames],
    }


def histogram_bins(
    values: Sequence[int], num_bins: int = 30
) -> dict[str, list]:
    """単純な等幅ヒストグラム。空/単一値のエッジケースも扱う。"""
    if not values:
        return {"counts": [], "edges": [], "centers": []}
    lo, hi = min(values), max(values)
    if lo == hi:
        return {
            "counts": [len(values)],
            "edges": [float(lo), float(lo + 1)],
            "centers": [float(lo) + 0.5],
        }
    width = (hi - lo) / num_bins
    edges = [lo + i * width for i in range(num_bins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(num_bins)]
    counts = [0] * num_bins
    for v in values:
        idx = int((v - lo) / width)
        if idx >= num_bins:
            idx = num_bins - 1
        counts[idx] += 1
    return {"counts": counts, "edges": edges, "centers": centers}


def summary_stats(
    final_cuts: Sequence[int], target_cut: int
) -> dict[str, float]:
    """集計統計 (平均/最大/最小/BKS 到達回数)。"""
    if not final_cuts:
        return {"mean": 0.0, "max": 0, "min": 0, "num_optimal": 0}
    return {
        "mean": sum(final_cuts) / len(final_cuts),
        "max": max(final_cuts),
        "min": min(final_cuts),
        "num_optimal": sum(1 for c in final_cuts if c == target_cut),
    }


def grid_shape_for(n: int) -> tuple[int, int]:
    """N スピンを描画する際の (cols, rows) を決める (縦横比 ~1.25)。"""
    if n <= 0:
        return (1, 1)
    cols = max(1, int(round((n * 1.25) ** 0.5)))
    rows = (n + cols - 1) // cols
    return (cols, rows)


def build_plot_data(record: RunRecord) -> dict[str, Any]:
    """HTML に埋め込む JSON 用のデータ辞書を構築。"""
    cols, rows = grid_shape_for(record.n_spins)
    return {
        "method": record.method,
        "graph_name": record.graph_name,
        "n_spins": record.n_spins,
        "n_edges": record.n_edges,
        "num_trials": record.num_trials,
        "num_outer_steps": record.num_outer_steps,
        "target_cut": record.target_cut,
        "wall_time_sec": record.wall_time_sec,
        "timestamp": record.timestamp,
        "config": {k: _json_safe(v) for k, v in record.config.items()},
        "final_cuts": list(record.final_cuts),
        "trajectory": snapshots_to_arrays(record.trajectory),
        "histogram": histogram_bins(list(record.final_cuts), num_bins=30),
        "stats": summary_stats(list(record.final_cuts), record.target_cut),
        "spin_frames": spin_frames_to_arrays(record.spin_frames),
        "grid": {"cols": cols, "rows": rows},
    }


def _json_safe(v: Any) -> Any:
    """numpy スカラーなどを Python 組込み型に変換。"""
    if hasattr(v, "item"):
        try:
            return v.item()
        except (ValueError, AttributeError):
            pass
    return v


# ============================================================
#  HTML template (AHC 風プレーヤー内蔵)
# ============================================================
_HTML_HEAD = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>__METHOD__ Run — __GRAPH__ — __TIMESTAMP__</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Yu Gothic', sans-serif;
         margin: 0; padding: 16px; background: #0b1220; color: #e5e7eb; }
  h1 { margin: 0 0 8px 0; font-size: 20px; color: #f3f4f6; }
  h2 { margin: 24px 0 8px 0; font-size: 15px; color: #93c5fd;
       border-left: 3px solid #3b82f6; padding-left: 8px; }
  .summary { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px;
             background: #111827; padding: 12px; border-radius: 6px; }
  .stat { display: flex; flex-direction: column; min-width: 90px; }
  .stat-label { font-size: 11px; color: #9ca3af; text-transform: uppercase;
                letter-spacing: 0.5px; }
  .stat-value { font-size: 18px; color: #f9fafb; font-weight: 600;
                font-family: 'SF Mono', Consolas, monospace; }
  .chart { background: #111827; padding: 8px; border-radius: 6px;
           margin: 8px 0; }
  table { border-collapse: collapse; margin-top: 8px; min-width: 320px; }
  td, th { padding: 4px 12px; font-size: 12px;
           border-bottom: 1px solid #374151; }
  th { color: #9ca3af; text-align: left; font-weight: 400; }
  td.key { color: #93c5fd; font-family: monospace; }
  td.value { font-family: monospace; color: #f9fafb; text-align: right; }
  .legend-note { font-size: 11px; color: #9ca3af; margin: 4px 0 0 4px; }

  /* ---- AHC プレーヤー UI ---- */
  .player { background: #111827; padding: 12px; border-radius: 6px;
            margin: 8px 0; }
  .player-row { display: flex; align-items: center; gap: 10px;
                flex-wrap: wrap; margin-bottom: 8px; }
  .player button { background: #1f2937; color: #e5e7eb; border: 1px solid #374151;
                   border-radius: 4px; padding: 6px 10px; font-size: 13px;
                   font-family: monospace; cursor: pointer;
                   min-width: 44px; }
  .player button:hover { background: #2d3748; }
  .player button.active { background: #2563eb; border-color: #3b82f6; }
  .player input[type=range] { flex: 1; min-width: 200px; accent-color: #3b82f6; }
  .player select { background: #1f2937; color: #e5e7eb;
                   border: 1px solid #374151; padding: 4px 6px;
                   font-family: monospace; font-size: 12px; border-radius: 4px; }
  .player .readout { font-family: monospace; font-size: 13px;
                     color: #f9fafb; min-width: 90px; text-align: right; }
  .player .hint { font-size: 10px; color: #6b7280; margin-left: auto; }

  /* ---- キャンバス ---- */
  .canvas-wrap { display: flex; gap: 16px; flex-wrap: wrap; }
  .canvas-box { flex: 1; min-width: 360px; background: #111827;
                padding: 8px; border-radius: 6px; }
  canvas { display: block; width: 100%; height: auto;
           image-rendering: pixelated; background: #030712;
           border-radius: 4px; }
  .canvas-label { font-size: 11px; color: #9ca3af; margin-bottom: 4px;
                  font-family: monospace; }
  .frame-info { margin-left: auto; font-family: monospace;
                font-size: 12px; color: #9ca3af; }
</style>
</head>
<body>
<h1>__METHOD__ Run — __GRAPH__ (N=__N_SPINS__, K=__N_EDGES__)</h1>
<div class="summary">
  <div class="stat"><div class="stat-label">Trials</div><div class="stat-value">__NUM_TRIALS__</div></div>
  <div class="stat"><div class="stat-label">Outer Steps</div><div class="stat-value">__NUM_STEPS__</div></div>
  <div class="stat"><div class="stat-label">Mean Cut</div><div class="stat-value">__MEAN_CUT__</div></div>
  <div class="stat"><div class="stat-label">Max Cut</div><div class="stat-value">__MAX_CUT__</div></div>
  <div class="stat"><div class="stat-label">Min Cut</div><div class="stat-value">__MIN_CUT__</div></div>
  <div class="stat"><div class="stat-label">Optimal</div><div class="stat-value">__NUM_OPTIMAL__ / __NUM_TRIALS__</div></div>
  <div class="stat"><div class="stat-label">Target (BKS)</div><div class="stat-value">__TARGET_CUT__</div></div>
  <div class="stat"><div class="stat-label">Wall Time</div><div class="stat-value">__WALL_TIME__s</div></div>
  <div class="stat"><div class="stat-label">Timestamp</div><div class="stat-value" style="font-size:11px">__TIMESTAMP__</div></div>
</div>

<!-- ======================================================
     AHC 風 プレーヤー
     ====================================================== -->
<h2>インタラクティブプレーヤー</h2>
<div class="legend-note">
  スピン状態と各指標を時間軸に沿って再生。スライダーで seek、
  キーボード: ← / → (±1 frame), Shift+←/→ (±10), Space (play/pause)
</div>
<div class="player" id="player" tabindex="0">
  <div class="player-row">
    <button id="btn-begin" title="先頭へ">⏮</button>
    <button id="btn-prev" title="1 フレーム戻る">◀</button>
    <button id="btn-play" title="再生/停止">▶</button>
    <button id="btn-next" title="1 フレーム進む">▶|</button>
    <button id="btn-end" title="末尾へ">⏭</button>
    <span class="readout" id="readout-step">step 0</span>
    <label style="font-size:11px;color:#9ca3af">speed
      <select id="select-speed">
        <option value="2">0.5×</option>
        <option value="1">1×</option>
        <option value="0.5" selected>2×</option>
        <option value="0.25">4×</option>
        <option value="0.1">10×</option>
      </select>
    </label>
    <span class="readout" id="readout-cut">cut=—</span>
    <span class="hint">keys: ←/→ ±1, Shift+←/→ ±10, Space</span>
  </div>
  <div class="player-row">
    <input type="range" id="slider" min="0" max="0" value="0" step="1">
  </div>
</div>

<div class="canvas-wrap">
  <div class="canvas-box">
    <div class="canvas-label">
      Spin grid (index order) — blue: x&gt;0 (+1), red: x&lt;0 (−1), brightness ∝ |x|
      <span class="frame-info" id="frame-info"></span>
    </div>
    <canvas id="spin-canvas" width="600" height="480"></canvas>
  </div>
  <div class="canvas-box">
    <div class="canvas-label">
      Spin sorted by |x| (descending) — 振幅の成長順
    </div>
    <canvas id="sort-canvas" width="600" height="80"></canvas>
    <div class="canvas-label" style="margin-top:8px">
      |x| amplitude distribution (this frame)
    </div>
    <canvas id="hist-canvas" width="600" height="180"></canvas>
  </div>
</div>

<h2>1. Cut 値の時間発展 (代表 trial)</h2>
<div class="legend-note">黄線 = best_cut (単調), 青線 = 現在の cut, ×印 = β_inj リセット, ▲ = 改善点, 点線 = 目標 (BKS), 縦線 = 再生ヘッド</div>
<div id="chart_cut" class="chart"></div>

<h2>2. 振幅ダイナミクス |x_i|</h2>
<div class="legend-note">左軸: mean |x|, 右軸: std |x|。α の大小で収束先が変わる。</div>
<div id="chart_x" class="chart"></div>

<h2>3. エラー変数 e_i (CAC フィードバック)</h2>
<div class="legend-note">左軸: mean e, 右軸: std e。e は結合強度の動的補正係数。</div>
<div id="chart_e" class="chart"></div>

<h2>4. 結合ランプ β_inj と目標振幅² a(t)</h2>
<div class="legend-note">左軸: β_inj (γ で線形成長, τ でリセット), 右軸: a(t) = α + ρ·tanh(δ·ΔH)</div>
<div id="chart_beta" class="chart"></div>

<h2>5. スピン対称性 (正スピン数)</h2>
<div class="legend-note">点線 = N/2 (完全バランス)。対称性の崩れ方が収束の質を示す。</div>
<div id="chart_sign" class="chart"></div>

<h2>6. 最終 cut 分布 (全 __NUM_TRIALS__ trial)</h2>
<div class="legend-note">黄点線 = 目標 (BKS)。分布の右端がピーク到達能力を示す。</div>
<div id="chart_hist" class="chart"></div>

<h2>7. ハイパーパラメータ</h2>
<table>
  <tr><th>Parameter</th><th>Value</th></tr>
  __CONFIG_ROWS__
</table>

<script id="viz-data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('viz-data').textContent);

// ------------------------------------------------------------
//  Plotly charts (既存の時系列 + 再生ヘッド shape)
// ------------------------------------------------------------
const darkBase = {
  paper_bgcolor: '#111827', plot_bgcolor: '#111827',
  font: { color: '#e5e7eb', size: 11, family: 'SF Mono, Consolas, monospace' },
  margin: { l: 60, r: 60, t: 20, b: 40 }, height: 260,
  legend: { orientation: 'h', y: 1.12, bgcolor: 'rgba(0,0,0,0)' },
};
const axis = { gridcolor: '#374151', zerolinecolor: '#374151' };
const playheadShape = (step, yref) => ({
  type: 'line', xref: 'x', yref: yref || 'paper',
  x0: step, x1: step, y0: 0, y1: 1,
  line: { color: '#22d3ee', width: 1, dash: 'solid' },
});

// --- 1. Cut trajectory ---
const resetY = DATA.trajectory.reset_steps.map(() => DATA.target_cut);
const improveY = DATA.trajectory.improve_cuts;
Plotly.newPlot('chart_cut', [
  { x: DATA.trajectory.step, y: DATA.trajectory.cut,
    name: 'current cut', mode: 'lines',
    line: { color: '#60a5fa', width: 1 } },
  { x: DATA.trajectory.step, y: DATA.trajectory.best_cut,
    name: 'best_cut', mode: 'lines',
    line: { color: '#fbbf24', width: 2 } },
  { x: DATA.trajectory.reset_steps, y: resetY,
    name: 'β_inj reset', mode: 'markers',
    marker: { color: '#ef4444', symbol: 'x', size: 9 } },
  { x: DATA.trajectory.improve_steps, y: improveY,
    name: 'improvement', mode: 'markers',
    marker: { color: '#34d399', symbol: 'triangle-up', size: 8 } },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'outer step' }),
  yaxis: Object.assign({}, axis, { title: 'cut' }),
  shapes: [
    { type: 'line', x0: 0, x1: DATA.num_outer_steps,
      y0: DATA.target_cut, y1: DATA.target_cut,
      line: { color: '#9ca3af', width: 1, dash: 'dot' } },
    playheadShape(0),
  ],
}), { responsive: true });

// --- 2. Amplitude ---
Plotly.newPlot('chart_x', [
  { x: DATA.trajectory.step, y: DATA.trajectory.mean_abs_x,
    name: 'mean |x|', mode: 'lines', line: { color: '#60a5fa' } },
  { x: DATA.trajectory.step, y: DATA.trajectory.std_abs_x,
    name: 'std |x|', mode: 'lines', line: { color: '#a78bfa' }, yaxis: 'y2' },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'outer step' }),
  yaxis: Object.assign({}, axis, { title: 'mean |x|' }),
  yaxis2: Object.assign({}, axis, {
    overlaying: 'y', side: 'right', title: 'std |x|', color: '#a78bfa' }),
  shapes: [playheadShape(0)],
}), { responsive: true });

// --- 3. Error variable e ---
Plotly.newPlot('chart_e', [
  { x: DATA.trajectory.step, y: DATA.trajectory.mean_e,
    name: 'mean e', mode: 'lines', line: { color: '#34d399' } },
  { x: DATA.trajectory.step, y: DATA.trajectory.std_e,
    name: 'std e', mode: 'lines', line: { color: '#fbbf24' }, yaxis: 'y2' },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'outer step' }),
  yaxis: Object.assign({}, axis, { title: 'mean e' }),
  yaxis2: Object.assign({}, axis, {
    overlaying: 'y', side: 'right', title: 'std e', color: '#fbbf24' }),
  shapes: [playheadShape(0)],
}), { responsive: true });

// --- 4. beta_inj & a_t ---
Plotly.newPlot('chart_beta', [
  { x: DATA.trajectory.step, y: DATA.trajectory.beta_inj,
    name: 'β_inj', mode: 'lines', line: { color: '#f87171' } },
  { x: DATA.trajectory.step, y: DATA.trajectory.a_t,
    name: 'a(t)', mode: 'lines', line: { color: '#60a5fa' }, yaxis: 'y2' },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'outer step' }),
  yaxis: Object.assign({}, axis, { title: 'β_inj' }),
  yaxis2: Object.assign({}, axis, {
    overlaying: 'y', side: 'right', title: 'a(t)', color: '#60a5fa' }),
  shapes: [playheadShape(0)],
}), { responsive: true });

// --- 5. spin balance ---
Plotly.newPlot('chart_sign', [
  { x: DATA.trajectory.step, y: DATA.trajectory.num_positive,
    name: 'positive spins', mode: 'lines', line: { color: '#fbbf24' } },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'outer step' }),
  yaxis: Object.assign({}, axis, { title: '# positive spins',
                                    range: [0, DATA.n_spins] }),
  shapes: [
    { type: 'line', x0: 0, x1: DATA.num_outer_steps,
      y0: DATA.n_spins / 2, y1: DATA.n_spins / 2,
      line: { color: '#9ca3af', width: 1, dash: 'dot' } },
    playheadShape(0),
  ],
}), { responsive: true });

// --- 6. final cut histogram ---
const hist = DATA.histogram;
const maxCount = Math.max(...hist.counts, 1);
Plotly.newPlot('chart_hist', [
  { x: hist.centers, y: hist.counts, type: 'bar',
    marker: { color: '#60a5fa' }, name: 'trials' },
], Object.assign({}, darkBase, {
  xaxis: Object.assign({}, axis, { title: 'cut value' }),
  yaxis: Object.assign({}, axis, { title: 'count' }),
  shapes: [{ type: 'line', x0: DATA.target_cut, x1: DATA.target_cut,
             y0: 0, y1: maxCount,
             line: { color: '#fbbf24', width: 2, dash: 'dash' } }],
  annotations: [{ x: DATA.target_cut, y: maxCount,
                  text: 'BKS=' + DATA.target_cut, showarrow: false,
                  xanchor: 'left', yanchor: 'top',
                  font: { color: '#fbbf24', size: 11 } }],
}), { responsive: true });

// ------------------------------------------------------------
//  AHC-style player
// ------------------------------------------------------------
const SF = DATA.spin_frames;
const NUM_FRAMES = SF.step ? SF.step.length : 0;
const N_SPINS = DATA.n_spins;
const GRID_COLS = DATA.grid.cols;
const GRID_ROWS = DATA.grid.rows;

const slider = document.getElementById('slider');
const btnBegin = document.getElementById('btn-begin');
const btnPrev = document.getElementById('btn-prev');
const btnPlay = document.getElementById('btn-play');
const btnNext = document.getElementById('btn-next');
const btnEnd = document.getElementById('btn-end');
const selectSpeed = document.getElementById('select-speed');
const readoutStep = document.getElementById('readout-step');
const readoutCut = document.getElementById('readout-cut');
const frameInfo = document.getElementById('frame-info');
const playerEl = document.getElementById('player');

const spinCanvas = document.getElementById('spin-canvas');
const sortCanvas = document.getElementById('sort-canvas');
const histCanvas = document.getElementById('hist-canvas');
const spinCtx = spinCanvas.getContext('2d');
const sortCtx = sortCanvas.getContext('2d');
const histCtx = histCanvas.getContext('2d');

// 内部バックバッファ: グリッド 1 セル = 1 ピクセルで描いて CSS で拡大
spinCanvas.width = GRID_COLS;
spinCanvas.height = GRID_ROWS;
// 表示サイズを比率維持で固定
(function fitSpinCanvas() {
  const ratio = GRID_ROWS / GRID_COLS;
  spinCanvas.style.width = '100%';
  spinCanvas.style.height = Math.round(600 * ratio) + 'px';
})();

let currentFrame = 0;
let playing = false;
let playTimer = null;

if (NUM_FRAMES > 0) {
  slider.max = NUM_FRAMES - 1;
} else {
  // フレーム 0 件: プレーヤー無効化
  playerEl.style.opacity = '0.5';
  playerEl.style.pointerEvents = 'none';
  frameInfo.textContent = '(spin_frames 無し — 集計のみの記録)';
}

function renderFrame(idx) {
  if (NUM_FRAMES === 0) return;
  idx = Math.max(0, Math.min(NUM_FRAMES - 1, idx));
  currentFrame = idx;
  slider.value = String(idx);

  const step = SF.step[idx];
  const cut = SF.cut[idx];
  const scale = SF.scale[idx];
  const xq = SF.x_q[idx];

  readoutStep.textContent = 'step ' + step + ' (frame ' + (idx + 1) + '/' + NUM_FRAMES + ')';
  readoutCut.textContent = 'cut=' + cut;
  frameInfo.textContent = 'step=' + step + '  cut=' + cut + '  scale=' + scale.toExponential(2);

  // --- spin grid ---
  const img = spinCtx.createImageData(GRID_COLS, GRID_ROWS);
  for (let i = 0; i < xq.length; i++) {
    const v = xq[i];       // int8
    const a = Math.min(1, Math.abs(v) / 96); // 強度 (127 より早く飽和)
    const base = i * 4;
    if (v >= 0) { // +1 side (blue)
      img.data[base + 0] = Math.round(30 + 80 * a);
      img.data[base + 1] = Math.round(80 + 100 * a);
      img.data[base + 2] = Math.round(200 + 55 * a);
    } else {      // -1 side (red)
      img.data[base + 0] = Math.round(220 + 35 * a);
      img.data[base + 1] = Math.round(60 + 40 * a);
      img.data[base + 2] = Math.round(60 + 40 * a);
    }
    img.data[base + 3] = 255;
  }
  // 未使用セル (グリッドに余りがある場合)
  for (let i = xq.length; i < GRID_COLS * GRID_ROWS; i++) {
    const base = i * 4;
    img.data[base + 0] = 10; img.data[base + 1] = 12;
    img.data[base + 2] = 20; img.data[base + 3] = 255;
  }
  spinCtx.putImageData(img, 0, 0);

  // --- sorted lane (top: 振幅降順) ---
  const sorted = xq.slice().sort((a, b) => Math.abs(b) - Math.abs(a));
  const sw = sortCanvas.width, sh = sortCanvas.height;
  sortCtx.fillStyle = '#030712';
  sortCtx.fillRect(0, 0, sw, sh);
  for (let i = 0; i < sorted.length; i++) {
    const v = sorted[i];
    const a = Math.min(1, Math.abs(v) / 96);
    const x = Math.floor(i * sw / sorted.length);
    const w = Math.max(1, Math.ceil(sw / sorted.length));
    sortCtx.fillStyle = v >= 0
      ? 'rgb(' + Math.round(30 + 80 * a) + ',' + Math.round(80 + 100 * a) + ',' + Math.round(200 + 55 * a) + ')'
      : 'rgb(' + Math.round(220 + 35 * a) + ',' + Math.round(60 + 40 * a) + ',' + Math.round(60 + 40 * a) + ')';
    sortCtx.fillRect(x, 0, w, sh);
  }

  // --- amplitude histogram ---
  const hw = histCanvas.width, hh = histCanvas.height;
  histCtx.fillStyle = '#030712';
  histCtx.fillRect(0, 0, hw, hh);
  const NB = 40;
  const bins = new Array(NB).fill(0);
  for (let i = 0; i < xq.length; i++) {
    // -127..127 → 0..NB-1
    const b = Math.min(NB - 1, Math.max(0, Math.floor((xq[i] + 128) * NB / 256)));
    bins[b] += 1;
  }
  const bmax = Math.max(...bins, 1);
  const bw = hw / NB;
  for (let b = 0; b < NB; b++) {
    const bh = (bins[b] / bmax) * (hh - 20);
    const xc = (b - NB / 2) / (NB / 2); // -1..1
    histCtx.fillStyle = xc >= 0 ? '#60a5fa' : '#ef4444';
    histCtx.fillRect(b * bw, hh - bh - 10, bw - 1, bh);
  }
  // ゼロ線
  histCtx.strokeStyle = '#6b7280';
  histCtx.setLineDash([3, 3]);
  histCtx.beginPath();
  histCtx.moveTo(hw / 2, 0); histCtx.lineTo(hw / 2, hh);
  histCtx.stroke();
  histCtx.setLineDash([]);
  histCtx.fillStyle = '#9ca3af';
  histCtx.font = '10px monospace';
  histCtx.fillText('x=0', hw / 2 + 4, 12);
  histCtx.fillText('−|x|', 4, hh - 2);
  histCtx.fillText('+|x|', hw - 32, hh - 2);

  // --- 全チャートの playhead shape 更新 ---
  const chartIds = ['chart_cut', 'chart_x', 'chart_e', 'chart_beta', 'chart_sign'];
  chartIds.forEach((id, chartIdx) => {
    const gd = document.getElementById(id);
    const shapes = (gd.layout.shapes || []).slice();
    // 再生ヘッドは常に最後 (playheadShape を最後に push している)
    shapes[shapes.length - 1] = playheadShape(step);
    Plotly.relayout(gd, { shapes: shapes });
  });
}

function step(delta) {
  renderFrame(currentFrame + delta);
}

function setPlaying(on) {
  playing = on;
  btnPlay.textContent = on ? '⏸' : '▶';
  btnPlay.classList.toggle('active', on);
  if (playTimer) { clearInterval(playTimer); playTimer = null; }
  if (on && NUM_FRAMES > 0) {
    const mul = parseFloat(selectSpeed.value);
    const interval = Math.max(16, 80 * mul); // 80ms @ 1x
    playTimer = setInterval(() => {
      if (currentFrame >= NUM_FRAMES - 1) {
        setPlaying(false);
        return;
      }
      step(1);
    }, interval);
  }
}

// --- event wiring ---
slider.addEventListener('input', (ev) => {
  renderFrame(parseInt(ev.target.value, 10));
});
btnBegin.addEventListener('click', () => { setPlaying(false); renderFrame(0); });
btnEnd.addEventListener('click', () => { setPlaying(false); renderFrame(NUM_FRAMES - 1); });
btnPrev.addEventListener('click', () => { setPlaying(false); step(-1); });
btnNext.addEventListener('click', () => { setPlaying(false); step(1); });
btnPlay.addEventListener('click', () => setPlaying(!playing));
selectSpeed.addEventListener('change', () => { if (playing) setPlaying(true); });

document.addEventListener('keydown', (ev) => {
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT') return;
  if (ev.code === 'Space') {
    ev.preventDefault();
    setPlaying(!playing);
  } else if (ev.key === 'ArrowLeft') {
    ev.preventDefault(); setPlaying(false);
    step(ev.shiftKey ? -10 : -1);
  } else if (ev.key === 'ArrowRight') {
    ev.preventDefault(); setPlaying(false);
    step(ev.shiftKey ? 10 : 1);
  } else if (ev.key === 'Home') {
    ev.preventDefault(); setPlaying(false); renderFrame(0);
  } else if (ev.key === 'End') {
    ev.preventDefault(); setPlaying(false); renderFrame(NUM_FRAMES - 1);
  }
});

// 初回描画
if (NUM_FRAMES > 0) {
  renderFrame(0);
}
</script>
</body>
</html>
"""


def render_html(record: RunRecord) -> str:
    """RunRecord から自己完結 HTML 文字列を生成する (ファイル I/O なし)。"""
    data = build_plot_data(record)
    stats = data["stats"]
    config_rows = "\n  ".join(
        f'<tr><td class="key">{_escape_html(str(k))}</td>'
        f'<td class="value">{_format_config_value(v)}</td></tr>'
        for k, v in data["config"].items()
    )
    data_json = _json_for_html_embed(data)

    replacements = {
        "__METHOD__": _escape_html(record.method),
        "__GRAPH__": _escape_html(record.graph_name),
        "__N_SPINS__": str(record.n_spins),
        "__N_EDGES__": str(record.n_edges),
        "__NUM_TRIALS__": str(record.num_trials),
        "__NUM_STEPS__": str(record.num_outer_steps),
        "__MEAN_CUT__": f"{stats['mean']:.2f}",
        "__MAX_CUT__": str(stats["max"]),
        "__MIN_CUT__": str(stats["min"]),
        "__NUM_OPTIMAL__": str(stats["num_optimal"]),
        "__TARGET_CUT__": str(record.target_cut),
        "__WALL_TIME__": f"{record.wall_time_sec:.1f}",
        "__TIMESTAMP__": _escape_html(record.timestamp),
        "__CONFIG_ROWS__": config_rows,
        "__DATA_JSON__": data_json,
    }
    html = _HTML_HEAD
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


def _format_config_value(v: Any) -> str:
    if isinstance(v, float):
        return _escape_html(f"{v:.6g}")
    return _escape_html(str(v))


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def _json_for_html_embed(obj: Any) -> str:
    """HTML 内 <script> に埋め込むための安全な JSON エンコード。

    - `<`, `>`, `&` を \\u00XX エスケープ → HTML パーサが script タグを
      切らない / HTML コメントとして解釈しない
    - U+2028, U+2029 (行区切り) もエスケープ → 古い JS パーサ対策
    """
    s = json.dumps(obj, ensure_ascii=False)
    return (
        s.replace("<", "\\u003c")
         .replace(">", "\\u003e")
         .replace("&", "\\u0026")
         .replace("\u2028", "\\u2028")
         .replace("\u2029", "\\u2029")
    )


def write_html(record: RunRecord, output_path: Path) -> Path:
    """HTML をファイルに書き出し、書いたパスを返す。"""
    html = render_html(record)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
