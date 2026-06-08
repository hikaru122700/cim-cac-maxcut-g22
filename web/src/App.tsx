import { useMemo, useState } from "react";

import { CimSimulator } from "./components/CimSimulator";
import { CutDegreeHist } from "./components/CutDegreeHist";
import { FileDrop } from "./components/FileDrop";
import { SpinGrid } from "./components/SpinGrid";
import { Summary } from "./components/Summary";
import { computeCutStats } from "./lib/analysis";
import { ParseError, parseAssignment, parseGraph } from "./lib/parse";
import type { Assignment, Graph } from "./types";

type Page = "visualizer" | "cim";

interface LoadedGraph {
  graph: Graph;
  filename: string;
}
interface LoadedAssignment {
  assignment: Assignment;
  filename: string;
}

/** Known Best Score テーブル (表示用, Gset の代表例)。 */
const BKS_TABLE: Record<string, number> = {
  G22: 13359,
  G23: 13344,
  G24: 13337,
  G25: 13340,
  G26: 13328,
  G27: 3341,
  G28: 3298,
};

function detectBks(filename: string): number | undefined {
  const m = filename.match(/G(\d+)/i);
  if (!m) return undefined;
  return BKS_TABLE["G" + m[1]];
}

export default function App() {
  const [page, setPage] = useState<Page>("visualizer");

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-titlebar">
          <h1>Ising Machine Lab</h1>
          <nav className="page-tabs">
            <button
              className={page === "visualizer" ? "active" : ""}
              onClick={() => setPage("visualizer")}
            >
              MAX-CUT Visualizer
            </button>
            <button
              className={page === "cim" ? "active" : ""}
              onClick={() => setPage("cim")}
            >
              CIM Simulator
            </button>
          </nav>
        </div>
        <div className="subtitle">
          {page === "visualizer"
            ? "Gset 形式のグラフ + N 行 0/1 割当ファイルをアップロードして cut 値を可視化。"
            : "コヒーレント・イジングマシン (CIM) の測定フィードバック・ダイナミクスを 1 ラウンドずつ物理シミュレーション。"}
        </div>
      </header>

      {page === "cim" ? <CimSimulator /> : <Visualizer />}

      <footer className="app-footer">
        <span>Client-side only. No data leaves your browser.</span>
      </footer>
    </div>
  );
}

function Visualizer() {
  const [graphLoaded, setGraphLoaded] = useState<LoadedGraph | null>(null);
  const [assignLoaded, setAssignLoaded] = useState<LoadedAssignment | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);

  const onGraphFile = (text: string, filename: string) => {
    try {
      const graph = parseGraph(text);
      setGraphLoaded({ graph, filename });
      setError(null);
    } catch (e) {
      setGraphLoaded(null);
      setError(e instanceof ParseError ? e.message : String(e));
    }
  };

  const onAssignmentFile = (text: string, filename: string) => {
    try {
      const assignment = parseAssignment(text, graphLoaded?.graph.n);
      setAssignLoaded({ assignment, filename });
      setError(null);
    } catch (e) {
      setAssignLoaded(null);
      setError(e instanceof ParseError ? e.message : String(e));
    }
  };

  // graph が後から load されたときのサイズ不整合も検知
  const stats = useMemo(() => {
    if (!graphLoaded || !assignLoaded) return null;
    if (assignLoaded.assignment.side.length !== graphLoaded.graph.n) {
      return null;
    }
    try {
      return computeCutStats(graphLoaded.graph, assignLoaded.assignment);
    } catch (e) {
      console.error(e);
      return null;
    }
  }, [graphLoaded, assignLoaded]);

  const bks =
    graphLoaded && assignLoaded
      ? detectBks(graphLoaded.filename) ?? detectBks(assignLoaded.filename)
      : undefined;

  const sizeMismatch =
    graphLoaded &&
    assignLoaded &&
    assignLoaded.assignment.side.length !== graphLoaded.graph.n;

  return (
    <>
      <section className="upload-row">
        <FileDrop
          label="1. Graph file (.txt)"
          hint="1 行目: N K, 続く K 行: u v [w]  (1-indexed, Gset 形式)"
          accept=".txt,.gr,.graph"
          filename={graphLoaded?.filename ?? null}
          onFile={onGraphFile}
        />
        <FileDrop
          label="2. Assignment file (.txt)"
          hint="N 行, 各行 0 または 1 (または ±1)。空白区切りの 1 ファイルでも可。"
          accept=".txt,.out,.sol"
          filename={assignLoaded?.filename ?? null}
          onFile={onAssignmentFile}
        />
      </section>

      {error && <div className="error-banner">⚠ {error}</div>}
      {sizeMismatch && (
        <div className="error-banner">
          ⚠ サイズ不整合: graph N={graphLoaded!.graph.n} に対し
          assignment は {assignLoaded!.assignment.side.length} 行
        </div>
      )}

      {!graphLoaded && !assignLoaded && (
        <div className="placeholder">
          <div>ファイルをドラッグ&ドロップ、またはクリックして選択。</div>
          <div style={{ marginTop: 12, fontSize: 12 }}>
            サンプル: <code>input/G22.txt</code> +
            <code> scripts.save_assignment</code> の出力ファイル
          </div>
        </div>
      )}

      {graphLoaded && assignLoaded && stats && (
        <>
          <Summary graph={graphLoaded.graph} stats={stats} bks={bks} />
          <SpinGrid
            graph={graphLoaded.graph}
            assignment={assignLoaded.assignment}
            stats={stats}
          />
          <CutDegreeHist stats={stats} />
        </>
      )}

      <footer className="app-footer">
        <span>
          Client-side only. No data leaves your browser.
        </span>
      </footer>
    </div>
  );
}
