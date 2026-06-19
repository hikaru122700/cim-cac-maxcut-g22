"""seed_autoresearch.py — 実実験の analysis/decision を AutoResearchClaw に注入する。

本物実装で得た実結果(analysis.md, decision.md)を run_dir の stage-14 / stage-15 に
配置し、checkpoint.json を「stage-15 まで完了」にして、
`researchclaw run --from-stage PAPER_OUTLINE --output <run_dir>` で
執筆段だけを駆動できるようにする(記憶の教訓: おもちゃ実験は使わず本物実験を入力にする)。

artifact 解決は run_dir/stage-*/<filename> を走査して名前一致で行われるので
(researchclaw/pipeline/_helpers.py:_read_prior_artifact)、stage 番号は緩く合わせる。

使い方:
  python scripts/utils/seed_autoresearch.py \
    --run-dir AutoResearchClaw/artifacts/maxcut_algo_combo \
    --analysis results/.../analysis.md \
    --decision results/.../decision.md \
    --figdir   results/.../figs
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--figdir", default=None, help="図を stage-14/charts へコピー")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    s14 = run_dir / "stage-14"
    s15 = run_dir / "stage-15"
    s14.mkdir(parents=True, exist_ok=True)
    s15.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.analysis, s14 / "analysis.md")
    shutil.copy(args.decision, s15 / "decision.md")

    # 図をコピー(paper 段が参照できるよう charts/ に)
    if args.figdir:
        charts = s14 / "charts"
        charts.mkdir(exist_ok=True)
        for p in Path(args.figdir).glob("*.png"):
            shutil.copy(p, charts / p.name)

    # checkpoint: stage-15(RESEARCH_DECISION)まで完了とする
    ckpt = {
        "last_completed_stage": 15,
        "last_completed_name": "RESEARCH_DECISION",
        "run_id": "seeded-real-experiments",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(run_dir / "checkpoint.json", "w", encoding="utf-8") as f:
        json.dump(ckpt, f, indent=2)

    print(f"seeded: {s14/'analysis.md'}")
    print(f"seeded: {s15/'decision.md'}")
    print(f"checkpoint → stage-15 complete")
    print(f"\n次に実行:\n  cd AutoResearchClaw\n  .venv/Scripts/python.exe -m researchclaw run "
          f"--topic \"<topic>\" --output {run_dir.name if run_dir.is_absolute() else args.run_dir} "
          f"--from-stage PAPER_OUTLINE")


if __name__ == "__main__":
    main()
