"""GA の選別が悪化する原因の切り分け(中間発表 v20 の講評・論点 4)。新しい GA は回さない。

ga_equal_time の本実験では、選別ありの条件(cim_cut・cim_div)は CIM K = 4N 回分の時間を差し引かれ、
選別なし(cim_all)は N 回分だけ。そこで選別ありも N 回分しか課金しない版を作り、同じ評価時刻で比べる。
  それでも選別ありが悪い → 時間ではなく、選別で初期集団の何かが失われている
  差が消える             → 悪化の原因は選別に使った CIM の時間

使い方(プロジェクトルートから):
  python scripts/benchmarks/ga_selection_charge_check.py --run results/2026-11-09/ga_equal_time/v1_..._pop10_main
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import scripts.benchmarks.ga_equal_time_analysis as gea  # noqa: E402

EXPERIMENT_KIND = "ga_charge_check"
SELECTED = ("cim_cut", "cim_div")


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    heads = [p.name.split("_", 1)[0] for p in kind_root.iterdir() if p.is_dir() and p.name.startswith("v")]
    return max([int(h[1:]) for h in heads if h[1:].isdigit()], default=0) + 1


def gaps_at_eval(run: Path, meta: dict, ds: str, charge_pop_only: bool) -> dict[str, np.ndarray]:
    """評価時刻(本実験と同じ)での各条件の BKS との差 [%]。charge_pop_only なら選別ありも N 回分だけ課金。"""
    info, t = meta["datasets"][ds], meta["timing"][ds]
    info_used = {**info, "K": info["pop_size"]} if charge_pop_only else info
    z = np.load(run / f"{ds}.npz")
    te = gea.t_eval(info, t, int(meta["args"]["eval_gens"]))  # 評価時刻は本実験のまま(K = 4N で決めた値)
    return {c: (info["bks"] - gea.best_at(te, c, 1.0, z, info_used, t)) / info["bks"] * 100
            for c in meta["conditions"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    meta = json.loads((args.run / "meta.json").read_text(encoding="utf-8"))
    out = {"run": str(args.run), "datasets": {}}
    for ds in meta["datasets"]:
        row = {}
        for label, pop_only in (("charged_K", False), ("charged_N", True)):
            g = gaps_at_eval(args.run, meta, ds, pop_only)
            row[label] = {"mean_gap_pct": {c: float(np.mean(v)) for c, v in g.items()},
                          "p_vs_cim_all": {c: float(mannwhitneyu(g[c], g["cim_all"], alternative="two-sided").pvalue)
                                           for c in SELECTED}}
        out["datasets"][ds] = row
        m = row["charged_N"]["mean_gap_pct"]
        print(ds, "charged N:", {c: round(v, 4) for c, v in m.items()},
              {c: round(p, 3) for c, p in row["charged_N"]["p_vs_cim_all"].items()})
    kind_root = get_kind_root()
    dst = kind_root / f"v{next_version(kind_root)}_{args.run.name}" if not args.tag else \
        kind_root / f"v{next_version(kind_root)}_{args.tag}"
    dst.mkdir(parents=True)
    (dst / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved:", dst)


if __name__ == "__main__":
    main()
