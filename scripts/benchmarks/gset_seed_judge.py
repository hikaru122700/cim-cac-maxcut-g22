"""ステップ 3 の判定(H3a〜H3d)を gset_seed_analysis.py の result.json から出す。

判定基準は docs/20261018/step3_plan.md で実行前に固定した:
  H3a  CIM の出力が多様な問題(異なる状態が候補の半分以上)で、開始温度 0.5 のいずれかの予算で C1
  H3b  開始温度 2.0・5000 掃引で C1 が成り立つ問題で、E(eig_diff)の区間の下限 > 0
  H3c  C1 が成り立つ条件で trial_sa_mean の区間の下限 > 0
  H3d  C1 が成り立つ条件で ls_n_gain_minus2_to_0 の区間の上限 < 0
加えて各指標の「C1 が成り立つ条件のうち、区間が予想の向き(なければ両側)で 0 を除いた割合」。

使い方(プロジェクトルートから):
  python scripts/benchmarks/gset_seed_judge.py --result results/2026-10-18/gset_seed_analysis/v2_step3/result.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# 予想の向き(+1 / -1)。記載のない指標は両側で数える。
EXPECTED_SIGN = {"eig_diff": 1, "trial_sa_mean": 1, "trial_sa_max": 1, "ls_n_gain_minus2_to_0": -1}


def excludes_zero(ci: list[float], sign: int | None) -> bool:
    lo, hi = ci
    if sign == 1:
        return lo > 0
    if sign == -1:
        return hi < 0
    return lo > 0 or hi < 0


def judge(rows: list[dict]) -> dict:
    go_rows = [r for r in rows if r["c1_go"]]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    h3a = {}
    for ds, rs in by_ds.items():
        ref = next((r for r in rs if r["t_start"] == 0.5), rs[0])  # H3a は開始温度 0.5 の実行で決める
        diverse = ref["distinct_states"] >= ref["candidates"] / 2
        h3a[ds] = {"distinct_states": ref["distinct_states"], "diverse": diverse,
                   "c1_at_T0.5": [r["sweeps"] for r in rs if r["t_start"] == 0.5 and r["c1_go"]]}

    def per_condition(feature: str, cond) -> list[dict]:
        out = []
        for r in go_rows:
            if not cond(r) or feature not in r["features"]:
                continue
            f = r["features"][feature]
            out.append({"dataset": r["dataset"], "t_start": r["t_start"], "sweeps": r["sweeps"],
                        "rho": f["rho"], "ci95": f["ci95"],
                        "hit": excludes_zero(f["ci95"], EXPECTED_SIGN.get(feature))})
        return out

    h3b = per_condition("eig_diff", lambda r: r["t_start"] == 2.0 and r["sweeps"] == 5000)
    h3c = per_condition("trial_sa_mean", lambda r: True)
    h3d = per_condition("ls_n_gain_minus2_to_0", lambda r: True)
    features = sorted({f for r in go_rows for f in r["features"]})
    summary = {}
    for f in features:
        cs = per_condition(f, lambda r: True)
        summary[f] = {"sign": EXPECTED_SIGN.get(f), "n_conditions": len(cs),
                      "n_hit": sum(c["hit"] for c in cs)}
    return {"n_conditions": len(rows), "n_c1_go": len(go_rows),
            "n_undetermined": sum(r["c1_go"] is None for r in rows),
            "H3a": h3a, "H3b": h3b, "H3c": h3c, "H3d": h3d, "feature_summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, required=True)
    args = ap.parse_args()
    if not args.result.is_file():
        ap.error(f"not found: {args.result}")
    rows = json.loads(args.result.read_text(encoding="utf-8"))["rows"]
    res = judge(rows)
    print(f"conditions={res['n_conditions']} C1 go={res['n_c1_go']} undetermined={res['n_undetermined']}")
    print("H3a:")
    for ds, v in res["H3a"].items():
        print(f"  {ds:>4} distinct={v['distinct_states']:>3} diverse={v['diverse']!s:<5} C1@T0.5 sweeps={v['c1_at_T0.5']}")
    for key in ["H3b", "H3c", "H3d"]:
        hits = sum(c["hit"] for c in res[key])
        print(f"{key}: {hits}/{len(res[key])}")
        for c in res[key]:
            print(f"  {c['dataset']:>4} T={c['t_start']:<4g} sw={c['sweeps']:<5} rho={c['rho']:+.2f} "
                  f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] {'HIT' if c['hit'] else ''}")
    print("feature summary (hits / C1 conditions):")
    for f, v in res["feature_summary"].items():
        print(f"  {f:<24} sign={v['sign']!s:<4} {v['n_hit']}/{v['n_conditions']}")
    out = args.result.parent / "judgement.json"
    if out.exists():
        raise SystemExit(f"refuse to overwrite {out}")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
