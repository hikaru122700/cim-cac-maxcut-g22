"""ステップ 4:CIM の種の選別を、同じ総時間で比べる(論文化計画 v2)。

パイプライン:CIM で K 個の候補 → 指標で m 個を選ぶ → 各々に予算 B の SA を 1 回 → 最良値。
ステップ 3 の出力(候補ごとに独立な SA を R 回)から、選んだ候補の結果を 1 つずつ引いて
新しい SA を回さずに比べる。総時間は実測の処理時間で揃える:

  T = K·t_cim + m·t_sa(B)      (CIM で K 個作り、そのまま m 個に SA)
  余分な費用(LS・指標の計算・試し打ち)を使う方法は、その分だけ最後にかける SA の本数を減らす。
  LS と指標の費用は selection_costs.py の実測(1 スレッド)を PARALLEL で割る。CIM と SA の時間は
  ステップ 3 の PARALLEL スレッドでの実測(1 本あたりに直した値)なので、同じ並列度に揃える。
  費用は方法ごとに使う処理だけを数える(cim_ls は LS、model_eig は LS + 内積 + 固有ベクトル 1 回 など)。
  ランダム初期化の SA は、CIM の時間も SA に回す(T / t_sa(B) 本)。SA は同じ実装で反転数も
  同じなので、1 本の時間は CIM の種の場合と同じ t_sa(B)(ステップ 3 の実測)を使う。

学習モデル(model_*)は、ほかの問題のデータだけで学習したリッジ回帰で、
問題ごとに標準化した指標から、問題ごとに標準化したラベルを予測する(未見の問題への予測)。

使い方(プロジェクトルートから):
  python scripts/benchmarks/selection_eval.py --runs <step3 dirs> --random <random_init_sa dirs> \
      --costs <selection_costs.json> --t-start 0.5 --sweeps 5000 --pool 256 --m 16 --tag main
  --free-costs を付けると LS・指標・試し打ちの費用を 0 にする(選別の効果の上限)。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EXPERIMENT_KIND = "selection_eval"
MC = 4000
MC_SEED = 20261021
RIDGE = 1.0
TARGET_QUANTILE = 0.99
LOCAL = ["n_gain_minus2_to_0", "ls_n_gain_minus2_to_0", "n_zero_gain", "ls_n_zero_gain"]
EIG = ["eig_diff", "eig_a1", "eig_a2"]
MODELS = {"model_all": LOCAL + EIG, "model_local": LOCAL, "model_eig": EIG}
BASELINE = "cim_ls"
PARALLEL = 24
RANDOM_T_START = 2.0  # ランダム初期化の SA の開始温度(step4_plan.md)  # ステップ 3 の NUMBA_NUM_THREADS
# 主条件の判定(docs/20261018/step4_plan.md):(名前, 方法 A, 方法 B)で A − B > 0 を片側検定
PRIMARY = [("H4a", "model_all", "cim_ls"), ("H4b", "cim_trial", "cim_ls"), ("H4c", "cim_ls", "random_init")]


def get_kind_root() -> Path:
    out = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out


def next_version(kind_root: Path) -> int:
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def with_timing(p: dict, timing: dict) -> dict:
    """pipeline_timing.py の実測(ほかの計算がない状態)で t_cim・t_sa・t_trial を置き換えた新しい dict。"""
    t = timing[p["name"]]
    t_sa = t["t_sa"]
    return {**p, "t_cim": t["t_cim"], "t_sa": t_sa[f"T{p['t_start']:g}_S{p['sweeps']}"],
            "t_trial": p["trial_runs"] * t_sa[f"T{p['t_start']:g}_S50"]}


def load_problem(run_dir: Path, sweeps: int) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    meta = summary["meta"]
    z = np.load(run_dir / "raw.npz")
    n = z["cim_signs"].shape[1]
    budget = sweeps * n
    if budget not in meta["budgets"]:
        raise ValueError(f"{run_dir}: budget {budget} not found")
    k, r = z[f"cuts_B{budget}"].shape
    trial_b = 50 * n
    t_sa = summary["budgets"][str(budget)]["sa_time_s"] / (k * r)
    t_trial_run = summary["budgets"][str(trial_b)]["sa_time_s"] / (k * r)  # 試し打ちと同じ予算の 1 本
    feats = {f: z[f"feat_{f}"].astype(float) for f in LOCAL + EIG + ["trial_sa_mean"] if f"feat_{f}" in z.files}
    return {
        "name": meta["dataset"], "n": n, "bks": meta["bks"], "t_start": meta["t_start"],
        "cuts": z[f"cuts_B{budget}"].astype(float), "init_cut": z["init_cut"].astype(float),
        "ls_cut": z["ls_cut"].astype(float), "signs": z["cim_signs"], "feats": feats,
        "t_cim": meta["cim_time_s"] / k, "t_sa": t_sa, "t_trial": meta.get("trial", {}).get("runs", 0) * t_trial_run,
        "sweeps": sweeps, "trial_runs": meta.get("trial", {}).get("runs", 0),
    }


def design(p: dict, names: list[str]) -> np.ndarray:
    cols = [zscore(p["init_cut"]), zscore(p["ls_cut"])] + [zscore(p["feats"][f]) for f in names]
    return np.column_stack(cols)


def lopo_scores(problems: list[dict], names: list[str]) -> list[np.ndarray]:
    """問題を 1 つ抜いて学習したリッジ回帰の予測値(抜いた問題での選別に使う)。"""
    X = [design(p, names) for p in problems]
    y = [zscore(p["cuts"].mean(axis=1)) for p in problems]
    out = []
    for i in range(len(problems)):
        Xtr = np.vstack([X[j] for j in range(len(problems)) if j != i])
        ytr = np.concatenate([y[j] for j in range(len(problems)) if j != i])
        beta = np.linalg.solve(Xtr.T @ Xtr + RIDGE * np.eye(Xtr.shape[1]), Xtr.T @ ytr)
        out.append(X[i] @ beta)
    return out


def top_m(score: np.ndarray, pool: np.ndarray, m: int, rng) -> np.ndarray:
    """pool の中で score の上位 m 個(同点は乱数で崩す)。"""
    jitter = rng.random(len(pool)) * 1e-9
    order = np.lexsort((jitter, -score[pool]))
    return pool[order[:m]]


def diverse_m(p: dict, pool: np.ndarray, m: int, m_pool: int, rng) -> np.ndarray:
    """LS 後 cut の上位 2·m_pool 個から、互いの距離(全反転を同一視)が大きくなるよう貪欲に m 個選ぶ。
    m_pool は費用を引く前の本数(費用で m が減っても候補の幅は変えない)。"""
    cand = top_m(p["ls_cut"], pool, min(2 * m_pool, len(pool)), rng)
    s = p["signs"][cand].astype(np.float64)
    overlap = np.abs(s @ s.T) / s.shape[1]
    chosen = [0]
    for _ in range(min(m, len(cand)) - 1):
        closest = overlap[:, chosen].max(axis=1)
        closest[chosen] = np.inf
        chosen.append(int(np.argmin(closest)))
    return cand[chosen]


def overheads(p: dict, cost: dict | None, K: int) -> dict:
    """方法ごとの、K 候補に対する余分な費用(秒)。cost が None なら 0(選別の効果の上限)。"""
    if cost is None:
        free = {s: 0.0 for s in ["cim_random", "cim_cut", "cim_ls", "cim_div", "cim_trial", *MODELS]}
        return {**free, "random_init": -K * p["t_cim"]}
    ls = K * cost["t_ls"] / PARALLEL
    local = K * cost["t_local"] / PARALLEL
    eig = K * cost["t_eig"] / PARALLEL + cost["t_eig_once"]
    return {"cim_random": 0.0, "cim_cut": 0.0, "cim_ls": ls, "cim_div": ls, "cim_trial": K * p["t_trial"],
            "random_init": -K * p["t_cim"],  # CIM の時間も SA に回す
            "model_all": ls + local + eig, "model_local": ls + local, "model_eig": ls + eig}


def simulate(p: dict, rand_cuts: np.ndarray, scores: dict, cost: dict | None, K: int, m: int, rng) -> dict:
    """戦略ごとに MC 回の「最良値」を返す。"""
    total = K * p["t_cim"] + m * p["t_sa"]
    over = overheads(p, cost, K)
    # 浮動小数の丸めで 1 本少なくならないよう、割った後に小さな余裕を足してから切り捨てる
    budgets = {s: max(int(np.floor((m * p["t_sa"] - o) / p["t_sa"] + 1e-9)), 0) for s, o in over.items()}
    budgets["cim_random"] = budgets["cim_cut"] = m
    m_feat, m_trial, m_rand = budgets["cim_ls"], budgets["cim_trial"], budgets["random_init"]
    if m_rand > len(rand_cuts):
        # 復元抽出では最良値を過小に見積もるので、足りなければ止める
        raise ValueError(f"{p['name']}: random-init needs {m_rand} runs but only {len(rand_cuts)} exist")
    k_all, r = p["cuts"].shape
    out = {s: np.empty(MC) for s in budgets}
    for i in range(MC):
        pool = rng.choice(k_all, size=K, replace=False)
        picks = {
            "cim_random": rng.choice(pool, size=m, replace=False),
            "cim_cut": top_m(p["init_cut"], pool, m, rng),
            "cim_ls": top_m(p["ls_cut"] + 1e-6 * p["init_cut"], pool, m_feat, rng),
            "cim_div": diverse_m(p, pool, budgets["cim_div"], m, rng) if budgets["cim_div"] > 0 else pool[:0],
            "cim_trial": top_m(p["feats"]["trial_sa_mean"], pool, m_trial, rng)
            if "trial_sa_mean" in p["feats"] else pool[:0],
            **{k: top_m(v, pool, budgets[k], rng) for k, v in scores.items()},
        }
        for s, idx in picks.items():
            if len(idx) == 0:
                out[s][i] = np.nan
                continue
            draws = p["cuts"][idx, rng.integers(0, r, len(idx))]
            out[s][i] = draws.max()
        out["random_init"][i] = rng.choice(rand_cuts, size=m_rand, replace=False).max() if m_rand > 0 else np.nan
    return {"runs_per_strategy": budgets, "best": out, "total_time_s": total}


def holm(pvals: dict) -> dict:
    """Holm 補正。pvals は {名前: p}。"""
    order = sorted(pvals, key=pvals.get)
    out, running = {}, 0.0
    for rank, k in enumerate(order):
        running = max(running, pvals[k] * (len(order) - rank))
        out[k] = float(min(1.0, running))
    return out


def primary_tests(per_problem: dict) -> dict:
    """PRIMARY の各比較を、問題を単位にした片側 Wilcoxon 符号順位検定 + Holm で判定する。"""
    res = {}
    for name, a, b in PRIMARY:
        d = np.array([v["strategies"][a]["mean_best"] - v["strategies"][b]["mean_best"] for v in per_problem.values()])
        n_all = len(d)
        d = d[np.isfinite(d)]
        if len(d) < n_all:
            print(f"warning: {name} uses {len(d)} of {n_all} problems (others have no runs)", flush=True)
        p_val = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        res[name] = {"a": a, "b": b, "diffs": d.tolist(), "n_used": int(len(d)), "wins": int((d > 0).sum()),
                     "losses": int((d < 0).sum()), "p_one_sided": p_val}
    for name, hp in holm({k: v["p_one_sided"] for k, v in res.items()}).items():
        res[name]["holm_p"] = hp
        res[name]["supported"] = hp < 0.05
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, nargs="+", required=True)
    ap.add_argument("--random", type=Path, nargs="+", required=True)
    ap.add_argument("--costs", type=Path, required=True, help="selection_costs.py の costs.json")
    ap.add_argument("--free-costs", action="store_true", help="LS・指標・試し打ちの費用を 0 にする")
    ap.add_argument("--timing", type=Path, default=None,
                    help="pipeline_timing.py の timing.json(省略時はステップ 3 の実行で記録した時間)")
    ap.add_argument("--t-start", type=float, required=True)
    ap.add_argument("--sweeps", type=int, required=True)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--exclude", nargs="*", default=[], help="除く問題(候補が多様でない問題など)")
    ap.add_argument("--tag", default="main")
    args = ap.parse_args()
    if not args.tag.replace("_", "").isalnum():
        ap.error("tag must be alphanumeric/underscore")

    problems = []
    for d in args.runs:
        p = load_problem(d.resolve(), args.sweeps)
        if abs(p["t_start"] - args.t_start) < 1e-9 and p["name"] not in args.exclude:
            problems.append(p)
    rand = {}
    for d in args.random:
        meta = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        z = np.load(d / "raw.npz")
        key = f"cuts_S{args.sweeps}"
        if abs(meta["t_start"] - RANDOM_T_START) > 1e-9:
            continue  # ランダム初期化は開始温度 2.0 に固定(計画どおり)
        if key in z.files and len(z[key]) > len(rand.get(meta["dataset"], [])):
            rand[meta["dataset"]] = z[key]  # 同じ問題が複数あれば本数の多いもの
    names = [p["name"] for p in problems]
    if len(set(names)) != len(names):
        ap.error(f"duplicate problems in --runs for t_start={args.t_start}: {names}")
    problems = [p for p in problems if p["name"] in rand]
    costs = json.loads(args.costs.read_text(encoding="utf-8"))
    if args.timing is not None:
        timing = json.loads(args.timing.read_text(encoding="utf-8"))
        missing = [p["name"] for p in problems if p["name"] not in timing]
        if missing:
            ap.error(f"no timing for {missing}")
        problems = [with_timing(p, timing) for p in problems]
    missing = [p["name"] for p in problems if p["name"] not in costs]
    if missing:
        ap.error(f"no cost measurement for {missing}")
    if len(problems) < 2:
        ap.error("need at least two problems with random-init data")
    scores = {name: lopo_scores(problems, feats) for name, feats in MODELS.items()}

    rng = np.random.default_rng(MC_SEED)
    per_problem = {}
    for i, p in enumerate(problems):
        cost = None if args.free_costs else costs[p["name"]]
        sim = simulate(p, rand[p["name"]], {k: v[i] for k, v in scores.items()}, cost, args.pool, args.m, rng)
        target = np.quantile(p["cuts"], TARGET_QUANTILE)
        per_problem[p["name"]] = {
            "target": float(target), "bks": p["bks"], "total_time_s": sim["total_time_s"],
            "runs_per_strategy": sim["runs_per_strategy"],
            "strategies": {s: {"mean_best": float(np.nanmean(v)),
                               "se": float(np.nanstd(v) / np.sqrt(np.isfinite(v).sum())),
                               "p_target": float(np.nanmean(np.where(np.isnan(v), np.nan, v >= target))),
                               "gap_to_bks_pct": float((p["bks"] - np.nanmean(v)) / p["bks"] * 100)}
                           for s, v in sim["best"].items()},
        }
        row = per_problem[p["name"]]["strategies"]
        print(f"{p['name']:>4} " + " ".join(f"{s}={v['mean_best']:.1f}" for s, v in row.items()), flush=True)

    strategies = list(next(iter(per_problem.values()))["strategies"])
    compare = {}
    for s in strategies:
        d = np.array([per_problem[k]["strategies"][s]["mean_best"] - per_problem[k]["strategies"][BASELINE]["mean_best"]
                      for k in per_problem])
        ok = np.isfinite(d)
        p_val = float(wilcoxon(d[ok]).pvalue) if s != BASELINE and ok.sum() >= 5 and np.any(d[ok] != 0) else None
        compare[s] = {"mean_diff_vs_baseline": float(np.nanmean(d)), "wins": int((d > 0).sum()),
                      "losses": int((d < 0).sum()), "wilcoxon_p": p_val}
    for s, hp in holm({s: v["wilcoxon_p"] for s, v in compare.items() if v["wilcoxon_p"] is not None}).items():
        compare[s]["holm_p"] = hp
    primary = primary_tests(per_problem)
    kind_root = get_kind_root()
    free = "_free" if args.free_costs else ""
    out_dir = kind_root / f"v{next_version(kind_root)}_{args.tag}_T{args.t_start:g}_S{args.sweeps}_K{args.pool}_m{args.m}{free}".replace(".", "p")
    out_dir.mkdir(parents=True)
    result = {"t_start": args.t_start, "sweeps": args.sweeps, "pool": args.pool, "m": args.m,
              "mc": MC, "baseline": BASELINE, "models": MODELS, "ridge": RIDGE,
              "target_quantile": TARGET_QUANTILE, "excluded": args.exclude,
              "costs": str(args.costs), "free_costs": args.free_costs, "parallel": PARALLEL,
              "timing": str(args.timing) if args.timing else None,
              "times": {p["name"]: {"t_cim": p["t_cim"], "t_sa": p["t_sa"], "t_trial": p["t_trial"]} for p in problems},
              "per_problem": per_problem, "compare_vs_baseline": compare, "primary": primary}
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for s, v in compare.items():
        print(f"{s:>12}: diff={v['mean_diff_vs_baseline']:+.2f} W/L={v['wins']}/{v['losses']} "
              f"holm_p={v.get('holm_p')}")
    for k, v in primary.items():
        print(f"{k} {v['a']} - {v['b']}: W/L={v['wins']}/{v['losses']} p={v['p_one_sided']:.4f} "
              f"holm={v['holm_p']:.4f} supported={v['supported']}")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
