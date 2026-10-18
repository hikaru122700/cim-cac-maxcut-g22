"""Phase 0-1 パイロット: 同じ cut の CIM 解でも、後段 SA での価値はばらつくか。

論文化計画(docs/20261016/paper_plan_B_v1.md)の go/no-go 判定用。

手順:
  1. 独立 seed の CIM を K 回走らせ、各 run の最良状態を候補とする。
  2. 各候補を初期解に、warm SA(予算 B)を R 回、互いに異なる乱数で実行する。
  3. 候補ごとのラベル(R 回の平均最良 cut)が、初期 cut だけで説明できるかを見る。

判定(実行前に固定):
  R 回を偶数回・奇数回の 2 群に分け、各群の平均ラベルを「初期 cut」と
  「1-flip 貪欲局所探索後の cut」で回帰した残差どうしの相関を求める。
  乱数ノイズは 2 群で独立なので、この相関は「cut では説明できない、候補固有の
  価値」の信頼性を表す。CIM 出力は 1-flip 局所最適とは限らずすぐ取れる改善が
  残るため、それで説明できる差(自明な差)も統制する。
  Spearman-Brown 補正後の値がいずれかの予算で 0.3 以上、かつ候補ブートストラップ
  95% CI の下限が 0 を超えれば go。

使い方(プロジェクトルートから):
  uv run python scripts/benchmarks/seed_value_pilot.py --dataset G22
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "16")

import matplotlib
import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from modules.SA import simulate_sa_warm
from scripts.benchmarks.algo_registry import load_context, run_cim

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False

EXPERIMENT_KIND = "seed_value_pilot"
GO_RELIABILITY = 0.3
SA_SEED_OFFSET = 1_000_000  # CIM seed(0..K-1)と重ならない SA 乱数
TRIAL_SEED_OFFSET = 50_000_000  # 試し打ち SA の乱数(ラベルの SA と重ならない)


def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
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


def build_description(args) -> str:
    parts = [
        args.dataset,
        f"K{args.candidates}",
        f"R{args.repeats}",
        f"cim{args.cim_rounds}",
        f"T{args.t_start:g}",
    ]
    if args.cim_seed_start:
        parts.append(f"seed{args.cim_seed_start}")
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts).replace(".", "p")


# ------------------------------------------------------------------
#  候補の構造指標
# ------------------------------------------------------------------
def adjacency(ctx) -> csr_matrix:
    ip, ix, dt = ctx._csr
    rows = np.repeat(np.arange(ctx.n), np.diff(ip))
    return csr_matrix((dt, (rows, ix)), shape=(ctx.n, ctx.n))


def flip_gains(A: csr_matrix, signs_pm: np.ndarray) -> np.ndarray:
    """各頂点を反転したときの cut の増分 g_i = Σ_j w_ij s_i s_j。shape (K, n)。"""
    s = signs_pm.astype(np.float64)
    return s * (A @ s.T).T


@njit(cache=True)
def _greedy_local_opt(n, ip, ix, dt, s):
    """最良改善の 1-flip 貪欲探索を局所最適まで。s(±1)をその場で更新し cut 増分を返す。"""
    g = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for k in range(ip[i], ip[i + 1]):
            acc += dt[k] * s[i] * s[ix[k]]
        g[i] = acc
    total = 0.0
    while True:
        v = np.argmax(g)
        if g[v] <= 1e-9:
            return total
        total += g[v]
        s[v] = -s[v]
        g[v] = -g[v]
        for k in range(ip[v], ip[v + 1]):
            u = ix[k]
            g[u] += 2.0 * dt[k] * s[u] * s[v]


def greedy_local_opt(ctx, signs_pm: np.ndarray) -> np.ndarray:
    ip, ix, dt = ctx._csr
    dt = np.asarray(dt, dtype=np.float64)
    out = signs_pm.astype(np.float64).copy()
    for k in range(len(out)):
        _greedy_local_opt(ctx.n, ip, ix, dt, out[k])
    return np.where(out > 0, 1, -1).astype(np.int8)


def candidate_features(A: csr_matrix, signs_pm: np.ndarray) -> dict[str, np.ndarray]:
    """安価な構造指標(Phase 1 の予備)。全て O(辺数)。"""
    gains = flip_gains(A, signs_pm)
    coo = A.tocoo()
    largest_zero = np.zeros(len(signs_pm))
    for k, g in enumerate(gains):
        zero = np.abs(g) < 1e-9
        if not zero.any():
            continue
        keep = zero[coo.row] & zero[coo.col]
        sub = csr_matrix(
            (np.ones(int(keep.sum())), (coo.row[keep], coo.col[keep])), shape=A.shape
        )
        _, labels = connected_components(sub, directed=False)
        largest_zero[k] = np.bincount(labels[zero]).max()
    return {
        "n_pos_gain": (gains > 1e-9).sum(axis=1).astype(float),
        "n_zero_gain": (np.abs(gains) < 1e-9).sum(axis=1).astype(float),
        "sum_pos_gain": np.clip(gains, 0, None).sum(axis=1),
        "largest_zero_cluster": largest_zero,
        "n_gain_minus2_to_0": ((gains >= -2 - 1e-9) & (gains <= 1e-9)).sum(axis=1).astype(float),
    }


EIG_K = 2  # 固有ベクトル指標で使う本数(2026-10-18 の事前登録で固定)


def low_eigvecs(A: csr_matrix, k: int = EIG_K) -> tuple[np.ndarray, np.ndarray]:
    """A の最小固有値側 k 本(CIM の結合 J = c·A, c<0 の最大側)。昇順に並べて返す。"""
    v0 = np.random.default_rng(0).standard_normal(A.shape[0])
    vals, vecs = eigsh(A.astype(np.float64), k=k, which="SA", v0=v0)
    order = np.argsort(vals)
    return vals[order], vecs[:, order]


def eig_alignment(signs_pm: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """各解と sign(u_j) の重なり |<s, sign(u_j)>|/n(全反転を同一視)。shape (K, k)。"""
    s = signs_pm.astype(np.float64)
    return np.abs(s @ np.sign(vecs)) / s.shape[1]


def eig_features(signs_pm: np.ndarray, vecs: np.ndarray) -> dict[str, np.ndarray]:
    al = eig_alignment(signs_pm, vecs)
    return {"eig_a1": al[:, 0], "eig_a2": al[:, 1], "eig_diff": al[:, 0] - al[:, 1]}


def aligned_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """全反転を同一視した正規化ハミング距離(行ごと)。"""
    d = (a != b).mean(axis=1)
    return np.minimum(d, 1 - d)


# ------------------------------------------------------------------
#  解析
# ------------------------------------------------------------------
def residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """x は (K,) または (K, p) の共変量。定数項付き OLS の残差。"""
    X = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def split_half(cuts_cr: np.ndarray, cov: np.ndarray) -> tuple[float, float, float]:
    """偶数回・奇数回で 2 群に分けた、共変量 cov で回帰した残差の相関と Spearman-Brown 補正値。"""
    a = cuts_cr[:, 0::2].mean(axis=1)
    b = cuts_cr[:, 1::2].mean(axis=1)
    raw = float(np.corrcoef(a, b)[0, 1])
    rho = float(np.corrcoef(residual(a, cov), residual(b, cov))[0, 1])
    sb = 2 * rho / (1 + rho) if rho > -1 else float("nan")
    return raw, rho, sb


def bootstrap_split_half(cuts_cr, cov, seed, samples=2000):
    rng = np.random.default_rng(seed)
    k = len(cuts_cr)
    out = np.empty(samples)
    for i in range(samples):
        idx = rng.integers(0, k, size=k)
        out[i] = split_half(cuts_cr[idx], cov[idx])[2]
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def analyze_budget(cuts_cr, init_cut, ls_cut, features, seed):
    k, r = cuts_cr.shape
    ybar = cuts_cr.mean(axis=1)
    noise_var = float(cuts_cr.var(axis=1, ddof=1).mean() / r)
    total_var = float(ybar.var(ddof=1))
    true_var = max(total_var - noise_var, 0.0)

    both = np.column_stack([init_cut, ls_cut])
    resid = residual(ybar, init_cut)
    resid_ls = residual(ybar, both)
    beyond_cut_var = max(float(resid.var(ddof=2)) - noise_var, 0.0)
    beyond_ls_var = max(float(resid_ls.var(ddof=3)) - noise_var, 0.0)
    _, _, sb = split_half(cuts_cr, init_cut)
    lo, hi = bootstrap_split_half(cuts_cr, init_cut, seed)
    raw, _, sb_ls = split_half(cuts_cr, both)
    lo_ls, hi_ls = bootstrap_split_half(cuts_cr, both, seed + 1)

    # 同じ初期 cut の候補が 3 個以上ある水準での、ラベルの層内ばらつき
    levels = []
    for c in np.unique(init_cut):
        m = init_cut == c
        if m.sum() >= 3:
            levels.append(
                {
                    "init_cut": float(c),
                    "count": int(m.sum()),
                    "label_min": float(ybar[m].min()),
                    "label_max": float(ybar[m].max()),
                    "label_std": float(ybar[m].std(ddof=1)),
                }
            )

    feats = {}
    for name, f in features.items():
        if np.ptp(f) == 0:
            feats[name] = None
            continue
        rho_f, p_f = spearmanr(f, resid)
        rho_g, p_g = spearmanr(f, resid_ls)
        feats[name] = {
            "spearman_vs_cut_residual": float(rho_f),
            "p": float(p_f),
            "spearman_vs_cut_ls_residual": float(rho_g),
            "p_ls": float(p_g),
        }

    return {
        "label_mean": float(ybar.mean()),
        "label_std_between_candidates": float(np.sqrt(total_var)),
        "noise_se_per_candidate": float(np.sqrt(noise_var)),
        "true_std_between_candidates": float(np.sqrt(true_var)),
        "beyond_cut_true_std": float(np.sqrt(beyond_cut_var)),
        "beyond_cut_ls_true_std": float(np.sqrt(beyond_ls_var)),
        "spearman_init_cut_vs_label": float(spearmanr(init_cut, ybar)[0]),
        "spearman_ls_cut_vs_label": float(spearmanr(ls_cut, ybar)[0]),
        "split_half_raw": raw,
        "split_half_cut_resid_sb": sb,
        "split_half_cut_resid_sb_ci95": [lo, hi],
        "split_half_cut_ls_resid_sb": sb_ls,
        "split_half_cut_ls_resid_sb_ci95": [lo_ls, hi_ls],
        "go": bool(sb_ls >= GO_RELIABILITY and lo_ls > 0),
        "same_cut_levels": levels,
        "features": feats,
    }


def plot_overview(init_cut, ls_cut, results, raw, budgets, bks, out_dir, title):
    nb = len(budgets)
    fig, axes = plt.subplots(2, nb, figsize=(4.4 * nb, 8.4), dpi=150, squeeze=False)
    both = np.column_stack([init_cut, ls_cut])
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, len(init_cut))
    for j, b in enumerate(budgets):
        cuts_cr = raw[b]
        ybar = cuts_cr.mean(axis=1)
        se = cuts_cr.std(axis=1, ddof=1) / np.sqrt(cuts_cr.shape[1])
        res = results[str(b)]

        ax = axes[0, j]
        ax.errorbar(
            init_cut + jitter, ybar, yerr=1.96 * se, fmt="o", ms=3,
            color="#1f77b4", ecolor="#9ab8d6", elinewidth=0.8, alpha=0.8,
        )
        ax.set_xlabel("CIM 初期解の cut")
        ax.set_ylabel(f"SA 後の平均最良 cut(R={cuts_cr.shape[1]})")
        ax.set_title(f"予算 {b:,} 反転提案\nSpearman={res['spearman_init_cut_vs_label']:.2f}")
        ax.grid(True, ls=":", alpha=0.4)

        ax = axes[1, j]
        a = cuts_cr[:, 0::2].mean(axis=1)
        c = cuts_cr[:, 1::2].mean(axis=1)
        ax.scatter(residual(a, both), residual(c, both), s=8, color="#d62728", alpha=0.7)
        lo, hi = res["split_half_cut_ls_resid_sb_ci95"]
        ax.set_xlabel("偶数回の平均 − (cut, 貪欲LS後cut) による予測")
        ax.set_ylabel("奇数回の平均 − 同じ予測")
        ax.set_title(
            f"残差の信頼性 {res['split_half_cut_ls_resid_sb']:.2f} [{lo:.2f}, {hi:.2f}]"
            f"\n判定: {'go' if res['go'] else 'no-go'}"
        )
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.grid(True, ls=":", alpha=0.4)

    for ax in axes.flat:
        ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.suptitle(f"{title}(BKS={bks})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "overview.png", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="G22")
    ap.add_argument("--candidates", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--cim-rounds", type=int, default=600)
    ap.add_argument("--budgets", type=int, nargs="+", default=[100_000, 1_000_000, 10_000_000])
    ap.add_argument("--t-start", type=float, default=0.5)
    ap.add_argument("--t-end", type=float, default=0.001)
    ap.add_argument("--cim-seed-start", type=int, default=0)
    ap.add_argument("--sa-seed-offset", type=int, default=SA_SEED_OFFSET)
    ap.add_argument("--trial-runs", type=int, default=0,
                    help="試し打ち(短い SA)の回数。0 なら計算しない")
    ap.add_argument("--trial-sweeps", type=int, default=50)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    assert args.repeats % 2 == 0, "split-half のため R は偶数"

    kind_root = get_kind_root()
    out_dir = kind_root / f"v{next_version(kind_root)}_{build_description(args)}"
    out_dir.mkdir(parents=True)
    log_path = out_dir / "run.log"

    def log(msg):
        print(msg, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    ctx = load_context(args.dataset)
    A = adjacency(ctx)
    K, R = args.candidates, args.repeats

    t0 = time.perf_counter()
    cim_seeds = args.cim_seed_start + np.arange(K, dtype=np.int64)
    _, cim_signs = run_cim(ctx, args.cim_rounds, K, cim_seeds)
    cim_time = time.perf_counter() - t0
    signs_pm = np.where(np.asarray(cim_signs) > 0, 1, -1).astype(np.int8)
    init_cut = ctx.score(signs_pm).astype(float)
    log(
        f"{args.dataset} CIM K={K} rounds={args.cim_rounds}: {cim_time:.1f}s "
        f"cut mean={init_cut.mean():.1f} min={init_cut.min():.0f} max={init_cut.max():.0f} "
        f"distinct={len(np.unique(init_cut))}"
    )

    ls_signs = greedy_local_opt(ctx, signs_pm)
    ls_cut = ctx.score(ls_signs).astype(float)
    log(
        f"greedy LS: cut mean={ls_cut.mean():.1f} (+{(ls_cut - init_cut).mean():.1f}) "
        f"corr(init, ls)={np.corrcoef(init_cut, ls_cut)[0, 1]:.3f}"
    )

    t0 = time.perf_counter()
    features = candidate_features(A, signs_pm)
    features["ls_gain"] = ls_cut - init_cut
    for name, f in candidate_features(A, ls_signs).items():
        if name in ("n_zero_gain", "largest_zero_cluster", "n_gain_minus2_to_0"):
            features[f"ls_{name}"] = f
    eig_vals, eig_vecs = low_eigvecs(A)
    features.update(eig_features(signs_pm, eig_vecs))
    feat_time = time.perf_counter() - t0
    log(f"features: {feat_time:.2f}s  low eigenvalues of A: {np.round(eig_vals, 4).tolist()}")

    # 候補 c の r 回目は init=候補 c、seed=OFFSET + c*R + r。全て異なる乱数。
    init_rep = np.repeat(signs_pm, R, axis=0)
    sa_seeds = args.sa_seed_offset + np.arange(K * R, dtype=np.int64)
    simulate_sa_warm(ctx.n, ctx.edges, ctx.weights, init_rep[:2], 100,
                     t_start=args.t_start, t_end=args.t_end, seeds=sa_seeds[:2])

    if args.trial_runs > 0:
        t0 = time.perf_counter()
        tr_init = np.repeat(signs_pm, args.trial_runs, axis=0)
        tr_seeds = TRIAL_SEED_OFFSET + np.arange(len(tr_init), dtype=np.int64)
        _, tr_signs = simulate_sa_warm(
            ctx.n, ctx.edges, ctx.weights, tr_init, args.trial_sweeps * ctx.n,
            t_start=args.t_start, t_end=args.t_end, seeds=tr_seeds,
        )
        tr_cut = ctx.score(tr_signs).reshape(K, args.trial_runs).astype(float)
        features["trial_sa_mean"] = tr_cut.mean(axis=1)
        features["trial_sa_max"] = tr_cut.max(axis=1)
        log(f"trial SA: {args.trial_runs} x {args.trial_sweeps} sweeps, {time.perf_counter() - t0:.1f}s")

    raw, results, out_align = {}, {}, {}
    for b in args.budgets:
        t0 = time.perf_counter()
        _, out_signs = simulate_sa_warm(
            ctx.n, ctx.edges, ctx.weights, init_rep, b,
            t_start=args.t_start, t_end=args.t_end, seeds=sa_seeds,
        )
        sa_time = time.perf_counter() - t0
        cuts_cr = ctx.score(out_signs).reshape(K, R).astype(float)
        dist = aligned_distance(init_rep, out_signs).reshape(K, R)
        raw[b] = cuts_cr
        out_align[b] = eig_alignment(out_signs, eig_vecs).reshape(K, R, EIG_K)
        res = analyze_budget(cuts_cr, init_cut, ls_cut, features, seed=20261016 + b % 997)
        res["mean_distance_from_seed"] = float(dist.mean())
        res["sa_time_s"] = sa_time
        results[str(b)] = res
        lo, hi = res["split_half_cut_ls_resid_sb_ci95"]
        log(
            f"B={b:>10,}: {sa_time:.1f}s label={res['label_mean']:.1f} "
            f"between_sd={res['label_std_between_candidates']:.2f} "
            f"noise_se={res['noise_se_per_candidate']:.2f} "
            f"beyond_cut_sd={res['beyond_cut_true_std']:.2f} "
            f"beyond_cut+LS_sd={res['beyond_cut_ls_true_std']:.2f} "
            f"rel(cut)={res['split_half_cut_resid_sb']:.3f} "
            f"rel(cut+LS)={res['split_half_cut_ls_resid_sb']:.3f} [{lo:.3f}, {hi:.3f}] "
            f"go={res['go']} dist_from_seed={res['mean_distance_from_seed']:.3f}"
        )
        for name, v in res["features"].items():
            if v:
                log(
                    f"    {name:>24}: rho(cut-resid)={v['spearman_vs_cut_residual']:+.3f} "
                    f"rho(cut+LS-resid)={v['spearman_vs_cut_ls_residual']:+.3f} "
                    f"p={v['p_ls']:.3g}"
                )

    summary = {
        "meta": {
            "dataset": args.dataset,
            "bks": ctx.bks,
            "candidates": K,
            "repeats": R,
            "cim_rounds": args.cim_rounds,
            "cim_seeds": [int(cim_seeds[0]), int(cim_seeds[-1])],
            "sa_seed_rule": f"{args.sa_seed_offset} + c*R + r",
            "low_eigenvalues_A": eig_vals.tolist(),
            "trial": {"runs": args.trial_runs, "sweeps": args.trial_sweeps,
                      "seed_offset": TRIAL_SEED_OFFSET},
            "t_start": args.t_start,
            "t_end": args.t_end,
            "budgets": args.budgets,
            "go_rule": (
                "Spearman-Brown corrected split-half correlation of labels residualized "
                f"on (initial cut, greedy 1-flip LS cut) >= {GO_RELIABILITY} and "
                "bootstrap 95% CI lower bound > 0 at any budget"
            ),
            "cim_time_s": cim_time,
            "feature_time_s": feat_time,
        },
        "init_cut": {
            "mean": float(init_cut.mean()),
            "min": float(init_cut.min()),
            "max": float(init_cut.max()),
            "distinct": int(len(np.unique(init_cut))),
        },
        "ls_cut": {
            "mean": float(ls_cut.mean()),
            "min": float(ls_cut.min()),
            "max": float(ls_cut.max()),
        },
        "budgets": results,
        "go_any": any(r["go"] for r in results.values()),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    np.savez_compressed(
        out_dir / "raw.npz",
        init_cut=init_cut,
        ls_cut=ls_cut,
        cim_signs=signs_pm,
        **{f"cuts_B{b}": raw[b] for b in args.budgets},
        **{f"out_eig_align_B{b}": out_align[b] for b in args.budgets},
        eig_vals=eig_vals,
        **{f"feat_{k}": v for k, v in features.items()},
    )
    plot_overview(
        init_cut, ls_cut, results, raw, args.budgets, ctx.bks, out_dir,
        f"{args.dataset}: 同じ初期 cut でも SA 後の価値は違うか(K={K}, R={R})",
    )
    log(f"go_any={summary['go_any']}  saved: {out_dir}")


if __name__ == "__main__":
    main()
