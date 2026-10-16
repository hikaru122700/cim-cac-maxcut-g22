"""modules/GA.py の _tabu_search_one と同じ処理に、選んだ手の種類を数える計測を加えた診断用コピー。

2026-10-16、G55/G70 で TS が悪化方向の手を一度も選ばないことの確認に使用。

使い方(プロジェクトルートから):
  uv run python scripts/utils/ts_move_probe.py G70 G55 G22
"""
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from modules.GA import _init_gains, _apply_flip, _TENURE_PATTERN, _build_csr
from scripts.benchmarks.algo_registry import load_context, run_cim, PARAMS


@njit(cache=False)
def ts_probe(n, indptr, indices, data, s, ts_iters, cr, gamma_pert, alpha_tenure, seed,
             random_ties):
    np.random.seed(seed)
    gains, cut = _init_gains(n, indptr, indices, data, s)
    best_cut = cut
    no_improve = 0
    tabu_until = np.zeros(n, dtype=np.int64)
    n_zero = 0
    n_neg = 0
    n_pos = 0
    n_pert = 0
    kick_max = np.full(2000, -1.0)  # 各摂動後〜次の摂動までに到達した最大 cut
    kick_after = np.full(2000, -1.0)  # 摂動直後の cut
    for it in range(ts_iters):
        tenure = alpha_tenure * _TENURE_PATTERN[(it // 100) % 15]
        best_v = -1
        best_g = -1.0e18
        asp_v = -1
        asp_g = -1.0e18
        n_ties = 0
        for v in range(n):
            g = gains[v]
            if it >= tabu_until[v]:
                if g > best_g:
                    best_g = g
                    best_v = v
                    n_ties = 1
                elif random_ties and g == best_g:
                    # 同点は等確率に選ぶ(reservoir sampling)
                    n_ties += 1
                    if np.random.randint(0, n_ties) == 0:
                        best_v = v
            else:
                if cut + g > best_cut and g > asp_g:
                    asp_g = g
                    asp_v = v
        if asp_v >= 0 and asp_g > best_g:
            mv = asp_v
            mg = asp_g
        else:
            mv = best_v
            mg = best_g
        if mg > 1e-9:
            n_pos += 1
        elif mg < -1e-9:
            n_neg += 1
        else:
            n_zero += 1
        _apply_flip(mv, s, gains, indptr, indices, data)
        cut += mg
        tabu_until[mv] = it + tenure
        if n_pert > 0 and n_pert <= 2000 and cut > kick_max[n_pert - 1]:
            kick_max[n_pert - 1] = cut
        if cut > best_cut:
            best_cut = cut
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cr:
                for _ in range(gamma_pert):
                    rv = np.random.randint(0, n)
                    _apply_flip(rv, s, gains, indptr, indices, data)
                gains_new, cut = _init_gains(n, indptr, indices, data, s)
                for i in range(n):
                    gains[i] = gains_new[i]
                no_improve = 0
                if n_pert < 2000:
                    kick_after[n_pert] = cut
                n_pert += 1
    return best_cut, n_pos, n_zero, n_neg, n_pert, kick_after, kick_max


EXPERIMENT_KIND = "ts_move_probe"


def out_dir_for(names):
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    vs = [int(p.name.split("_", 1)[0][1:]) for p in root.iterdir()
          if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    d = root / f"v{max(vs, default=0) + 1}_{'_'.join(names)}_it200k"
    d.mkdir()
    return d


def probe(name, log, random_ties, gamma_override):
    ctx = load_context(name)
    p = PARAMS[name]["GA"]
    gamma = gamma_override if gamma_override else max(50, int(0.05 * ctx.n))
    ea = np.array([e[0] for e in ctx.edges], dtype=np.int64)
    eb = np.array([e[1] for e in ctx.edges], dtype=np.int64)
    ip, ix, dt = _build_csr(ctx.n, ea, eb, np.ones(len(ea)))
    _, s = run_cim(ctx, 600, 2, np.arange(2))
    starts = {"cim": np.asarray(s[0]).astype(np.int8),
              "rand": (np.random.default_rng(0).random(ctx.n) < 0.5).astype(np.int8)}
    log(f"{name} params {p} gamma_pert {gamma} random_ties {random_ties}")
    for label, s0 in starts.items():
        r = ts_probe(ctx.n, ip, ix, dt, s0.copy(), 200_000, p["cr"], gamma,
                     p["alpha_tenure"], 0, random_ties)
        best, npos, nzero, nneg, npert, after, kmax = r
        k = min(npert, 2000)
        log(f"{label}: best={best:.0f} moves +:{npos} 0:{nzero} -:{nneg} perturbations={npert}")
        if k:
            log(f"   cut right after kick: mean={after[:k].mean():.1f}  "
                  f"max reached before next kick: mean={kmax[:k].mean():.1f} max={kmax[:k].max():.0f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["G70"])
    ap.add_argument("--random-ties", action="store_true",
                    help="Wu & Hao (2012) と同じく同点の手を等確率に選ぶ")
    ap.add_argument("--gamma", type=int, default=0, help="摂動の強さ(0 なら GA.py の既定)")
    args = ap.parse_args()
    names = args.names
    tag = names + (["randties"] if args.random_ties else []) + (
        [f"gamma{args.gamma}"] if args.gamma else [])
    out = out_dir_for(tag)

    def log(msg):
        print(msg, flush=True)
        with (out / "run.log").open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    for name in names:
        probe(name, log, args.random_ties, args.gamma)
    log(f"saved: {out}")


if __name__ == "__main__":
    main()
