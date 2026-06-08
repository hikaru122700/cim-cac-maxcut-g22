---
name: run-cim-cac-maxcut-g22
description: Build, run, and verify the cim-cac-maxcut-g22 Ising-machine MAX-CUT solvers (CIM / CAC / SB / SA) and the dated parallel-tempering (PT) experiment modules. Use to run or smoke-test a solver, launch a benchmark, reproduce a CIM/CAC run on a G-Set graph, drive the PT experiments, or check the best cut after a code change.
---

# Run cim-cac-maxcut-g22

A Python + **Numba-JIT** research codebase that solves Stanford **G-Set MAX-CUT**
with four Ising-machine algorithms (CIM / CAC / SB / SA). There is **no GUI and
no server** — the "app" is a set of solvers and dated experiment modules that
run a simulation and write numbers + figures to `results/`. You drive it
programmatically: the **smoke driver** calls the CIM solver and independently
re-verifies the returned cut; the **experiment modules** run full benchmarks.

The central benchmark is **G22** (N=2000, K=19990, known-best **13359**).
Paths below are relative to the `cim-cac-maxcut-g22/` project root.
The repo's Python lives in a local venv at `.venv/Scripts/python.exe` (Windows).

## Prerequisites / Setup

Dependencies are managed by [uv](https://docs.astral.sh/uv/) (Python ≥3.13;
`numpy`, `scipy`, `numba`, `matplotlib`, `wandb`, `pymupdf`, …).

```bash
uv sync          # creates/populates .venv  (needs network access)
```

If `.venv` already exists, **do not** rely on a bare `uv run` (it re-syncs and
fails on restricted networks — see Gotchas). Use `--no-sync` or the venv python
directly, both shown below.

## Run — agent path (smoke driver)

The driver (`.claude/skills/run-cim-cac-maxcut-g22/driver.py`) runs the actual
CIM solver on a G-Set graph **without wandb/plotting**, then re-counts the cut
independently (`compute_cut_from_edges`) and checks it matches the kernel and
clears a ratio threshold. **Exit 0 = solver ran and verified.**

```bash
uv run --no-sync python .claude/skills/run-cim-cac-maxcut-g22/driver.py
```

Verified output (G22, this container):

```
[run] CIM G22 N=2000 K=19990  rounds=1500 trials=16
[SOLUTION OK] len=2000, 集合A=1001頂点, 集合B=999頂点
[result] best_cut=13308  independent_recut=13308  ratio_to_known_best=0.9962  wall=0.4s
[PASS] CIM solver verified
```

Equivalent, bypassing uv entirely (use the repo venv directly):

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe .claude/skills/run-cim-cac-maxcut-g22/driver.py
```

Other graphs / sizes (any file in `input/`: G1 G14 G15 G22 G23 G32 G39 G55 G70 G77 G81):

```bash
uv run --no-sync python .claude/skills/run-cim-cac-maxcut-g22/driver.py --graph G15 --trials 8 --min-ratio 0.95
```

The first run on a cold machine spends a few seconds in **Numba JIT compile**
before the wall time shown above.

## Run — full experiments (the PT modules)

Each experiment is a **self-contained, date-named module** run as a script (the
filename starts with a digit, so it is **not** importable — run it directly).
It writes figures + `summary.json` to `results/YYYY-MM-DD/<kind>/v{N}_<desc>/`.
Always set UTF-8 first (these modules print Japanese):

```bash
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 .venv/Scripts/python.exe "modules/2026-06-06_CIM_PT_v3.py"
```

Verified tail of that run (this session):

```
================================================================================================
Method                        Ntrial       Mean       Best      Worst      Std   Time[s]
ランプCIM                           300    13276.7    13326.0    13220.0     21.0      3.97
CIM+PT v2(固定ポンプ反転)               100    13183.0    13271.0    13137.0     23.7      3.89
v3 swap無(ランプ集団のみ)                100    13291.9    13337.0    13262.0     17.0      8.57
CIM+PT v3(各レプリカ・ランプ)             100    13294.3    13337.0    13257.0     15.9      3.83
[output] dir=results\2026-06-08\cim_pt_v3\v1_rounds1500_swap10_perreplica_ramp
```

Companion modules: `modules/2026-06-06_CIM_PT_v2.py` (reversed-ladder PT),
`modules/2026-06-08_CIM_PT_v3_swap_ablation.py` (swap on/off ablation).
**When comparing PT vs a single solver, always include the swap-OFF control** —
the v3 ablation showed the gain over ramp-CIM is ~87 % multi-start, not the swap.

## Run — human path (with wandb)

`modules/CIM.py`'s `main()` opens a **wandb run** and logs per-round metrics
(needs wandb login/offline). Useful for one-off inspection, not for headless use:

```bash
uv run --no-sync python -m modules.CIM        # → wandb.init, single 1500-round run
```

The README lists more `-m` entry points (`modules.CAC`, `modules.SA`,
`scripts.benchmarks.compare`, `scripts.tuning.tune_cac`, …); they follow the
same "run from project root" rule and write to `results/<date>/`.

## Test

```bash
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest -q       # 63 passed in ~0.4s
```

## Gotchas (battle scars)

- **`uv run` re-syncs and can fail offline.** A bare `uv run …` re-resolves the
  lockfile; in this container it died with
  `Failed to download scikit-learn==1.9.0 … invalid peer certificate: UnknownIssuer`.
  When `.venv` is already populated, use **`uv run --no-sync`** or call
  `.venv/Scripts/python.exe` directly.
- **Windows cp932 console vs Japanese output.** `modules/verify.py` and every
  experiment module print Japanese (`[SOLUTION OK] 集合A=…頂点`). Under the
  default Windows codepage you get `UnicodeEncodeError: 'cp932' codec can't
  encode character`. Fix: set **`PYTHONIOENCODING=utf-8`** (and `PYTHONUTF8=1`),
  or `sys.stdout.reconfigure(encoding="utf-8")` — `driver.py` does the latter so
  it needs no env.
- **Numba cache + importlib dynamic load → `ModuleNotFoundError: No module named '<dynamic>'`.**
  Dated modules have digit-prefixed filenames that can't be `import`ed by name.
  Loading one via `importlib.util.spec_from_file_location` **and then calling its
  `@njit(cache=True)` kernel** fails on cache load. Fix: keep each experiment
  module **self-contained** (copy the kernel + pure helpers in; only import
  pure-Python helpers across modules) and run dated modules as scripts.
- **`python -m modules.CIM` starts wandb.** For programmatic/headless use call
  `simulate_cim_batch(...)` directly (as `driver.py` does) — no wandb, no plots.
- **Equal-compute convention.** Benchmarks give ramp-CIM `NR×NT` trials vs PT's
  `NT` trials × `NR` replicas so wall-for-wall comparisons are fair. Keep this
  when adding a method to a comparison.
- **Results are versioned, never overwritten.** Output lands in
  `results/YYYY-MM-DD/<kind>/v{N}_<desc>/` with an auto-incrementing `v{N}`.
  Re-running makes a new `v` folder (see `CLAUDE.md`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `invalid peer certificate: UnknownIssuer` from `uv sync`/`uv run` | venv already populated → `uv run --no-sync …` or `.venv/Scripts/python.exe …` |
| `UnicodeEncodeError: 'cp932' codec can't encode …` | prefix `PYTHONIOENCODING=utf-8 PYTHONUTF8=1` |
| `ModuleNotFoundError: No module named '<dynamic>'` | numba cache + importlib of a dated module — make the module self-contained, run as a script |
| `[FAIL] graph not found` from driver | `--graph` must match a file in `input/` (e.g. `G22`, not `G22.txt`) |
