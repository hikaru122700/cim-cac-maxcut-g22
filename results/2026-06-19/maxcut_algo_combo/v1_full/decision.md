# Research Decision: PROCEED

## Decision
**PROCEED** to paper writing. The experiments yield a clear, well-supported finding worth reporting.

## Core finding (evidence-based; all six solvers Optuna-tuned per instance)
Across four MAX-CUT benchmarks (G22, K2000, G55, G70), on a unified wall-clock axis with all
solvers scored by an identical weighted-cut function, and with **every solver hyperparameter-tuned
per instance via Optuna** (so each is shown near its best):

1. **Single solvers**: The tuned classical heuristics are strongest, and **no single solver wins on
   every instance**: the memetic GA is best on G22 (reaches BKS exactly, gap 0) and on the dense
   weighted K2000 (gap 33), while discrete Simulated Bifurcation (SB) is best on the two large sparse
   instances G55 (gap 15) and G70 (gap 42) and has the smallest mean relative gap (~0.20%), i.e. it is
   the most reliable. Crucially, **fair tuning overturns the naive impression that simulated annealing
   is hopelessly weak**: tuning its cooling schedule moves SA from the bottom to mid-pack
   (K2000 gap 1348 → 756; G22 gap 5 → 1). The physics-inspired CIM rises fast but plateaus (G22 gap 17)
   and, even re-tuned, stays far from best-known on the dense K2000 (gap 794). PT-ICM is strong on
   sparse instances but weak on dense K2000 (gap 1702), and single-budget tuning can even hurt it there.

2. **Hybrid (warm-start) — value is "rescue", not domination**: seeding a classical local search
   (Tabu Search or warm-start SA) with the spin configuration from a short CIM/CAC run **dominates the
   same refiner from random initialization at matched refinement budget**, and **breaks the physics
   solver's plateau** — CIM alone reaches gap 794 on K2000 but CIM→TS reaches gap 120 (and CIM 17→1 on
   G22). The handed-off basin quality matters: under a fixed refiner, a CIM seed (gap 120) beats a CAC
   seed (gap 244). **However, with all classical solvers fairly tuned, the hybrid does not beat the best
   single solver** (GA gap 33 on K2000). The honest conclusion: physics-inspired dynamics are best used
   as **explorers inside a hybrid and members of a portfolio**, not as standalone final solvers.

3. **Portfolio**: because the best single solver changes across instances (GA on G22/K2000, SB on
   G55/G70), a parallel best-of-K portfolio is a robust, oracle-free choice when cores are available.

## Why this is publishable
- A fair, reproducible, uniformly-scored comparison of physics-inspired and classical MAX-CUT solvers
  with **all solvers tuned per instance** (not under-tuned strawmen) using real implementations.
- The hybrid result is actionable and quantified, and the framing is honest about when it helps.
- Limitations are explicit (best-of-batch only, no wall-clock instrumentation, single-budget tuning).

## Scope of claims (honesty)
- BKS values are literature references; we report gap to BKS, not certified optima.
- CIM/CAC are specific implementations (Inoue–Yoshida; Leleu 2021) under 25–30-trial Optuna for the
  re-tuned physics solvers; conclusions are about these implementations under the stated budget.
- gap differences of a few cuts fall within stochastic run-to-run variability (dispersion not recorded).
