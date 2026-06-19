# Research Decision: PROCEED

## Decision
**PROCEED** to paper writing. The experiments yield a clear, well-supported finding worth reporting.

## Core finding (evidence-based)
Across four MAX-CUT benchmarks (G22, K2000, G55, G70), on a unified wall-clock axis with all
solvers scored by an identical weighted-cut function:

1. **Single solvers**: Classical heuristics — Simulated Bifurcation (dSB), the new memetic GA,
   and Simulated Annealing — are the most time-efficient and reach within ~0.01% of best-known
   solutions (BKS) in well under one second on G22. The physics-inspired Coherent Ising Machine
   (CIM) rises fast but **plateaus below BKS** (G22 gap 17) and **fails to transfer across
   instances** without per-dataset re-tuning (untuned CIM on dense K2000 reaches only ~4287 of
   BKS 33337; re-tuned, ~32350). Chaotic Amplitude Control (CAC) is strong on unweighted graphs
   but its unweighted internal selection handicaps it on the weighted K2000 instance.

2. **Hybrid (warm-start) is the headline result**: seeding a classical local search
   (Tabu Search or warm-start SA) with the spin configuration produced by a short CIM/CAC run
   **dominates the same refiner from random initialization at matched refinement budget**, and
   **breaks CIM's plateau** — CIM alone tops out at 13342 on G22, but CIM→TS reaches BKS−1
   (13358) in ≈0.22 s. At the smallest refinement budget the warm start improves the mean cut by
   +250 over cold start. The division of labor "physics solver = fast global explorer, classical
   local search = refiner" gives a better efficiency/quality trade-off than either alone.

3. **Portfolio**: no single solver dominates across all four datasets, so a parallel portfolio is
   a robust practical choice when multiple cores are available.

## Why this is publishable
- It is a fair, reproducible, wall-clock comparison of physics-inspired and classical MAX-CUT
  solvers under per-dataset hyperparameter tuning, using real (not toy) implementations.
- The hybrid result is actionable and quantified, not anecdotal.
- Limitations are explicit (CAC weighted-selection; CIM per-instance scaling), pointing to
  concrete future work.

## Scope of claims (honesty)
- "Best-known solution (BKS)" values are literature references; we report gap to BKS, not proofs
  of optimality.
- CIM/CAC are specific implementations (Inoue–Yoshida traveling-wave; Leleu 2021); conclusions
  are about these implementations under the stated tuning budget, not all possible CIM/CAC.
- K2000 CAC numbers reflect a documented unweighted-internal-selection limitation of this CAC
  implementation, mitigated by uniform weighted re-scoring.
