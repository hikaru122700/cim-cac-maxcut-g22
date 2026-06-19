---
created: '2026-06-19T13:05:09+00:00'
evidence:
- stage-19/paper_revised.md
id: paper_revision-rc-20260619-123829-1ef52e
run_id: rc-20260619-123829-1ef52e
stage: 19-paper_revision
tags:
- paper_revision
- stage-19
- run-rc-20260
title: 'Stage 19: Paper Revision'
---

# Stage 19: Paper Revision

# PACE: When Tuned Heuristics Outrun Ising Machines on MAX-CUT

# Abstract

Coherent Ising Machines (CIM) and Chaotic Amplitude Control (CAC) are promoted as superior MAX-CUT solvers, yet the comparisons behind these claims typically pit a heavily tuned physics solver against classical heuristics left at literature-default settings — a protocol that inflates the apparent physics advantage. A credible comparison must tune every contender and score them with one identical objective. We present PACE (Parity-Adjusted Cut Evaluation), a framework that tunes all six contending solvers — CIM, CAC, simulated annealing, discrete simulated bifurcation, parallel tempering with isoenergetic cluster moves, and a memetic genetic algorithm — per instance with Optuna, scores them with a single weighted-cut functional, and composes them into warm-start hybrids and a best-of-K portfolio. Across four standard G-set and Sherrington–Kirkpatrick benchmarks, tuned classical heuristics lead on every instance: the memetic algorithm matches the best-known cut on G22 and leads the dense weighted K2000 instance, while discrete simulated bifurcation is the most reliable solver at a 0.19% mean relative gap. Fair tuning lifts annealing from last to mid-pack, ahead of both physics solvers. Warm-starting local search from a physics run cuts CIM's dense-instance gap by 85%, yet no hybrid beats the best tuned single solver — positioning CIM and CAC as explorers and portfolio members rather than standalone finishers.

# Introduction

MAX-CUT is among the most studied NP-hard combinatorial problems: given a weighted graph, the goal is to partition its vertices into two sets so that the total weight of edges crossing the partition is maximized (Karp, 1972). Beyond its theoretical centrality — it admits the celebrated 0.878-approximation via semidefinite relaxation (Goemans & Williamson, 1995) — MAX-CUT has become the de facto benchmark for a generation of physics-inspired optimization hardware and algorithms. Coherent Ising Machines (CIM) encode the problem in the steady state of coupled optical parametric oscillators (McMahon et al., 2016; Inagaki et al., 2016), Chaotic Amplitude Control (CAC) augments such dynamics with error-correcting auxiliary variables to escape local traps (Leleu et al., 2021), and Simulated Bifurcation (SB) numerically integrates an adiabatic Hamiltonian on commodity processors (Goto et al., 2019, 2021). Because these methods promise high-quality cuts in a fixed budget, the question that decides deployment value is direct: at a matched compute budget, which solver returns the better cut? That question can only be answered if every contender is given the same chance to be configured well.

The literature that motivates physics-inspired hardware rests on comparisons that are methodologically fragile. Three recurring problems undermine the standard "physics beats classical" narrative. First, the physics solver is usually tuned extensively — its couplings, pump schedules, and step sizes optimized per instance — while the classical opponents (simulated annealing, parallel tempering, genetic algorithms) are run at textbook default parameters, so the contest compares an optimized method against an un-optimized one (Hamerly et al., 2019; Leleu et al., 2021). Second, cut quality is often measured inconsistently: internal spin counters that ignore edge weights are adequate for unweighted G-set graphs but misreport quality on weighted Sherrington–Kirkpatrick instances. Third, results are reported as a single final cut without a fixed budget protocol, which lets a fast-converging solver be credited against a competitor that was simply given less effort (Aramon et al., 2019; Mohseni et al., 2022). Taken together, these practices leave a clear methodological hole: there exists no uniformly tuned, uniformly scored comparison of physics-inspired and classical MAX-CUT solvers. Without one, claims of physical advantage cannot be separated from tuning asymmetry.

To close this gap we introduce PACE (Parity-Adjusted Cut Evaluation), a fairness-first benchmark and composition harness. The central methodological move is parity: every solver — CIM, CAC, simulated annealing (SA), discrete simulated bifurcation (dSB), parallel tempering with isoenergetic cluster moves (PT-ICM), and a memetic genetic algorithm (GA) we implement specifically as a strong missing classical baseline — is tuned per instance with the same Optuna budget and the same objective. A single weighted-cut scorer is applied to the raw spin output of every solver, so weighted and unweighted instances are treated identically. The same tuned solvers are then recombined through two composition mechanisms: a warm-start hybrid that hands a physics solver's configuration to a classical local-search refiner, and a parallel best-of-K portfolio that hedges across solvers when no single one dominates. Each design choice removes one of the three asymmetries above.

Our study makes the fol

... (truncated, see full artifact)
