---
created: '2026-06-19T08:35:56+00:00'
evidence:
- stage-19/paper_revised.md
id: paper_revision-rc-20260619-081359-3c29ca
run_id: rc-20260619-081359-3c29ca
stage: 19-paper_revision
tags:
- paper_revision
- stage-19
- run-rc-20260
title: 'Stage 19: Paper Revision'
---

# Stage 19: Paper Revision

# WARP: Warm-Starting and Portfolio Strategies for Physics-Inspired MAX-CUT Solvers

# Abstract

Coherent Ising Machines (CIM) and Chaotic Amplitude Control (CAC) are promoted as fast MAX-CUT solvers, yet their reported advantage over tuned classical heuristics rests on incomparable measurements: solver-internal cut counts, mixed time units, and under-tuned classical baselines. Whether physics-inspired dynamics are best deployed standalone or combined with classical search has not been quantified under a common scoring rule. We present WARP, a benchmark that scores every solver with one shared weighted-cut function recomputed from the spins it returns, paired with two combination strategies — a warm-start hybrid that hands a brief physics run to a classical refiner, and a parallel best-of-K portfolio. Across four MAX-CUT instances spanning sparse, dense weighted, and large-scale regimes, the discrete Simulated Bifurcation solver attains the smallest mean relative gap to best-known cuts (0.20%), while CIM and CAC sit mid-pack once scored on the true weighted objective. No single solver wins everywhere: Simulated Bifurcation leads on three instances and parallel tempering on the fourth. Warm-starting Tabu Search from a brief CIM run reduces CIM's gap on the dense instance by 85%, rescuing the physics solver into a competitive range. These results indicate that physics-inspired dynamics are most useful as components — explorers within hybrids and members of portfolios — rather than as standalone optimizers.

# Introduction

MAX-CUT is among the most studied NP-hard combinatorial optimization problems: given a graph, it asks for a bipartition of the vertices that maximizes the total weight of edges crossing the partition. Its reach extends well beyond graph theory, because MAX-CUT is the canonical zero-field Ising ground-state problem, and a broad family of scheduling, circuit-layout, and statistical-physics tasks reduces to it (Barahona et al., 1988; Lucas, 2014). This dual identity has turned MAX-CUT into the standard testbed for an emerging class of physics-inspired solvers that emulate analog dynamical systems. Coherent Ising Machines exploit the bistable phase of degenerate optical parametric oscillators to relax toward low-energy spin configurations (McMahon et al., 2016; Inoue & Yoshida, 2022); Simulated Bifurcation discretizes the adiabatic evolution of a network of nonlinear oscillators (Goto et al., 2019; 2021); and Chaotic Amplitude Control augments such dynamics with error feedback that equalizes oscillator amplitudes (Leleu et al., 2021). These methods report striking time-to-solution figures and have driven sustained interest in dedicated hardware accelerators (Mohseni et al., 2022).

Despite this momentum, the literature offers little that lets a practitioner answer a basic question: for a fixed search budget, which solver returns the best cut, and is a physics-inspired solver worth using at all? Three methodological gaps recur. First, physics-machine evaluations frequently report solver-internal cut counts that ignore edge weights or use machine-native iteration units, making cross-solver comparison apples-to-oranges (Tiunov et al., 2019; Kalinin & Berloff, 2018). Second, classical baselines such as simulated annealing (Kirkpatrick et al., 1983) and tabu search (Glover, 1989) are often included as strawmen rather than tuned competitors, even though memetic and tabu-based algorithms remain the strongest known MAX-CUT heuristics on standard instances (Wu & Hao, 2013; Benlic & Hao, 2013). Third, although physics dynamics and classical local search are widely assumed to be complementary, the value of combining them is seldom measured under a matched budget and a common scoring rule, in contrast to the controlled warm-start studies emerging in quantum optimization (Egger et al., 2021). Left unaddressed, these gaps leave claims of physics-inspired superiority resting on incomparable numbers.

To close these gaps, we build WARP on a single methodological commitment: every solver is judged by one shared weighted-cut score, recomputed from the spins it actually returns. Around this protocol we tune all six solvers per instance so that each is shown near its best, then study two ways of combining them. The first is a warm-start hybrid that runs a physics explorer (CIM or CAC) for a short fixed budget, takes its spin configuration, and hands it to a classical refiner (Tabu Search or warm-start annealing), charging the total time of explorer plus refiner. The central hypothesis is that the quality of the handed-off basin governs the refined result — a hypothesis we test by varying the explorer while holding the refiner fixed. The second strategy is a parallel portfolio that runs several tuned solvers concurrently and reports their running best, converting the absence of a universal champion into an oracle-free robustness mechanism.

Our contributions are threefold:

- **A uniformly scored benchmark** of six

... (truncated, see full artifact)
