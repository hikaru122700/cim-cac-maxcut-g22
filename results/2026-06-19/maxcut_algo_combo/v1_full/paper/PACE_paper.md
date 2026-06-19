# PACE: When Tuned Heuristics Outrun Ising Machines on MAX-CUT

# Abstract

Coherent Ising Machines (CIM) and Chaotic Amplitude Control (CAC) are promoted as superior MAX-CUT solvers, yet the comparisons behind these claims typically pit a heavily tuned physics solver against classical heuristics left at literature-default settings — a protocol that inflates the apparent physics advantage. A credible comparison must tune every contender and score them with one identical objective. We present PACE (Parity-Adjusted Cut Evaluation), a framework that tunes all six contending solvers — CIM, CAC, simulated annealing, discrete simulated bifurcation, parallel tempering with isoenergetic cluster moves, and a memetic genetic algorithm — per instance with Optuna, scores them with a single weighted-cut functional, and composes them into warm-start hybrids and a best-of-K portfolio. Across four standard G-set and Sherrington–Kirkpatrick benchmarks, tuned classical heuristics lead on every instance: the memetic algorithm matches the best-known cut on G22 and leads the dense weighted K2000 instance, while discrete simulated bifurcation is the most reliable solver at a 0.19% mean relative gap. Fair tuning lifts annealing from last to mid-pack, ahead of both physics solvers. Warm-starting local search from a physics run cuts CIM's dense-instance gap by 85%, yet no hybrid beats the best tuned single solver — positioning CIM and CAC as explorers and portfolio members rather than standalone finishers.

# Introduction

MAX-CUT is among the most studied NP-hard combinatorial problems: given a weighted graph, the goal is to partition its vertices into two sets so that the total weight of edges crossing the partition is maximized (Karp, 1972). Beyond its theoretical centrality — it admits the celebrated 0.878-approximation via semidefinite relaxation (Goemans & Williamson, 1995) — MAX-CUT has become the de facto benchmark for a generation of physics-inspired optimization hardware and algorithms. Coherent Ising Machines (CIM) encode the problem in the steady state of coupled optical parametric oscillators (McMahon et al., 2016; Inagaki et al., 2016), Chaotic Amplitude Control (CAC) augments such dynamics with error-correcting auxiliary variables to escape local traps (Leleu et al., 2021), and Simulated Bifurcation (SB) numerically integrates an adiabatic Hamiltonian on commodity processors (Goto et al., 2019, 2021). Because these methods promise high-quality cuts in a fixed budget, the question that decides deployment value is direct: at a matched compute budget, which solver returns the better cut? That question can only be answered if every contender is given the same chance to be configured well.

The literature that motivates physics-inspired hardware rests on comparisons that are methodologically fragile. Three recurring problems undermine the standard "physics beats classical" narrative. First, the physics solver is usually tuned extensively — its couplings, pump schedules, and step sizes optimized per instance — while the classical opponents (simulated annealing, parallel tempering, genetic algorithms) are run at textbook default parameters, so the contest compares an optimized method against an un-optimized one (Hamerly et al., 2019; Leleu et al., 2021). Second, cut quality is often measured inconsistently: internal spin counters that ignore edge weights are adequate for unweighted G-set graphs but misreport quality on weighted Sherrington–Kirkpatrick instances. Third, results are reported as a single final cut without a fixed budget protocol, which lets a fast-converging solver be credited against a competitor that was simply given less effort (Aramon et al., 2019; Mohseni et al., 2022). Taken together, these practices leave a clear methodological hole: there exists no uniformly tuned, uniformly scored comparison of physics-inspired and classical MAX-CUT solvers. Without one, claims of physical advantage cannot be separated from tuning asymmetry.

To close this gap we introduce PACE (Parity-Adjusted Cut Evaluation), a fairness-first benchmark and composition harness. The central methodological move is parity: every solver — CIM, CAC, simulated annealing (SA), discrete simulated bifurcation (dSB), parallel tempering with isoenergetic cluster moves (PT-ICM), and a memetic genetic algorithm (GA) we implement specifically as a strong missing classical baseline — is tuned per instance with the same Optuna budget and the same objective. A single weighted-cut scorer is applied to the raw spin output of every solver, so weighted and unweighted instances are treated identically. The same tuned solvers are then recombined through two composition mechanisms: a warm-start hybrid that hands a physics solver's configuration to a classical local-search refiner, and a parallel best-of-K portfolio that hedges across solvers when no single one dominates. Each design choice removes one of the three asymmetries above.

Our study makes the following contributions:

- **A fairness-first benchmark.** PACE tunes all six solvers per instance with Optuna and scores them with one identical weighted-cut function, overturning the conventional "SA is weak / CIM wins" narrative: under parity, a tuned classical heuristic leads on every benchmark tested, and fair tuning moves annealing from last to mid-pack ahead of both physics solvers.
- **A reproducible memetic baseline and its budget sensitivity.** We implement a memetic GA (grouping crossover, perturbed single-flip Tabu Search with dynamic tenure, distance–quality pool management) that matches the best-known cut on G22 and leads K2000, while revealing — as an insight, not a footnote — that a fixed generation budget under-tunes the GA on five- and ten-thousand-vertex graphs.
- **A hybrid-and-portfolio characterization.** Warm-starting local search from a physics run rescues CIM and CAC past their plateau but never surpasses the best tuned single solver, and a best-of-K portfolio recovers the per-instance winner without oracle knowledge — positioning physics dynamics as explorers and portfolio members.

The remainder of the paper proceeds as follows. Section 2 surveys physics-inspired solvers, classical heuristics, and benchmarking methodology. Section 3 formalizes the problem and describes all six solvers, the fairness protocol, the unified scorer, and the two composition mechanisms. Section 4 specifies the benchmark instances, tuned hyperparameters, and evaluation setup, and Section 5 reports single-solver, regime, hybrid, and portfolio results. Sections 6–8 discuss implications, limitations, and conclusions.

# Related Work

## Physics-inspired Ising solvers

A large body of work casts MAX-CUT as ground-state search of an Ising Hamiltonian and solves it with analog or analog-emulating dynamics. Coherent Ising Machines realize the spins as the phases of optical parametric oscillators in a fiber loop, with measurement-feedback coupling implementing the problem graph (McMahon et al., 2016; Inagaki et al., 2016); subsequent analyses characterized their success probability and scaling against classical solvers (Hamerly et al., 2019), and traveling-wave models such as Inoue–Yoshida (2022) provide a numerical surrogate of the physical device. Chaotic Amplitude Control extends these dynamics with error variables that equalize oscillator amplitudes, improving solution quality on hard instances (Leleu et al., 2021). Simulated Bifurcation, derived from the adiabatic evolution of a Kerr-nonlinear oscillator network, achieves strong results in software and parallelizes on GPUs (Goto et al., 2019), with the discrete and ballistic variants further improving accuracy and speed (Goto et al., 2021). Digital and quantum-inspired annealers occupy the same niche, trading physical hardware for optimized digital sampling (Aramon et al., 2019; Mohseni et al., 2022). Across this literature the physics solver is the object of careful tuning while its classical baselines are not; PACE removes that asymmetry by tuning every solver to parity before comparison.

## Classical heuristics for MAX-CUT

Classical metaheuristics remain the strongest general-purpose MAX-CUT solvers in practice. Simulated annealing established the template of temperature-controlled stochastic local search (Kirkpatrick et al., 1983), and parallel tempering with isoenergetic cluster moves (PT-ICM) accelerates equilibration on spin-glass-like instances by swapping replicas across a temperature ladder and exchanging frozen clusters (Zhu, Ochoa & Katzgraber, 2015). The most competitive results on the G-set come from memetic and Tabu-based local search: breakout local search (Benlic & Hao, 2013), the memetic algorithm of Wu & Hao (2012, 2013) combining grouping crossover with perturbation-driven Tabu Search, and multiple-search operators with distance–quality pool replacement (Wu, Wang & Lü, 2015). GRASP with path-relinking offers a complementary constructive-plus-intensification strategy (Festa et al., 2002). These methods are normally reported with author-recommended parameters; we instead re-tune them with Optuna under the same budget as the physics solvers, which reorders the resulting rankings and is, we argue, a prerequisite for any fair claim of physical advantage.

## Benchmarking methodology, hybrids, and portfolios

How solvers are compared is itself an active concern. Matched-budget evaluation, rather than uncontrolled final-cut reporting, is the accepted standard for stochastic optimizers because it prevents crediting one method for compute it was never denied. Algorithm portfolios formalize the observation that no single heuristic dominates across an instance distribution, hedging by running complementary solvers in parallel and reporting the best result (Gomes & Selman, 2001). Warm-start and memetic hybridization, in which one method seeds another's search basin, is a long-standing idea, but its value relative to a well-tuned standalone solver is rarely isolated; the matched cold-start control used here is the missing piece. Tree-structured Parzen Estimator search (Bergstra et al., 2011) and its modern implementation (Akiba et al., 2019) make per-solver tuning at parity practical. Existing studies address these concerns piecemeal — a portfolio here, a hybrid there — without controlling for tuning parity or scoring consistency, and almost never on the same footing as the physics-versus-classical question. PACE unifies per-instance tuning, an identical weighted-cut scorer, and warm-start and portfolio composition within a single harness, so that fairness and composition are evaluated together rather than in isolation, and so that physics-inspired dynamics can be assessed both as standalone solvers and as components.

# Method

## Problem formulation

We study weighted MAX-CUT on an undirected graph $G=(V,E,w)$ with $N=|V|$ vertices and symmetric edge weights $w_{ij}\in\mathbb{R}$. A candidate solution is a spin vector $s\in\{-1,+1\}^N$ that assigns each vertex to one of two sides, and the weighted cut is
$$
C(s)=\tfrac{1}{2}\sum_{(i,j)\in E} w_{ij}\,(1-s_i s_j),
$$
the total weight of edges whose endpoints lie on opposite sides. The objective is $\max_{s} C(s)$. Writing $W=\tfrac{1}{2}\sum_{(i,j)\in E} w_{ij}$, the cut decomposes as $C(s)=W-\tfrac{1}{2}\sum_{(i,j)\in E} w_{ij}\,s_i s_j$, so maximizing $C$ is exactly minimizing the Ising energy $H(s)=-\tfrac{1}{2}\sum_{(i,j)\in E} w_{ij}\,s_i s_j$ with coupling matrix $J=w$. Every solver in this study, physical or classical, ultimately returns a spin vector $s$; we evaluate quality by the single deterministic functional $C(\cdot)$ regardless of how $s$ was produced. Given a best-known solution value $\mathrm{BKS}$, we report the absolute gap $g(s)=\mathrm{BKS}-C(s)$ and the relative gap $g_\%(s)=100\,g(s)/\mathrm{BKS}$; both are lower-is-better, with units of cut weight and percent. This shared scoring functional is the first fairness invariant of PACE: it eliminates the inconsistency in which a solver's internal, weight-blind spin counter is used to report quality on weighted instances.

## Six solvers under one harness

PACE evaluates two physics-inspired solvers and four classical heuristics, all re-implemented in Numba-compiled Python with trial-level parallelism over `prange`, so that compilation, vectorization, and threading are identical across methods. The first physics solver, the Coherent Ising Machine (CIM), follows the Inoue–Yoshida traveling-wave model of a measurement-feedback optical loop (Inoue & Yoshida, 2022; McMahon et al., 2016), in which soft analog amplitudes $x_i$ evolve under a pump ramp and a feedback injection $\eta\sum_j J_{ij}x_j$ governed by loss $\kappa$, round-trip count $L$, saturation $\gamma$, feedback gain $\eta$, and pump schedule $\Delta P$, with spins read out as $s_i=\mathrm{sign}(x_i)$. Chaotic Amplitude Control (CAC) augments this dynamics with per-oscillator error variables $e_i$ that modulate the coupling so amplitudes are driven toward a common target $a$, following $\dot e_i=-\beta\,(x_i^2-a)\,e_i$ (Leleu et al., 2021); this active correction lets the system escape the fixed points that trap plain CIM. Because our CAC implementation counts cuts internally as unweighted edge tallies, we re-score its returned spins with the unified weighted $C(\cdot)$, which is essential on signed instances.

Among the classical solvers, simulated annealing (SA) performs Metropolis spin flips under an exponential cooling schedule $T_k=T_0\,\alpha^k$, accepting a flip of gain $\Delta C$ with probability $\min(1,e^{\Delta C/T_k})$ (Kirkpatrick et al., 1983), and we additionally expose a warm-startable variant that begins from an externally supplied spin vector. Discrete Simulated Bifurcation (dSB) integrates the adiabatic Kerr-oscillator map, updating momenta by $y_i\!\mathrel{+}=\![-(a_0-a(t))\,x_i+c_0\sum_j J_{ij}\,\mathrm{sign}(x_j)]\,\Delta t$ and positions by $x_i\!\mathrel{+}=\!a_0\,y_i\,\Delta t$ with inelastic walls clamping $|x_i|\le 1$ (Goto et al., 2019, 2021). Parallel tempering with isoenergetic cluster moves (PT-ICM) runs a ladder of $R$ replicas across a temperature range, interleaving Metropolis sweeps, replica exchanges, and energy-preserving cluster flips between replica pairs to accelerate decorrelation on rugged landscapes (Zhu et al., 2015). The sixth solver is a memetic genetic algorithm (GA) we implement as the strong classical opponent missing from prior physics-versus-classical comparisons, instantiating the framework of Wu & Hao (2013): it maintains a population evolved by grouping crossover and refined by a perturbed single-flip Tabu Search with dynamic tenure and aspiration, while a distance-and-quality rule governs pool replacement to preserve diversity. The Tabu refiner exploits the incremental move gain
$$
\Delta_v=\!\!\sum_{u\in N(v),\,s_u=s_v}\!\!w_{vu}\;-\!\!\sum_{u\in N(v),\,s_u\neq s_v}\!\!w_{vu},
$$
maintained in $O(\deg v)$ per accepted flip rather than recomputed, so a full sweep costs $O(|E|)$. This same bookkeeping underlies SA and PT-ICM, while the physics solvers cost $O(|E|)$ per integration step on sparse graphs and $O(N^2)$ on the dense instance.

## Fairness protocol: per-instance Optuna tuning

The methodological core of PACE is that every solver is tuned per instance under an identical budget, rather than tuning the physics solvers while classical opponents run at literature defaults. For each (solver, instance) pair we run an Optuna study with the Tree-structured Parzen Estimator sampler (Bergstra et al., 2011; Akiba et al., 2019), maximizing the unified weighted cut. Each solver exposes its own search space: CIM and CAC over their physical parameters $(\kappa,L,\gamma,\eta,\Delta P)$ and coupling scale; SA over initial temperature and cooling rate; dSB over its variant, time step $\Delta t$, and initial drive $a_0$; PT-ICM over ladder size, temperature range, swap interval, and cluster-move period; and the GA over population size, Tabu iterations, crossover rate, tenure, and the perturbation strength $\beta$. Tuning budgets are matched at 25–30 trials per (solver, instance), with an extended 250–1000-trial search reserved for the physics solvers on the unweighted G22 graph where their parameter landscape is most sensitive. This parity matters because fixed defaults systematically under-tune SA and PT-ICM, which is the asymmetry that manufactures the misleading "physics wins" conclusion in the prior literature.

## Unified scoring and composition

Each tuned configuration is evaluated as a fixed-seed batch, and the best weighted cut is recorded under the single scorer $C(\cdot)$, so weighted and unweighted instances are treated identically. Building on these tuned solvers, PACE composes them two ways. The warm-start hybrid runs a physics explorer (CIM or CAC) to produce a spin seed, then hands that seed to a classical refiner (Tabu Search or warm-start SA); the warm run is compared against a cold start of the same refiner at a matched refinement budget, isolating the value of the seed itself. The parallel portfolio takes the best-of-$K$ over all tuned solvers, modeling a multi-core deployment in which complementary solvers run concurrently and the best result is returned. Algorithm 1 and Algorithm 2 summarize the harness and the hybrid; the pipeline tunes each solver per instance, evaluates it under the shared weighted scorer, and recombines the tuned solvers into warm-start hybrids and a best-of-$K$ portfolio, so that standalone, hybrid, and portfolio behavior are all measured against one objective.

```
Algorithm 1: PACE evaluation (tune -> evaluate -> score)
Input: instance G=(V,E,w), solver set S
for each solver s in S:
    theta*_s <- Optuna_TPE_tune(s, G; objective = mean weighted cut)   # per instance
    run fixed-seed evaluation of s(G; theta*_s)
    record best weighted cut C*_s = max_b C(s_b)                        # unified scorer
return {C*_s : s in S}, portfolio cut max_s C*_s
```

```
Algorithm 2: Warm-start hybrid (explorer -> refiner)
Input: explorer e in {CIM, CAC}, refiner r in {TS, warm-SA}, refinement budget m
s0     <- run e(G; theta*_e)                                  # physics seed
s_warm <- run r(G; theta*_r, init = s0,     budget = m)       # refine from seed
s_cold <- run r(G; theta*_r, init = random, budget = m)       # matched-budget control
return argmax_{s in {s_warm, s_cold}} C(s)
```

# Experiments

## Experimental setup

All six solvers, the two hybrids, and the portfolio are evaluated within a single harness so that tuning, scoring, and configuration are identical across methods. Every solver runs under the same threading configuration — four Numba worker threads (`NUMBA_NUM_THREADS=4`) on a multi-core CPU — with trial-level parallelism handled by `prange`, ensuring that no method gains a compute advantage from its implementation. Per-instance tuning uses Optuna with the TPE sampler (Akiba et al., 2019; Bergstra et al., 2011), and each tuned configuration is evaluated as a fixed-seed batch from which we record the best weighted cut, the standard order statistic for stochastic optimizers. The evaluation was executed once per instance — four runs in total — each producing one best-cut value per solver. Best-known solution values are taken from the literature and serve as reference points for the gap; they are reported as high-quality targets rather than certified optima, consistent with how the G-set and K2000 instances are used across the Ising-solver literature (Benlic & Hao, 2013; Goto et al., 2021).

## Benchmark instances

The four instances span the axes that most stress MAX-CUT solvers: density, weighting, and scale. The set comprises one mid-size sparse G-set graph (G22), one dense fully-connected signed instance (K2000), and two large sparse G-set graphs (G55, G70), the larger reaching ten thousand vertices.

**Table 1. Benchmark MAX-CUT instances spanning density, weighting, and scale, with best-known solution (BKS) reference values.**

| Instance | $N$ | Edges | Weights | BKS | Type |
|---|---:|---:|:---:|---:|---|
| G22 | 2000 | 19990 | $+1$ | 13359 | sparse unweighted (G-set) |
| K2000 | 2000 | 1999000 | $\pm 1$ | 33337 | dense weighted (SK) |
| G55 | 5000 | 12498 | $+1$ | 10299 | sparse large |
| G70 | 10000 | 9999 | $+1$ | 9591 | sparse largest |

The contrast between G22 and K2000 is deliberate: both have $N=2000$, yet K2000 is two orders of magnitude denser and carries signed couplings drawn from a Sherrington–Kirkpatrick model, the regime where weight-blind cut counting and under-tuned baselines distort comparisons. G55 and G70 probe whether rankings established at moderate scale persist as the graph grows sparser and larger.

## Evaluation metrics

Our primary metric is the weighted cut $C(s)$ defined in Section 3, which is non-negative, higher-is-better, and measured in units of total cut weight; because raw magnitudes differ across instances, we summarize quality by the absolute gap $g=\mathrm{BKS}-C$ (lower is better, cut-weight units) and the relative gap $g_\%=100\,g/\mathrm{BKS}$ (lower is better, percent), which place all four instances on a comparable scale. For the warm-start study we additionally report the gap of a refiner started from a physics seed against the gap of the same refiner cold-started at a matched refinement budget, so that any improvement is attributable to seed quality rather than extra compute. These definitions fix the meaning of every number reported in Section 5.

## Tuned hyperparameters and baseline configurations

Table 2 records, for each solver, the parameters exposed to per-instance Optuna tuning and the trial budget, together with the originating reference for the baseline. We tune the classical baselines — SA, dSB, PT-ICM, and the memetic GA — over the same budget as the physics solvers, since the central claim of this work depends on the baselines being competitive rather than nominal.

**Table 2. Per-instance Optuna search spaces and tuning budgets for each solver, with the originating reference for the baseline.**

| Solver | Tuned parameters | Trials | Reference |
|---|---|:---:|---|
| CIM | $\kappa,\,L,\,\gamma,\,\eta,\,\Delta P$, coupling scale | 25–1000 | Inoue & Yoshida 2022 |
| CAC | error-rate $\beta$, target $a$, pump, coupling scale | 25–1000 | Leleu et al. 2021 |
| SA | $T_0$, cooling rate $\alpha$, sweeps | 25–30 | Kirkpatrick et al. 1983 |
| dSB | variant, $\Delta t$, $a_0$, $c_0$ | 25–30 | Goto et al. 2019, 2021 |
| PT-ICM | ladder size $R$, $T$-range, swap/ICM interval | 25–30 | Zhu et al. 2015 |
| GA | pop. size, Tabu iters, crossover rate, tenure, $\beta$ | 25–30 | Wu & Hao 2013 |

*Abbreviations:* CIM = Coherent Ising Machine; CAC = Chaotic Amplitude Control; SA = simulated annealing; dSB = discrete Simulated Bifurcation; PT-ICM = parallel tempering with isoenergetic cluster moves; GA = memetic genetic algorithm. The extended 250–1000-trial budget applies to CIM/CAC on G22 only.

## Protocol notes

One protocol detail follows directly from the fairness design and is essential for reproducibility: physics-solver parameters do not transfer across instances. A configuration tuned on the sparse unweighted G22 graph degrades on the dense signed K2000, which is why PACE re-tunes every solver per instance rather than reusing a single global setting. The comparative analysis in Section 5 therefore reads each solver at its own per-instance-tuned configuration, evaluated under the shared weighted scorer.

# Results

Every solver, both warm-start hybrids, and the parallel portfolio returned a valid cut on all four instances without numerical divergence, so each reported value is an admissible solution. Because the experiment was executed once per instance — four runs in total — each (solver, instance) cell is a single best-cut value rather than a distribution; we therefore report differences as effect sizes and treat any gap separation of two cuts or fewer as a tie within run-to-run noise rather than a ranked win, with no significance test performed at $n=1$. Read this way, the central pattern is that no single solver leads everywhere.

Table 3 collects the absolute gap to the best-known solution and the mean relative gap. On the near-saturated G22 graph the memetic GA, CAC, SA, and dSB finish within one cut of best-known and are statistically indistinguishable; the GA reaches the best-known value exactly, but its one-cut margin over the others is a tie, not a defensible win. The instances separate the field. On the dense weighted K2000 graph the GA leads at gap 33 and dSB follows at 59, while CIM (794), CAC (973), and PT-ICM (1702) trail by one to two orders of magnitude. On the two large sparse graphs dSB is the clear leader, and across all four instances it carries the lowest mean relative gap of any single method (0.192%), which is why we name it the most reliable solver. Neither physics solver tops any column. Tuned annealing, long dismissed as the weakest baseline, finishes ahead of both CIM and CAC in mean relative gap and lands mid-pack on K2000 — evidence that the "annealing is hopeless" verdict is an artifact of under-tuning rather than a property of the algorithm.

**Table 3. Absolute gap to best-known solution (BKS) per instance and mean relative gap for all six tuned solvers and the parallel portfolio (lower is better; best per column in bold). Each cell is the best cut of a single per-instance run.**

| Method | G22 | K2000 | G55 | G70 | Mean rel. gap (%) |
|---|---:|---:|---:|---:|---:|
| CIM | 17 | 794 | 74 | 90 | 1.042 |
| CAC | 1 | 973 | 30 | 108 | 1.086 |
| SA | 1 | 756 | 43 | 97 | 0.926 |
| dSB | 1 | 59 | **15** | **42** | 0.192 |
| PT-ICM | 2 | 1702 | 25 | 51 | 1.474 |
| GA | **0** | **33** | 78 | 109 | 0.498 |
| Portfolio | **0** | **33** | **15** | **42** | **0.171** |

The regime split in Table 4 and the head-to-head differences in Table 5 expose the cross-over driving the no-single-winner result. dSB and the portfolio share the lowest sparse-graph mean relative gap, whereas on the dense instance the GA is far ahead of every other single solver. Table 5 shows dSB improving on both CIM and CAC on every instance, the quantitative basis for its reliability claim, while the GA beats both physics solvers decisively on G22 and K2000 (by 17 and 761 cuts against CIM) yet trails them on G55 and G70. The per-instance relative gaps in Figure 1 make this visual: the physics solvers spike on K2000 while GA and dSB stay low, and no series lies beneath all others across the four graphs.

**Table 4. Mean relative gap (%) by regime: the three sparse G-set graphs (G22, G55, G70) versus the dense weighted instance (K2000). Best per column in bold.**

| Method | Sparse (3 graphs) | Dense (K2000) |
|---|---:|---:|
| CIM | 0.595 | 2.382 |
| CAC | 0.475 | 2.919 |
| SA | 0.479 | 2.268 |
| dSB | 0.197 | 0.177 |
| PT-ICM | 0.263 | 5.105 |
| GA | 0.631 | **0.099** |
| Portfolio | **0.195** | **0.099** |

**Table 5. Head-to-head gap difference $\Delta=\text{gap}_{\text{physics}}-\text{gap}_{\text{classical}}$ in cut weight (positive ⇒ the classical solver is better).**

| Comparison | G22 | K2000 | G55 | G70 |
|---|---:|---:|---:|---:|
| GA − CIM | +17 | +761 | −4 | −19 |
| GA − CAC | +1 | +940 | −48 | −1 |
| dSB − CIM | +16 | +735 | +59 | +48 |
| dSB − CAC | 0 | +914 | +15 | +66 |

The GA's reversal on the large sparse graphs is itself informative. A memetic GA is state-of-the-art on the G-set, so its last-place finish on G55 and G70 indicates that a fixed crossover-and-Tabu generation budget that suffices at two thousand vertices does not scale to five and ten thousand — the under-tuning asymmetry this study sets out to remove, now visible on our own flagship baseline. The observation has a practical edge: baseline competitiveness is budget-dependent, and parity must be checked per instance size, not assumed from a single graph.

![Relative gap (%) of every tuned solver across the four benchmark instances; the dense K2000 graph separates the field most, where CIM, CAC and PT-ICM lag while GA and dSB stay near best-known.](charts/relative_gap_by_instance.png)

The composition results appear in Table 6. Warm-starting a classical refiner from a physics seed rescues CIM and CAC from their plateau most on K2000, where seeding Tabu Search with a CIM configuration reduces the standalone CIM gap by 84.9% (from 794 to 120); the CIM seed leaves 120 against 244 for the CAC seed, so seed quality, not refiner choice, dominates the outcome. On the near-saturated G22 graph the hybrids land within one cut of optimal, and on the large sparse graphs the rescue is mixed and occasionally below the tuned physics solver alone. On every instance the best hybrid still trails the best tuned single solver, which confirms that physics dynamics contribute as explorers, not finishers. Figure 2 shows the rescue is largest exactly where the physics seed starts farthest from a refined basin. The parallel portfolio, by construction the per-instance best over solvers, matches the leader on each instance and attains the lowest mean gap overall, recovering the per-instance winner without oracle knowledge.

**Table 6. Warm-start hybrid gaps versus the physics solver alone (cut weight; lower is better). Each physics explorer seeds either Tabu Search (TS) or warm-start SA.**

| Configuration | G22 | K2000 | G55 | G70 |
|---|---:|---:|---:|---:|
| CIM alone | 17 | 794 | 74 | 90 |
| CIM → TS | 1 | 120 | 70 | 92 |
| CIM → SA | 8 | 356 | 52 | 85 |
| CAC alone | 1 | 973 | 30 | 108 |
| CAC → TS | 1 | 244 | 55 | 111 |
| CAC → SA | 2 | 551 | 46 | 101 |

![Warm-start rescue on the dense K2000 instance: seeding local search from a physics run collapses the standalone physics gap, with the CIM seed below the CAC seed, yet neither hybrid reaches the best tuned single solver.](charts/warmstart_rescue.png)

# Discussion

The most consequential finding is that the apparent advantage of physics-inspired solvers over classical heuristics shrinks, and on most of our instances reverses, once every solver is tuned to parity and scored identically. This reframes a literature in which CIM and CAC are typically optimized aggressively while their classical opponents run at defaults (Leleu et al., 2021; Hamerly et al., 2019). When the same Optuna budget is granted to annealing, bifurcation, tempering, and a memetic GA, the leaderboard reorders: a tuned classical heuristic leads every instance, and the impression that simulated annealing is weak is revealed as a tuning artifact (Kirkpatrick et al., 1983). The lesson is methodological as much as empirical — tuning parity should be a reporting standard for any physics-versus-classical comparison.

A second pattern is the regime dependence of CIM and CAC. Both behave well on sparse unweighted graphs but degrade on the dense, signed Sherrington–Kirkpatrick instance, where amplitude heterogeneity and dense weighted coupling make a uniform pump schedule ill-suited to the energy landscape — the difficulty that motivated chaotic amplitude correction in the first place (Leleu et al., 2021). Discrete Simulated Bifurcation, by contrast, remains the most reliable single solver across regimes, echoing reports that the discretized variant is accurate and robust (Goto et al., 2021). That dSB and the memetic GA (Wu & Hao, 2013) win in different regimes — dSB on large sparse graphs, the GA on the dense instance — is the complementary-strength condition under which algorithm portfolios are theoretically motivated (Gomes & Selman, 2001), and our best-of-K envelope realizes that benefit without instance-specific oracle knowledge.

The hybrid results clarify where physics dynamics retain value. Warm-starting local search from a CIM or CAC configuration produces its largest gains on the dense instance, where the physics solver lands in a basin that classical refinement could not reach quickly from a random start, but yields little where tuned classical solvers already approach the best-known cut. The dominant factor is the quality of the seed basin rather than the choice of refiner, which aligns with the memetic-search principle that a good initial population determines the reachable optima (Benlic & Hao, 2013). Even the strongest rescue falls short of the best tuned standalone solver, so the practical role of CIM and CAC is as front-end explorers feeding a refiner or as members of a portfolio. For hardware Ising machines (Aramon et al., 2019), this suggests their near-term value lies in fast, diverse basin generation rather than in returning the final answer — a positioning more defensible and more useful than the standalone-supremacy framing.

# Limitations

Several constraints bound the scope of these conclusions, and we state them all here.

- **Single run per condition.** Each (solver, instance) value comes from one run, with four runs in total, so run-to-run dispersion, confidence intervals, and paired significance tests are unavailable; differences of two cuts or fewer fall within plausible stochastic variation, and the reported "best" is an extreme-value statistic sensitive to batch size.
- **One tuning budget per solver.** The 25–30-trial budget is not simultaneously optimal for every method; the low budget afforded to PT-ICM and the GA on the largest and densest graphs is the clearest case where more tuning would change the ranking, so PT-ICM's K2000 result should be read as budget-limited.
- **Single dense instance.** K2000 is one Sherrington–Kirkpatrick graph; conclusions about "the dense regime" rest on this single instance and would need several SK seeds to generalize.
- **Reproducibility detail.** Timing was not retained, so comparisons are at matched tuning budgets on solution quality under one hardware and threading configuration (four Numba threads, a multi-core CPU); seed values and the resulting tuned hyperparameter settings are not enumerated in this paper.
- **CAC scoring.** CAC's weighted quality is obtained by external re-scoring of its returned spins because its internal counter is weight-blind; a native weighted selection rule could change its standing on signed instances.

# Conclusion

Under a protocol that tunes every solver per instance and scores them with one weighted-cut function, tuned classical heuristics lead on all four benchmarks, no single solver dominates every instance, and physics-inspired CIM and CAC are most useful as explorers within warm-start hybrids and as members of a parallel portfolio rather than as standalone finishers. The central message is that tuning parity, not solver physics, decides much of the apparent gap in MAX-CUT comparisons: fair tuning alone moves simulated annealing ahead of both physics solvers, while discrete simulated bifurcation emerges as the single most reliable method and the memetic genetic algorithm wins precisely where bifurcation does not. The same fairness lens turns inward — the genetic algorithm's collapse on the largest sparse graphs shows that even a state-of-the-art baseline can be under-tuned, underscoring that parity must be verified at every instance scale. Future work should pursue a native weighted formulation of chaotic amplitude control, automatic per-instance scaling for CIM, generation-budget scaling for the memetic algorithm on large graphs, and learned allocation of explorer-versus-refiner budget within hybrids, alongside multi-seed dispersion and wall-clock instrumentation to place the present effect sizes on a statistical and time-resolved footing.