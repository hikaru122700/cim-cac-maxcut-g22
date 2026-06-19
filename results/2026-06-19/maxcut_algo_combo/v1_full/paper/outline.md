# Paper Outline — PACE
### *Fair Anytime Comparison of Tuned Physics-Inspired and Classical MAX-CUT Solvers, with Warm-Start Hybrids and Portfolios*

---

## 0. Method Name & Title Candidates

**Committed method name: `PACE`** (4 chars)
— *Physics-And-Classical anytime Evaluation*: a unified harness that (i) tunes **every** solver per instance via Optuna, (ii) scores all of them on one wall-clock × weighted-cut axis, and (iii) composes them into warm-start hybrids and best-of-K portfolios. The name also puns on the wall-clock ("pace") theme central to anytime evaluation.

### Candidate titles (each ≤ 14 words)

| # | Title | Words | Memorability | Specificity | Novelty signal |
|---|-------|-------|:---:|:---:|:---:|
| **T1** | **PACE: A Fair Anytime Benchmark of Tuned Physics-Inspired and Classical MAX-CUT Solvers** | 12 | 4 | 5 | 3 |
| **T2** | **PACE: When Tuned Heuristics Outrun Ising Machines on MAX-CUT** | 10 | 5 | 4 | 4 |
| **T3** | **PACE: Warm-Start Hybrids Rescue Physics-Inspired MAX-CUT Solvers, but Tuning Wins** | 11 | 4 | 5 | 4 |

**Recommendation:** Use **T2** as the headline title (declarative, surprising, memorable) with **T1**'s descriptive content folded into the abstract. T2 best advertises the paper's most counter-intuitive, evidence-backed claim: *fair per-instance tuning overturns the physics-vs-classical narrative*. Rationale for ratings — T2 scores highest on memorability/novelty (a surprising declarative claim about Ising machines "losing"); T1 is the safest, most specific descriptive option; T3 foregrounds the hybrid "rescue" framing but is the longest.

---

## 1. Abstract — *target 180–220 words*

**Goal.** Deliver the PMR+ arc in one paragraph; name PACE by sentence 3; carry ≥3 concrete numbers.

- **S1–S2 (Problem).** Coherent Ising Machines (CIM) and Chaotic Amplitude Control (CAC) are promoted as strong MAX-CUT solvers, yet head-to-head claims against classical heuristics routinely compare an optimized physics solver against *under-tuned* classical baselines — an unfair setup that inflates the apparent advantage.
- **S3–S4 (Method).** We introduce **PACE**, a fair anytime evaluation framework that tunes *all six* solvers (CIM, CAC, SA, SB, PT-ICM, and a newly implemented memetic GA) per instance with Optuna, scores every solver with one identical weighted-cut function on a shared wall-clock axis, and composes warm-start hybrids and parallel portfolios from the same tuned solvers.
- **S5–S6 (Results).** Across four benchmarks (G22, K2000, G55, G70): tuned classical heuristics dominate — the memetic GA reaches the best-known cut exactly on G22 (gap 0) and leads dense K2000 (gap 33), while discrete SB is most reliable (mean gap ≈0.20%). Fair tuning rescues SA from "hopeless" to mid-pack (K2000 gap 1348→756). Warm-starting Tabu Search from a CIM run breaks the physics plateau (CIM gap 794→120 on K2000) but **does not beat the best tuned single solver**. We conclude physics-inspired dynamics are best used as **explorers in hybrids and members of portfolios**, not standalone solvers.

**Avoid:** per-seed ranges, defensive hedging, excessive `texttt`.

---

## 2. Introduction — *target 800–1000 words, 4 paragraphs, 8–12 citations*

**Goal.** Motivate fair comparison; expose the under-tuning gap; state PACE; list contributions.

- **Para 1 — Motivation (≈220 w).** MAX-CUT as a canonical NP-hard combinatorial problem [Karp 1972; Goemans–Williamson 1995], its role as the standard benchmark for Ising machines and physics-inspired hardware/algorithms (CIM, CAC, SB, digital annealers) [McMahon 2016; Inagaki 2016; Leleu 2021; Goto 2019/2021]. Why "time-to-quality" (anytime) matters for practical deployment.
- **Para 2 — Gap (≈260 w, cite 3–5).** Existing physics-vs-classical comparisons (a) tune the physics solver heavily but leave SA/PT/GA at literature-default parameters, (b) use inconsistent cut-scoring (e.g., unweighted internal counts on weighted instances), and (c) report final cut rather than the wall-clock trajectory. Cite Leleu 2021, Goto 2021, Tiunov/Hamerly-style CIM-vs-SA claims, Aramon/Mohseni reviews. State the methodological hole: **no uniformly-tuned, uniformly-scored, anytime comparison exists.**
- **Para 3 — Our approach (≈200 w).** Introduce **PACE**: per-instance Optuna tuning for *all six* solvers; one weighted-cut scorer applied to every solver's spin output; an anytime grid over each solver's principal compute knob; plus two composition mechanisms — warm-start hybrid (physics explorer → classical refiner) and parallel best-of-K portfolio. Mention the newly implemented memetic GA (grouping crossover + perturbed single-flip Tabu Search + DisQual pool update) as a strong, missing classical baseline.
- **Para 4 — Contributions (bullet list, 3–4 items, ≈140 w):**
  1. **A fairness-first anytime benchmark** in which every solver is per-instance Optuna-tuned and scored by an identical weighted-cut function — overturning the "SA is weak / CIM wins" narrative (SA K2000 1348→756; GA G22 →0).
  2. **A quantified honest hybrid story:** warm-starting TS/SA from CIM/CAC *rescues* physics solvers past their plateau (CIM K2000 794→120) but does **not** beat the best tuned single solver (GA gap 33).
  3. **A portfolio robustness result:** no single solver dominates all instances (GA on G22/K2000, SB on G55/G70), so a best-of-K portfolio is the oracle-free practical default.
  4. **Open, reproducible implementations** of all six solvers (Numba-JIT, trial-parallel) and the harness.

---

## 3. Related Work — *target 600–800 words, ≥15 references, 3 subsections*

**Goal.** Organize by sub-topic; end each subsection with how PACE differs.

- **3.1 Physics-inspired Ising solvers (≈250 w).** CIM hardware & models [Inoue–Yoshida 2022; McMahon 2016; Inagaki 2016; Hamerly 2019], CAC [Leleu 2021], Simulated Bifurcation [Goto 2019; Goto 2021], digital/quantum-inspired annealers [Aramon 2019; Mohseni 2022]. *Differs:* prior work tunes physics solvers but rarely tunes classical opponents to parity.
- **3.2 Classical heuristics for MAX-CUT (≈250 w).** Simulated annealing [Kirkpatrick 1983]; parallel tempering / isoenergetic cluster moves (PT-ICM) [Zhu–Ochoa–Katzgraber 2015]; memetic / Tabu / breakout local search [Wu–Hao 2012, 2013; Wu–Wang–Lü 2015; Benlic–Hao 2013]; GRASP/path-relinking [Festa et al.]. *Differs:* we re-tune these with Optuna instead of using fixed recommended values, which materially changes rankings.
- **3.3 Benchmarking methodology & hybrids (≈200 w).** Anytime/time-to-target evaluation, algorithm portfolios [Gomes–Selman], warm-start and memetic hybridization, fairness pitfalls in solver comparison. *Differs:* PACE unifies per-instance tuning + identical weighted scoring + anytime trajectory + hybrid/portfolio composition in one harness.

---

## 4. Method — *target 1000–1500 words, flowing prose*

**Goal.** Formal problem, six solvers, fairness protocol, unified scoring, hybrid & portfolio construction.

- **4.1 Problem formulation (≈180 w).** Graph $G=(V,E,w)$, spin vector $s\in\{-1,+1\}^N$, weighted cut $C(s)=\tfrac12\sum_{(i,j)\in E} w_{ij}(1-s_i s_j)$; objective $\max_s C(s)$; equivalence to Ising energy minimization. Define gap $=\text{BKS}-C$ and relative gap.
- **4.2 Solvers (≈420 w, narrative).** CIM (Inoue–Yoshida traveling-wave: κ, L, γ, η, ΔP, coupling J); CAC (Leleu 2021 error-variable amplitude control — **note** internal unweighted counting, re-scored externally); SA (exponential cooling + `simulate_sa_warm`); SB (Goto dSB variant); PT-ICM (temperature ladder + isoenergetic cluster moves); **memetic GA** (grouping crossover, perturbed single-flip TS with dynamic tenure + aspiration, DisQual pool update, O(deg) incremental move gain $\Delta_v=\sum_{u\in N(v),s_u=s_v}w_{vu}-\sum_{u\in N(v),s_u\neq s_v}w_{vu}$). Note all are Numba-JIT, `prange` trial-parallel.
- **4.3 Fairness protocol — per-instance Optuna tuning (≈300 w).** TPESampler; objective = unified weighted mean cut; tuning budgets (CIM/CAC: G22 long search 250–1000 trials, others 25–30; classical SA/SB/PT/GA tuned over cooling temp / variant·dt·a0 / ladder size·range·swap·ICM interval / pop·TS-iters·cr·tenure·β). State *why this is the core methodological move*: fixed defaults under-tune SA & PT-ICM, producing the misleading "physics wins" story.
- **4.4 Unified anytime scoring (≈200 w).** Geometric grid over each solver's principal compute knob (rounds/steps/iters/sweeps/generations); `num_trials=16` fixed-seed batches; record batch wall-clock $t$ and weighted cut via `ctx.score()`; plot $x=\log t$, $y=\max$ cut; JIT warmup excluded; adaptive cutoff past time cap.
- **4.5 Composition (≈200 w).** **Warm-start hybrid:** explorer ∈ {CIM, CAC} for short fixed budget → spin seed → refiner ∈ {TS, warm-start SA}; total time = explorer + refiner; compared against cold-start at matched refinement budget. **Parallel portfolio:** per-time best-of-K envelope across all tuned solvers; plus the K-split time-sharing variant as a single-core baseline.
- Include **Algorithm 1** (PACE harness: tune → anytime sweep → score) and **Algorithm 2** (warm-start hybrid) as `algorithm` environments.

---

## 5. Experiments — *target 800–1200 words*

**Goal.** Setup as subsection; instance table; hyperparameter table (Table 1); reference figures; cite baseline papers.

- **5.1 Setup (≈250 w).** Hardware note (`NUMBA_NUM_THREADS=4` uniform across solvers); Optuna config; 16-trial fixed-seed batches; weighted-cut scorer; BKS values as literature references (gap-to-BKS, not certified optima).
- **5.2 Instances — Table (benchmark instances).**

  | Dataset | N | edges | weights | BKS | type |
  |---|---|---|---|---|---|
  | G22 | 2000 | 19990 | +1 | 13359 | sparse unweighted (G-set) |
  | K2000 | 2000 | 1999000 | ±1 | 33337 | dense weighted (SK) |
  | G55 | 5000 | 12498 | +1 | 10299 | sparse large |
  | G70 | 10000 | 9999 | +1 | 9591 | sparse largest |
- **5.3 Tuned hyperparameters — Table 1** (per-solver, per-instance tuned ranges/values; cite each baseline paper here: SA [Kirkpatrick 1983], SB [Goto 2019/2021], PT-ICM [Zhu 2015], GA [Wu–Hao 2013], CIM [Inoue–Yoshida 2022], CAC [Leleu 2021]).
- **5.4 Protocol notes (≈150 w).** G22 physics params do not transfer (untuned CIM collapses on K2000, max≈4287/33337) → per-instance tuning mandatory; low-budget tuning on dense K2000 can hurt PT-ICM (honest caveat). Reference figures: *"As shown in Figure 1 (G22 anytime), ..."* (`figs/G22_anytime.png`), summary figs (`figs/{K2000,G70,G55,G22}_summary.png`).

---

## 6. Results — *target 600–800 words; do NOT repeat Experiments numbers verbatim where avoidable*

**Goal.** Main tables + ablation (tuning effect) + analysis tying numbers to insights; reference figures.

- **6.1 Single-solver anytime (≈220 w).** Main table per instance (gap):
  - G22: GA 0, CAC/SA/SB 1, PT 2, CIM 17 (`figs/G22_anytime.png`).
  - K2000: GA 33, SB 59, SA 756, CIM 794, CAC 973, PT 1702.
  - G55: SB 15, PT 25, CAC 30, SA 43, CIM 74, GA 78.
  - G70: SB 42, PT 51, CIM 90, SA 97, CAC 108, GA 109.
  - **Insight:** no single winner (GA on G22/K2000; SB on G55/G70); SB most reliable (mean gap ≈0.20%); CIM rises fast then plateaus; physics solvers far from BKS on dense K2000.
- **6.2 Ablation — the effect of fair tuning (≈170 w).** Default → tuned: SA K2000 1348→756, G22 5→1; PT-ICM G22 36→2; GA G22 1→0, K2000 57→33, G70 253→109. *This is the paper's pivotal ablation:* fairness reorders the leaderboard and dissolves the "SA is weak" claim.
- **6.3 Warm-start hybrid (≈230 w).** G22: cold TS gap 34 → CIM→TS / CAC→TS gap 1 (`figs/G22_TS_warm_vs_cold.png`); SA cold 5 → CAC→SA 2. K2000: CIM 794 → **CIM→TS 120**, CAC→TS 244 (`figs/K2000_TS_warm_vs_cold.png`); seed quality dominates (CIM seed ≈ half the CAC-seed gap). **But** best tuned single solver (GA 33) still wins. G55/G70: hybrid ≈ physics single, below SB — rescue is instance-dependent.
- **6.4 Portfolio (≈120 w).** Best-of-K parallel envelope = best single per instance (G22 gap 0, K2000 33, G55 15, G70 42); time-split single-core variant is K× slower and loses (`figs/{G22,G55}_summary.png`). Robustness without an oracle.

---

## 7. Discussion — *target 400–600 words; cite prior work here*

**Goal.** Reconcile with prior claims; explain surprises; broader implications.

- Why physics-vs-classical "advantage" shrinks under fair tuning — contrast with Leleu 2021 / Goto 2021 framing.
- Why CIM/CAC plateau on dense SK K2000 (amplitude heterogeneity, weighted coupling) yet seed good basins for local search — the *explorer* role.
- Why warm-start helps most where the physics solver is far from a refined basin (rescue) but is redundant where classical solvers already reach near-BKS quickly.
- Implication for hardware Ising machines: value as **front-end explorers / portfolio members**, not standalone final solvers; tuning-parity must be a reporting standard.

## 8. Limitations — *target 200–300 words; ALL caveats here (3–5 concrete)*

1. Public aggregates report only the per-batch best cut (order statistic); run-to-run dispersion and significance tests are not recorded — few-cut gaps are within stochastic variation (descriptive effect sizes).
2. No fine-grained wall-clock instrumentation (CPU model, per-solver timing) or cold-start refiner controls beyond those shown.
3. Single-budget Optuna tuning is not optimal across the whole anytime grid (K2000 PT-ICM regressed under low-budget tuning).
4. BKS are literature references, not certified optima; CIM/CAC conclusions hold for these specific implementations under 25–30-trial budgets.
5. CAC's weighted-cut handling is external re-scoring; native weighted internal selection is future work.

## 9. Conclusion — *target ~120 words*

- **Summary (2–3 sentences):** Under PACE's fair, uniformly-scored, per-instance-tuned anytime comparison, tuned classical heuristics (GA, SB) lead, no solver dominates all instances, and physics-inspired CIM/CAC are best as hybrid explorers and portfolio members rather than standalone solvers.
- **Future work (2–3 sentences):** native weighted CAC, automatic per-instance CIM scaling, learned explorer/refiner budget allocation, and dispersion/significance reporting with full wall-clock instrumentation.

---

## 10. Evidence ↔ Section Map (for the writer)

| Claim | Numbers | Figure(s) | Section |
|---|---|---|---|
| No single winner | GA G22/K2000; SB G55/G70 | `G22_anytime`, summaries | 6.1, 6.4 |
| Fair tuning reorders ranks | SA 1348→756; PT 36→2; GA →0 | — | 6.2 |
| Hybrid = rescue, not domination | CIM 794→120; GA 33 still best | `K2000_TS_warm_vs_cold` | 6.3 |
| Seed quality dominates | CIM 120 vs CAC 244 (K2000) | `G22_TS_warm_vs_cold` | 6.3 |
| Portfolio robustness | gaps 0/33/15/42 | `G22_summary`, `G55_summary` | 6.4 |

## 11. Reference seed list (expand to ≥18 for submission)
Leleu 2021 (CAC); Inoue–Yoshida 2022 (CIM); Goto 2019 & 2021 (SB); Zhu–Ochoa–Katzgraber 2015 (PT-ICM); Wu–Hao 2012/2013 & Wu–Wang–Lü 2015 (memetic/TS); Kirkpatrick 1983 (SA); Karp 1972; Goemans–Williamson 1995; McMahon 2016; Inagaki 2016; Hamerly 2019; Aramon 2019; Mohseni 2022; Benlic–Hao 2013; Festa et al. (GRASP); Gomes–Selman (portfolios).

---

**Note on data integrity (per prior-run lesson):** every quantitative claim in the outline above is drawn from the supplied result tables (§6–§8 of the analysis); the writer must keep these grounded values and must not export if the verified registry is empty.