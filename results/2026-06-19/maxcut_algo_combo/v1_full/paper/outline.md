# Paper Outline: WARP

**Method name:** **WARP** — **W**arm-start **A**nytime **R**efinement **P**ortfolio
(4 characters; encodes all three contributions — the warm-start hybrid, the wall-clock *anytime* evaluation axis, the *refinement* division of labor, and the *portfolio* robustness story.)

---

## Candidate Titles (MethodName: Subtitle, ≤14 words)

| # | Title | Words | Memorability | Specificity | Novelty signal |
|---|-------|:-:|:-:|:-:|:-:|
| **T1** | **WARP: Warm-Starting Physics-Inspired Ising Solvers for Anytime MAX-CUT Optimization** | 9 | **5** | **5** | 4 |
| **T2** | **WARP: When Physics Solvers Should Hand Off to Classical Local Search** | 11 | 4 | 4 | **5** |
| **T3** | **WARP: A Fair Wall-Clock Anytime Comparison of Hybrid and Portfolio MAX-CUT Solvers** | 12 | 4 | **5** | 3 |

**Recommendation: T1.** It names the method, signals the headline mechanism (warm-start), keeps both the physics-inspired and classical sides visible, and anchors the contribution in the *anytime* framing that distinguishes this paper from time-to-target comparisons. T2 is the strongest "declarative surprise" backup if a punchier framing is wanted. T3 is the most descriptive but reads most survey-like (lower novelty signal).

**Rating rubric:** memorability = how easily recalled/spoken; specificity = how precisely it conveys the actual content; novelty signal = how strongly it implies a new finding rather than a benchmark report.

---

## Global Targets

- **Total length:** ~7,500–8,500 words (main body), excl. references/appendix — fits an 8-page + appendix ML venue format.
- **Figures:** 5 (G22 anytime; G22 TS warm-vs-cold; K2000 summary panel; G70 summary panel; G22/G55 portfolio summary). All already produced under `figs/`.
- **Tables:** 4 core (instances; hyperparameters; single-solver anytime; hybrid warm-vs-cold) + 1 ablation/portfolio.
- **Citations:** ≥35 unique; ≥8–12 in Introduction, ≥15 in Related Work.
- **Reusable artifacts to cite throughout:** `modules/{CIM,CAC,SA,SB,PT_ICM,GA}.py`, harness `scripts/benchmarks/*`, tuning `scripts/tuning/tune_anytime_params.py`, `results/anytime_tuned_params.json`.

---

## Section-by-Section Plan

### Abstract — 200 words (PMR+)
**Goal:** State the gap (no fair wall-clock anytime comparison of physics-inspired vs. classical MAX-CUT solvers; hybrid value unquantified), name WARP by S3, deliver ≥3 quantitative claims.

- **S1–S2 (Problem):** CIM/CAC are promising MAX-CUT solvers, but comparisons to classical heuristics rarely fix a *unified wall-clock axis* with *identical scoring*, and the value of combining physics + classical search is not quantified.
- **S3–S4 (Method):** Introduce **WARP** — a unified anytime benchmark across six tuned solvers, plus a warm-start hybrid (physics explorer → classical refiner) and a parallel portfolio.
- **S5–S6 (Results, concrete numbers):**
  - Single solvers: classical SB/GA/SA reach within **0.01%** of BKS in **< 1 s** on G22; CIM plateaus at gap 17 and collapses untuned on dense K2000 (**4287 / 33337**).
  - Hybrid: **CIM→TS breaks CIM's 13342 plateau to reach BKS−1 (13358) in ≈0.22 s**; on K2000 it cuts CIM's gap from **794 → 120**; at the smallest refine budget warm-start adds **+250 mean cut** over cold-start.
  - Portfolio: no solver wins all four datasets (SB on 3, PT-ICM on G70) → robustness argument.

**Evidence:** §6.1 G22 table; §6.2 K2000 table; §7.1/§7.2 hybrid tables; §8 portfolio table.
**Avoid:** per-seed ranges, defensive hedging, excessive `texttt`.

---

### 1. Introduction — 850–1000 words, 4 paragraphs, cite 8–12
**Goal:** Motivate the wall-clock anytime question and the hybrid hypothesis; end with a crisp contribution list.

- **¶1 Motivation:** MAX-CUT as a canonical NP-hard Ising problem; Ising machines (CIM, SB) and physics-inspired dynamics as a fast-growing solver class. Cite Goto (SB), Inoue–Yoshida (CIM), Leleu (CAC), McMahon/Hamerly CIM hardware lineage.
- **¶2 Gap (cite 3–5):** Most physics-machine evaluations report time-to-solution at fixed quality or use solver-internal (often unweighted) cut counts, making cross-solver comparison apples-to-oranges; classical baselines are frequently under-tuned; the *combination* of physics + classical local search is asserted but not budget-matched. Cite Tiunov, Kalinin–Berloff, Mohseni et al. (Ising machine review), and at least one tabu/memetic MAX-CUT reference (Wu & Hao).
- **¶3 Approach:** WARP fixes (i) a *unified weighted-cut score* applied to every solver's returned spins, (ii) a *wall-clock anytime axis*, (iii) per-dataset hyperparameter tuning so each solver is shown near its best, and (iv) two combination strategies — warm-start hybrid and parallel portfolio. State the central hypothesis: a short physics run is a *better-than-random global initializer* whose plateau is broken by a classical refiner.
- **¶4 Contributions (bullet list of 4):**
  1. A reproducible, identically-scored, wall-clock **anytime benchmark** of six tuned solvers (CIM, CAC, SA, SB, PT-ICM, memetic GA) on four MAX-CUT instances spanning sparse/dense, weighted/unweighted, N=2000→10000.
  2. A **new from-scratch memetic GA** (grouping crossover + perturbed single-flip Tabu Search + DisQual pool) as a strong, Numba-JIT classical baseline (`modules/GA.py`).
  3. The **warm-start hybrid finding**: physics-explorer → classical-refiner dominates cold-start at matched budget and breaks CIM's plateau (CIM→TS reaches BKS−1 on G22; gap 794→120 on K2000).
  4. A **portfolio analysis** quantifying that no solver dominates, motivating parallel portfolios for robustness.

**Evidence links:** Tables §2, §6, §7, §8; figs `G22_anytime.png`, `G22_TS_warm_vs_cold.png`.

---

### 2. Related Work — 700 words, ≥15 refs, 3 subsections
**Goal:** Organize by sub-topic; end each with how WARP differs.

- **2.1 Physics-inspired & Ising-machine solvers (CIM, CAC, SB).** Coherent Ising Machines (Inoue–Yoshida traveling-wave; McMahon; Hamerly), Chaotic Amplitude Control (Leleu 2021), Simulated Bifurcation / dSB (Goto 2019, 2021). *Differs:* we treat these as software solvers on a common wall-clock axis with uniform scoring rather than reporting machine-native metrics.
- **2.2 Classical heuristics & metaheuristics for MAX-CUT.** Simulated annealing, parallel tempering + isoenergetic cluster moves (Zhu–Ochoa–Katzgraber), tabu/memetic MAX-CUT (Wu & Hao 2012/2013; Wu–Wang–Lü 2015), Goemans–Williamson SDP as a quality reference. *Differs:* we tune these as first-class competitors, not strawmen, and add a fresh memetic GA implementation.
- **2.3 Warm-starting, hybridization, and algorithm portfolios.** Warm-started quantum/classical optimization (Egger et al. warm-start QAOA), hybrid global-then-local schemes, algorithm-selection and portfolio theory (SATzilla lineage; per-instance selection). *Differs:* we provide a *budget-matched* warm-vs-cold comparison on a wall-clock axis and an *anytime envelope* portfolio.

---

### 3. Method — 1200–1450 words, flowing prose + 1 algorithm box
**Goal:** Technical formulation of the unified evaluation and the WARP combination strategies — not a workflow log.

- **3.1 Problem formulation.** MAX-CUT objective on weighted graph $G=(V,E,w)$: maximize $\frac14\sum_{ij}w_{ij}(1-s_is_j)$, $s\in\{\pm1\}^n$; equivalence to Ising energy minimization. Define **move gain** $\Delta_v=\sum_{u\in N(v),s_u=s_v}w_{vu}-\sum_{u\in N(v),s_u\neq s_v}w_{vu}$ used by TS/GA, with O(deg) incremental updates.
- **3.2 Unified anytime evaluation protocol.** Each solver's complexity knob (CIM rounds, CAC steps, SA iters, SB steps, PT sweeps, GA generations) swept on a geometric grid; `num_trials=16` fixed-seed batches per point; record batch wall-clock $t$ and cut statistics under **one shared weighted score** `ctx.score()` recomputed from returned spin signs; JIT warm-up excluded; adaptive truncation past time cap. *Stress that uniform scoring neutralizes CAC's unweighted internal counting.*
- **3.3 Solvers (brief, with implementation anchors).** One sentence each: CIM (Inoue–Yoshida), CAC (Leleu), SA (+`simulate_sa_warm`), dSB, PT-ICM, and the **new memetic GA** (grouping crossover, perturbed single-flip TS with dynamic tenure/aspiration, DisQual pool update). Cite the corresponding `modules/*.py`.
- **3.4 WARP-Hybrid (warm-start).** Run a physics explorer ∈ {CIM, CAC} for a short fixed budget → spin configuration $s_0$ → seed a refiner ∈ {TS, warm-start SA}; total time = explorer + refiner; compared to random-init (cold) at *matched refine budget*. **Algorithm box: `WARP-Hybrid(explorer, refiner, budgets)`.**
- **3.5 WARP-Portfolio.** From single-solver anytime curves, construct the per-time best-of-K envelope (K-core parallel) and a 1-core time-sliced variant; define the robustness criterion (track the per-instance winner without knowing it in advance).

**Evidence:** equation from §3; protocol from §5; algorithm from §7 description.

---

### 4. Experimental Setup — 500 words (subsection-style)
**Goal:** Datasets, baselines, metrics, hardware, tuning — with **Table 1 (hyperparameters)**.

- **Datasets — Table (instances):** G22 (sparse unweighted, N=2000, BKS 13359), K2000 (dense weighted SK, BKS 33337), G55 (sparse N=5000, BKS 10299), G70 (sparse N=10000, BKS 9591). *(from §2)*
- **Baselines:** all six solvers; cite each baseline's source paper (not just name it).
- **Metrics:** reached cut, absolute gap to BKS, gap%, time-to-0.5%-of-BKS; all on uniform weighted score. Note BKS are literature references (gap, not optimality proofs).
- **Hardware/threads:** `NUMBA_NUM_THREADS=4` for all solvers (fair wall-clock); Numba JIT; trial-parallel `prange`.
- **Tuning — Table 1:** G22 uses prior Optuna results (CIM 1000-trial best 13307; CAC 250-trial best 13336); K2000/G55/G70 re-tune CIM/CAC via Optuna TPE (25–30 trials, objective = uniform weighted mean cut); SA/SB/PT/GA use literature-recommended instance-adaptive settings (SB auto_c0, PT geometric ladder). Tuned per-dataset means listed.

**Evidence:** §2 table, §4 tuning tables, `results/anytime_tuned_params.json`.

---

### 5. Results — 700–800 words (do NOT repeat setup numbers verbatim; analyze)
**Goal:** Three result blocks, each a table + analysis paragraph tying numbers to insight; reference figures.

- **5.1 Single-solver anytime (Table: G22 + cross-dataset).** "As shown in Figure 1 (`G22_anytime.png`), SB and SA reach BKS-neighborhood in 0.1–1 s; CIM rises fastest but plateaus at gap 17." Cross-dataset: SB most consistent (gap 1/45/12/53); **no solver dominates** (PT-ICM best on G70; GA degrades at N=10000, gap 253, due to O(n) single-flip scans); physics solvers far from BKS on dense K2000 (gap ~800–970).
- **5.2 Hybrid warm-vs-cold (Table: G22 warm/cold).** Reference `G22_TS_warm_vs_cold.png`: CIM→TS reaches 13358 (gap 1) in ≈0.22 s vs CIM-alone 13342; at ts_iters=2000, cold TS mean 13055 vs CIM→TS 13311, CAC→TS 13341 (**+250 mean**); TS a stronger refiner than SA. Cross-dataset (Table): K2000 CIM gap 794→120; hybrid helps most when the physics solver is far from BKS, not when single solvers already near BKS (G55).
- **5.3 Portfolio (Table: parallel-PF per dataset).** Reference `G22_summary.png` / `G55_summary.png` / `G70_summary.png`: parallel PF tracks the per-time best (gap 1/45/12/51), the winner changes by dataset; time-sliced PF is K× slower and loses to a single good solver.

**Evidence:** §6, §7, §8 tables; figs `G22_anytime.png`, `G22_TS_warm_vs_cold.png`, `K2000_summary.png`, `G70_summary.png`, `G22_summary.png`, `G55_summary.png`.

---

### 6. Ablation / Analysis — 350–450 words (1 table)
**Goal:** Isolate the *mechanism* of the warm-start gain.

- Vary refine budget (ts_iters geometric grid): show warm−cold gap shrinks as budget grows → warm-start's value is **front-loaded** (anytime advantage), consistent with "good basin handed off early."
- Explorer choice (CIM vs CAC) × refiner (TS vs SA): TS dominates SA as refiner; CAC→TS best mean on G22, CIM→TS best on far-from-BKS K2000.
- Negative/boundary result: on already-near-BKS instances (G55) a short explorer budget cannot beat the best single solver — delineates *when* WARP-Hybrid helps.

**Evidence:** §7.1 budget rows, §7.2 cross-dataset table.

---

### 7. Discussion — 450–550 words (cite papers here)
**Goal:** Interpret against prior work and broader implications.

- Reconcile with warm-start QAOA / hybrid optimization literature: WARP gives a *budget-matched, wall-clock* confirmation that physics dynamics are valuable as **global initializers**, less so as final refiners.
- Why CIM plateaus: amplitude-heterogeneity / lack of post-projection local moves; why classical TS escapes it. Why CAC suffers on weighted K2000 (unweighted internal selection) — an implementation, not conceptual, limit.
- Portfolio/algorithm-selection implication: per-instance winner unpredictability argues for parallel portfolios as a default given multi-core hardware.

---

### 8. Limitations — 250 words (ALL caveats here)
1. BKS are literature references; we report gap, not optimality proofs.
2. Conclusions are about *these* CIM (Inoue–Yoshida) and CAC (Leleu 2021) implementations under stated tuning budgets, not all CIM/CAC.
3. CAC K2000 reflects a documented unweighted-internal-selection limitation (mitigated by uniform re-scoring).
4. Four instances; tuning budgets modest (25–30 Optuna trials for re-tuned physics solvers).
5. Wall-clock is implementation/JIT/thread-count dependent; not hardware-Ising-machine timings.

---

### 9. Conclusion — 120 words
Summarize (2–3 sentences): classical heuristics lead in single-solver wall-clock efficiency, but a short physics run warm-starting a classical refiner breaks the physics plateau and dominates cold-start at matched budget; portfolios add robustness since no solver wins everywhere. Future work (2–3 sentences): weighted-aware CAC selection, theoretically-derived per-instance CIM scaling, and automatic explorer/refiner budget allocation.

---

### References — ≥35
Seed set from analysis: Leleu 2021 (Comm. Phys.), Inoue–Yoshida 2022 (Opt. Comm.), Goto 2019/2021 (Sci. Adv.), Zhu–Ochoa–Katzgraber 2015 (PRL), Wu–Hao 2012/2013, Wu–Wang–Lü 2015. **Add for venue bar:** Mohseni et al. Ising-machine review, McMahon/Hamerly CIM, Kalinin–Berloff, Tiunov CIM simulation, Egger et al. warm-start QAOA, Goemans–Williamson SDP, SATzilla/algorithm-portfolio, Boixo/Rønnow time-to-solution methodology.

---

## Figure/Table → Section Map (evidence index)

| Asset | Source (§) | Used in |
|---|---|---|
| Instances table | §2 | Setup (4) |
| Hyperparameter/tuning table | §4 | Setup (4), Table 1 |
| G22 single anytime table + `G22_anytime.png` | §6.1 | Results 5.1, Fig 1 |
| K2000/G55/G70 single tables + `K2000_summary.png`, `G70_summary.png` | §6.2 | Results 5.1 |
| G22 warm-vs-cold table + `G22_TS_warm_vs_cold.png` | §7.1 | Results 5.2, Ablation |
| Cross-dataset hybrid table + `K2000_TS_warm_vs_cold.png` | §7.2 | Results 5.2 |
| Portfolio table + `G22_summary.png`, `G55_summary.png` | §8 | Results 5.3 |

**One-line positioning sentence (for intro/abstract reuse):** *WARP shows that the right way to use a physics-inspired MAX-CUT solver is not as a standalone optimizer but as a fast global initializer handed off to a classical refiner — quantified on a fair wall-clock anytime axis, and made robust by a parallel portfolio.*