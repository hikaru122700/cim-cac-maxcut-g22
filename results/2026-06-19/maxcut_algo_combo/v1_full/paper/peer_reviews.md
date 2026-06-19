# Peer Review: *PACE: When Tuned Heuristics Outrun Ising Machines on MAX-CUT*

**Recommendation summary:** Major revision required. The empirical numbers in the tables are internally consistent and faithfully match the provided run evidence, the topic is well-scoped, and citation distribution is genuinely good. However, the paper's *central methodological contribution* — an **anytime, wall-clock-resolved** comparison — is not supported by any timing evidence (`elapsed_sec` is `null` in all four run files), and the experimental scale claimed in the text (16-trial batches; 25–1000 Optuna trials) is contradicted by the evidence, which records the experiment as executed **4 times (once per instance)** with single scalar outcomes and no dispersion. These two issues are disqualifying in their current form and must be resolved before the headline claims can stand.

---

## Reviewer A — Methodology Expert

### Strengths
- **The framing is correct and valuable.** "Tune every contender, score them with one identical functional, compare on a shared axis" is exactly the right correction to the asymmetric protocols common in this literature. The diagnosis in the Introduction (physics tuned, classical at defaults; weight-blind spin counters; final-cut-only reporting) is precise and well-argued.
- **The unified weighted-cut scorer is a real methodological asset.** Re-scoring CAC's spin output with the same `C(·)` on signed K2000 is the kind of invariant that prevents the exact distortion the paper criticizes, and it is described concretely (Section: Problem formulation).
- **Composition design is clean.** Algorithm 2's matched-budget cold-start control isolates *seed value* from *extra compute*, which is the correct way to evaluate warm-starting. This is better than most hybrid studies.

### Weaknesses
1. **The anytime axis — the paper's stated core — has no supporting data (CRITICAL).** Abstract, Introduction (contribution 1), Method ("Unified anytime scoring"), and Figure 2 all rest on wall-clock trajectories. Every run file reports `"elapsed_sec": null`. There is no timing instrumentation anywhere in the evidence. Figure 2 ("anytime curves … horizontal axis is elapsed wall-clock time") therefore cannot be generated from the data provided. A "fairness-first *anytime* evaluation framework" with zero time measurements is not anytime — it is final-cut reporting with anytime *language*.
2. **Claimed experimental scale contradicts the evidence (CRITICAL).** The text claims a "fixed-seed batch of 16 parallel trials" and Optuna budgets of "25–30" and "250–1000" trials. The evidence states the experiment was executed **4 times** and contains four JSON files of single scalar metrics. Either the 16-trial/Optuna machinery did not run as described, or only one summary value per (solver, instance) survived. As written, the Method over-describes a protocol the evidence cannot confirm.
3. **Contribution 4 is unsupported.** "We quantify its advantage over single-core time-sharing" appears nowhere in Results — there is no time-sharing baseline table or figure. The portfolio row is, by construction, the per-instance min over solvers; presented this way it is a *definition*, not an empirical finding, and certainly not a quantified speedup over time-sharing.
4. **The fairness narrative may be self-undermined by the GA.** A properly tuned memetic GA (Wu & Hao) is state-of-the-art on G-set, yet here it is the **worst** solver on G55 (gap 78) and G70 (gap 109). The most likely explanation is that 25–30 Optuna trials under-tune the GA on large graphs — i.e., the paper commits, on its own flagship baseline, the very under-tuning asymmetry it sets out to eliminate. This needs to be confronted, not glossed.

### Actionable revisions
- Either (a) instrument and report wall-clock per knob value and produce a genuine Figure 2, or (b) retitle/reframe the paper around *tuned final-cut* comparison and remove all "anytime/wall-clock/time-resolved" claims (Abstract, contribution 1, Method, Section "Protocol notes," Figure 2). Option (b) is honest and still publishable; option (a) is stronger but requires re-running with timing.
- Reconcile the trial-count language with what was actually run. If only one batch summary exists per instance, say so plainly and downgrade "16-trial batch" claims accordingly.
- Delete contribution 4 or add the missing time-sharing-vs-portfolio measurement that quantifies it.
- Add a tuning-budget sensitivity check for the GA (and PT-ICM) on G55/G70/K2000; otherwise soften "strong, reproducible memetic baseline" to reflect that it leads only on G22/K2000 and trails on large sparse graphs.

---

## Reviewer B — Domain Expert (Ising machines & MAX-CUT)

### Strengths
- **Solver descriptions are technically faithful.** The CIM traveling-wave / measurement-feedback account, CAC error-variable dynamics ($\dot e_i=-\beta(x_i^2-a)e_i$), dSB momentum/position updates with inelastic walls, and PT-ICM's replica + isoenergetic cluster moves are all correctly characterized and correctly attributed (Inoue–Yoshida, Leleu, Goto, Zhu).
- **BKS reference values are correct** (G22 = 13359, K2000 = 33337, G55 = 10299, G70 = 9591) and match the standard literature, so the gaps are meaningful to readers in the field.
- **The regime story is the right one.** CIM/CAC degrading on dense signed SK coupling while dSB stays robust, and PT-ICM collapsing on a million-edge instance under a small tuning budget, are both consistent with known behavior and well-motivated physically (uniform pump schedule vs. amplitude heterogeneity).
- **Strong, well-placed citations in Method/Experiments/Discussion**, not just Intro/Related Work — this is exactly what is wanted and is a real strength relative to typical submissions.

### Weaknesses
1. **The GA result is implausible for a "competitive classical opponent."** Being the worst solver on two of three G-set graphs contradicts the entire premise that this GA supplies "the strong missing classical baseline." Domain readers will immediately suspect under-tuning or an implementation defect (e.g., crossover/Tabu tenure not scaling to N=5000/10000). This must be diagnosed; otherwise the GA both over-claims (G22 zero gap headlined) and under-delivers (G55/G70).
2. **K2000 is one instance, not a distribution.** Calling it "the dense weighted instance" and generalizing to "the dense Sherrington–Kirkpatrick regime" overreaches from a single graph. SK conclusions need several K-instances or several SK seeds.
3. **PT-ICM's K2000 failure (gap 1702) is reported but quietly excused.** The paper attributes it to "low per-instance tuning budget." If the budget is too low to make PT-ICM competitive on dense instances, then the "parity" claim does not hold uniformly across solvers, and PT-ICM's number should not be presented as a fair representation of the method.
4. **Single-instance generalizations elsewhere.** "Reaches the best-known cut exactly on a standard G-set instance" (Method, contributions) is true only for G22 and is cherry-picked given the G55/G70 results.

### Actionable revisions
- Investigate and report why the memetic GA underperforms on G55/G70; if it is budget, raise it and re-report; if implementation, fix and re-report. State the resolution explicitly.
- Add at least 2–3 additional dense/SK instances (or SK seeds) before generalizing K2000 findings to "the dense regime."
- Either bring PT-ICM to genuine parity on K2000 or annotate its row as budget-limited and exclude it from any "fair ranking" claim on dense instances.
- Qualify the memetic-baseline contribution to match the data: lead on G22/K2000, weak on large sparse — and explain why.

---

## Reviewer C — Statistics / Rigor Expert

### Strengths
- **Arithmetic is correct and reproducible from the evidence.** I verified every cell: Table 1 absolute gaps match all four JSON files exactly; mean relative gaps (CIM 1.042, CAC 1.086, SA 0.926, dSB 0.192, PT-ICM 1.474, GA 0.498, Portfolio 0.171) all recompute correctly; Table 2 sparse/dense splits, Table 3 head-to-head $\Delta$ values, and Table 4 hybrid gaps all check out. The ~85% rescue figure ((794−120)/794 = 84.9%) is accurate.
- **The Limitations section is unusually candid** about the order-statistic issue and the absence of dispersion/significance tests, which is to the authors' credit.

### Weaknesses
1. **n = 1 per condition; no error bars, no CIs, no significance tests (CRITICAL).** Each (solver, instance) is a single reported value. The headline "GA reaches zero gap on G22" is a **1-cut** lead over CAC/SA/dSB (all gap 1), from a single run. On a near-saturated instance this is indistinguishable from noise. No "win" decided by 0–2 cuts (most of the G22 column; G55/G70 ties) is statistically defensible as stated.
2. **"Best of 16-trial batch" (max) is the wrong summary statistic for fair comparison.** The maximum is an extreme-value statistic that rewards high-variance solvers and is highly sensitive to batch size — yet the batch size and per-seed spread are exactly what the evidence does not contain. Mean/median with a dispersion measure across multiple seeds is required for any ranking claim.
3. **Significance tests are claimed-by-absence, not performed.** Table 3 explicitly says "p-values are not estimated." That is honest, but the Abstract/Discussion still use comparative verbs ("dominate," "leads," "most reliable") that imply established differences. Effect-size language without dispersion cannot license those verbs.
4. **Reproducibility gaps.** "Fixed-seed" is stated but the actual seed values are not given; hardware is only "a multi-core CPU" (no model/clock/core count); resulting tuned hyperparameter **values** are absent (Table 1 lists *which* parameters, not their tuned settings); and compute time is unrecorded (`elapsed_sec: null`). A reader cannot reproduce these numbers from the paper.

### Actionable revisions
- Run each (solver, instance) for **≥10–20 independent seeds** and report mean ± std (or median + IQR) and a 95% CI; reserve "best" as a secondary line.
- Apply paired nonparametric tests across seeds (Wilcoxon signed-rank per instance; Friedman + Nemenyi across instances for the ranking) and report effect sizes with the tests, not instead of them.
- Until dispersion exists, demote all ≤2-cut differences to "tied within noise" and remove ranking verbs for them (especially the G22 GA "exact win" headline and the G55/G70 ties).
- Add a reproducibility appendix: seed list, exact CPU, NUMBA thread count (stated: 4), Optuna sampler version, and the **tuned hyperparameter values** per (solver, instance).

---

## Consolidated checklist (the 8 required checks)

1. **Topic alignment — PASS.** The paper stays squarely on physics-inspired (CIM/CAC) vs. tuned classical (SA/SB/PT-ICM/memetic GA) fair Optuna comparison, warm-start hybrids, and parallel portfolio. No drift; environment/measurement gaps are placed in Limitations rather than dressed up as contributions — good.

2. **Claim–evidence alignment — MIXED, with critical gaps.** Supported: GA zero gap on G22 ✓; GA leads K2000 (33) ✓; dSB mean ≈0.192% ✓; ~85% CIM→TS rescue on K2000 ✓; hybrids never beat best single solver ✓; "no single winner" ✓. **Unsupported:** all *anytime/wall-clock* claims (no timing data); "16-trial batch / 25–1000 Optuna trials" (evidence = 4 runs, single values); contribution 4 "quantify advantage over single-core time-sharing" (no such measurement); "strong memetic baseline" partly contradicted by worst-on-G55/G70.

3. **Statistical validity — FAIL.** n=1 per condition, no error bars/CIs, no significance tests, "best-of-batch" max as headline metric, and wins decided by 0–2 cuts. The paper acknowledges this but still uses ranking verbs.

4. **Completeness — BELOW TARGET.** Conclusion 124 words (target 200–300, severely under); Introduction 754 (under 800–1000); Related Work 539 (under 600–800); Results 1094 (over 600–800); overall body likely below the 5,000–6,500-word NeurIPS expectation. Rebalance: trim Results prose, expand Conclusion, Intro, Related Work.

5. **Reproducibility — INSUFFICIENT.** Threading (NUMBA=4) and sampler (TPE) given, but no seed values, no concrete hardware, no tuned hyperparameter values, no compute time. Add a reproducibility appendix.

6. **Writing quality — NEEDS WORK.** Body sections (Method/Results/Discussion) are in flowing prose — good, no bullet lists there. However the Introduction "contributions" are a bullet list (acceptable at many venues but flag for conversion if the venue requires prose). **High weasel-word count (23):** "roughly," "about," "markedly," "materially," "sharply," "widely," "honest," "crushes" — replace with precise figures. Title = 9 words (≤14) ✓.

7. **Figures — AT RISK.** Figure 1 is an explicit **placeholder** ("a detailed framework diagram will be generated separately"), so it is not actually present. Figure 2 (anytime) **cannot be produced** from the evidence (no timing). Only Figures 3 (relative gap) and 4 (warm-start rescue) are backed by data. You are at the 2-figure minimum only if 3 and 4 are real renders — verify they exist as files, replace the Figure 1 placeholder with an actual diagram, and either produce real timing for Figure 2 or remove it.

8. **Citation distribution — PASS (strength).** Citations appear in Method (Inoue–Yoshida, McMahon, Leleu, Kirkpatrick, Goto, Zhu, Wu & Hao, Bergstra, Akiba), Experiments (Benlic & Hao, Goto, Akiba, Bergstra), and Discussion (Leleu, Hamerly, Goto, Gomes & Selman, Benlic & Hao, Aramon). Well distributed beyond Intro/Related Work.

### Additional editorial errors to fix
- **Cross-reference numbering is wrong throughout:** "the weighted cut … defined in Section 5" (it is in Method/Section 3); "every number reported in Section 6" (Results is Section 5); "C(s) defined in Section 5" in Evaluation metrics. Renumber all internal references.
- **Two "Table 1"s:** the hyperparameter table (Experiments) and the main gap table (Results) are both labeled Table 1. Renumber.
- **Solver-count inconsistencies:** Results says "All seven standalone methods" (there are six solvers; the portfolio is not standalone). The benchmark text ("Two are mid-size sparse G-set graphs … and two extend to large sparse graphs") implies five instances but only four exist; G55/G70 are described as both — fix the taxonomy to: one mid-size sparse (G22), one dense weighted (K2000), two large sparse (G55, G70).
- **Prior-run lesson:** the earlier quality-gate block ("VerifiedRegistry has zero real experiment values") now appears addressed — the tables are grounded in the four JSON files — but the missing timing data means a verifier could still reject the anytime claims. Ground or remove them before re-submission.