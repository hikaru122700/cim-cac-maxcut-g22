# Peer Review: "WARP: Warm-Starting Physics-Inspired Ising Solvers for Anytime MAX-CUT Optimization"

## Summary of Submission

The paper proposes a unified, identically-scored, wall-clock "anytime" benchmark for six MAX-CUT solvers (CIM, CAC, SA, SB, PT-ICM, memetic GA), plus a warm-start hybrid (physics explorer → classical refiner) and a parallel best-of-K portfolio. The headline claims are that (a) tuned classical solvers dominate, (b) physics solvers are best used as global initializers, and (c) warm-starting Tabu Search from a brief CIM run breaks the CIM plateau and dominates cold-start at matched budget.

I verified the paper's quantitative claims against the four supplied JSON evidence files (G22, K2000, G55, G70) and the automated quality flags. The internal arithmetic is impressively clean, but the work has **one structural defect that all three of us converge on independently: the central framing is temporal ("anytime," "wall-clock," "under one second") yet the supplied evidence contains no timing data at all (`elapsed_sec` is `null` in every file), and the key cold-start control has no reported numbers.**

---

## Reviewer A — Methodology Expert

### Strengths
- **The core methodological commitment is sound and addresses a real gap.** Re-scoring every solver from the returned spin signs through one shared weighted-cut function `C(sign(s))` (§3.2) is the right way to neutralize machine-native metrics, and the paper correctly identifies that CAC's internal unweighted counting would otherwise corrupt comparisons. This is the paper's most defensible contribution.
- **Budget-matched hybrid design (Algorithm 1) is, in principle, well-controlled.** Charging `t_E + t_R` and holding `b_R` fixed between warm and cold runs is exactly the right ablation logic to isolate the value of the initializer.
- **The component ablation (Table 5) is genuinely informative** and its numbers match the evidence exactly (e.g., CIM→TS K2000 = 120, CIM→warm-SA = 356; CAC→warm-SA G70 = 101). The explorer × refiner decomposition is a real strength.

### Weaknesses
- **The anytime axis is asserted, not measured.** Sections 3.2, 4 ("Hardware and runtime"), and Figures 2–4 all describe wall-clock time-versus-quality curves, but every evidence file reports `elapsed_sec: null` and a single final metric per solver — no time series, no per-knob grid points. The entire "anytime" methodology is therefore **unverifiable from the released artifacts**. A benchmark whose central axis is time must report time.
- **The cold-start control has no data.** Algorithm 1's final line and the abstract/conclusion claim of "dominating cold-start at matched refinement budget" are central, but the evidence contains only warm-start hybrids (`hybrid_CIM_TS`, etc.). There is no `cold_TS` or random-init refiner number anywhere. The decisive experiment of the paper is described but not evidenced.
- **The "collapse on a dense instance when parameters are not re-tuned" claim (abstract, §3.1, §4) has no supporting experiment.** The only K2000 CIM number is the *re-tuned* result (gap 794, 2.38%). There is no before/after datapoint demonstrating a collapse, so this reads as a tuning anecdote elevated to a finding.
- **Experimental scope vs. evidence mismatch.** The paper claims "16 fixed-seed trials" per (knob, instance) point on a geometric grid across six solvers; the evidence note states the experiment was **executed 4 times** (consistent with one run per instance, not a swept grid of 16-trial batches). The full anytime sweep that the Method requires cannot be reconstructed from four single-value JSON files.

### Actionable Revisions
1. **Release and plot the actual (time, cut) traces.** Populate `elapsed_sec`, report per-knob grid points, and regenerate Figures 2–4 from real data. Without this, remove every temporal claim ("anytime," "wall-clock," "under one second," "rises fastest," "plateau").
2. **Add the cold-start numbers to Table 4/5** as an explicit row/column (random-init TS and warm-SA at matched `b_R`). The warm-vs-cold delta is the paper's thesis and must be a number, not only Figure 3.
3. **Run and tabulate the "collapse" experiment** (G22-tuned CIM applied unchanged to K2000) or downgrade the claim to "requires per-instance re-tuning."
4. **Reconcile the trial count.** State unambiguously how many seeds/runs underlie each cell, and align the prose with the 4 executed runs.

---

## Reviewer B — Domain Expert (Ising machines / MAX-CUT)

### Strengths
- **The solver lineup and citations are appropriate and well-distributed.** CIM (Inoue & Yoshida), CAC (Leleu), dSB (Goto), PT-ICM with isoenergetic cluster moves (Zhu), and a memetic GA in the Wu & Hao / Benlic & Hao lineage are the right modern baselines. Critically, **citations appear in Method, Experiments, and Discussion**, not only Intro/Related Work — this satisfies the citation-distribution requirement and is above average for the area.
- **Treating classical methods as first-class competitors rather than strawmen is the correct stance**, and the empirical conclusion (SB and memetic GA are strong; physics solvers sit mid-pack on the true weighted objective) is credible and matches community experience on dense SK instances.
- **The instance selection is sensible** — sparse unweighted (G22), dense weighted SK (K2000), and large sparse (G55/G70) — and probes the regimes where physics solvers are known to be fragile.

### Weaknesses
- **The "physics solvers are fast" premise is never substantiated here.** Every speed claim ("despite their speed," "CIM climbs fastest," "fast early climb") depends on timing the paper does not provide. In a venue that knows CIM/SB speed claims are contentious, presenting speed without wall-clock numbers is a serious gap.
- **Four instances is thin for a domain generalization**, and three of them are sparse G-set graphs. The dense regime — where the paper's most interesting result lives — rests on a single instance (K2000). The "no single solver wins everywhere" narrative is largely "SB wins 3/4, PT-ICM wins G70," which is a weaker statement than the text's "the best single solver changes three times across four instances" (Results §5). That phrasing overstates the variability: SB is best-or-tied on G22, K2000, and G55, with a single handoff to PT-ICM on G70.
- **The portfolio's headline advantage is razor-thin and instance-specific.** Portfolio mean gap 0.198% vs. SB 0.203% (Table 3) is a 0.005-point difference driven essentially only by G70 (portfolio 51 vs. SB 53). By construction the portfolio is the per-instance minimum, so this "win" is tautological and, given no dispersion estimate, indistinguishable from noise.
- **Hardware reporting is insufficient for a hardware-adjacent claim.** Only `NUMBA_NUM_THREADS=4` is given — no CPU model, clock, RAM, or OS — yet the paper makes per-instance wall-clock and "under one second" claims.

### Actionable Revisions
1. **Add timing tables (median time-to-target and time-to-plateau per solver per instance)** so the speed/anytime claims become checkable; report the full hardware spec.
2. **Expand the dense/weighted coverage** with additional SK sizes and at least one more dense G-set instance, so the K2000 conclusion is not a single point.
3. **Soften "changes three times"** to reflect that SB is dominant with one handoff; recompute and report whether the portfolio's edge over SB survives any plausible noise band.
4. **Position the portfolio honestly** as an oracle-free envelope whose benefit over the best single solver (SB) is marginal on this instance set.

---

## Reviewer C — Statistics / Rigor Expert

### Strengths
- **Internal numerical consistency is excellent.** I recomputed Table 3 mean gaps from the per-instance percentages and all six solver means plus the portfolio match to three decimals (e.g., SB = (0.007+0.135+0.117+0.553)/4 = 0.203; portfolio = 0.198). The "≤0.2% gap" counts (SB 3/4, GA 2/4, portfolio 3/4) are all correct. Tables 4 and 5 match the JSON cell-for-cell, and the bolded best-refiner pattern in Table 5 is internally correct.
- **The 85% reduction claim is accurate:** K2000 CIM 794 → CIM→TS 120 = 84.9%. The "within a single cut" G22 claim (gap 1) is also exactly supported.
- **The Limitations section is honest** about the missing dispersion and explicitly declines p-values rather than fabricating them.

### Weaknesses
- **No measure of dispersion, so no inferential statistics are possible.** The released data records only the *best cut over each 16-trial batch* — an order statistic (maximum), not a sample with variance. Reporting `n=16` trials but retaining only the max means the effective reported quantity has `n=1` for purposes of interval estimation. The "95% CI" column in Table 3 is entirely "—".
- **Headline comparisons therefore rest on uncontrolled effect sizes.** The portfolio-vs-SB difference (0.198 vs 0.203) and several hybrid-vs-baseline deltas are well within the plausible run-to-run variability of stochastic solvers, but this cannot be assessed. Single-best comparisons also **bias toward whichever method had more effective restarts**, an uncontrolled confound.
- **The "0.01%" abstract claim is an overgeneralization.** "Tuned classical solvers reach within 0.01% of best-known cuts" holds only on G22 (SB/CAC = 0.007%); SA on G22 is already 0.037%, and on K2000/G55/G70 the best classical gaps are 0.135%/0.117%/0.553%. The claim is true for one instance and one-to-two solvers, not "classical solvers" generally.
- **"Under one second" has zero supporting measurement** (see Reviewers A/B). Pairing a precise numeric speed claim with `elapsed_sec: null` evidence is the most serious rigor violation in the paper.

### Actionable Revisions
1. **Retain per-trial cut values** (all 16 seeds × knob points) and report median ± IQR or bootstrap 95% CIs; fill the CI column. This is mandatory for a benchmark paper.
2. **Add paired tests where a head-to-head claim is made** (e.g., warm vs. cold per seed → Wilcoxon signed-rank), or explicitly restrict claims to descriptive effect sizes and label them as such.
3. **Replace "within 0.01%" with the instance-specific statement** ("SB and CAC reach within 0.01% on G22; best classical gaps elsewhere are 0.12–0.55%").
4. **Either measure and report wall-clock with variance, or strike every absolute-time claim.**

---

## Consolidated Checklist Verdict

| # | Criterion | Verdict |
|---|---|---|
| 1 | **Topic alignment** | ✅ On-topic throughout. Minor concern: the "collapse without re-tuning" tuning artifact and CAC's internal unweighted counting are partly framed as findings/contributions; keep these in Limitations, not as results. |
| 2 | **Claim–evidence alignment** | ⚠️ Supported: 85% reduction (794→120), G22 warm-start gap=1, Tables 3/4/5 arithmetic, "no single solver wins everywhere." **Unsupported:** "under one second" (no timing), "dominates cold-start at matched budget" (no cold-start data), "collapses when not re-tuned" (no experiment), "within 0.01%" (true only on G22). |
| 3 | **Statistical validity** | ❌ No CIs/error bars (column all "—"); only best-of-batch retained, so effective `n=1` for dispersion; no significance tests; key margins (portfolio 0.198 vs SB 0.203) untestable. |
| 4 | **Completeness** | ⚠️ All sections present. Estimated body length ≈ 4,500–5,200 words — at or below the 5,000-word floor. Results (§5) is thin relative to its centrality; expand with timing and cold-start data. |
| 5 | **Reproducibility** | ⚠️ Datasets, BKS, tuning trial counts (1000/250 on G22, 25–30 elsewhere) given; exact hyperparameters deferred to an artifact. **Missing:** specific seed values, full hardware spec, and — critically — any wall-clock numbers. |
| 6 | **Writing quality** | ⚠️ Mostly flowing prose (good — Method/Results/Discussion are not bulleted). **Flags:** Intro contributions are a bullet list (automated: 44% list density); 21 weasel words ("surprisingly little," "roughly," "essentially free," "sharply," "comfortably," "marginally"). Title = 9 words (✅ ≤14). |
| 7 | **Figures** | ⚠️ Four figures referenced (≥2, not a desk reject), **but Figure 1 is an unfilled placeholder** ("will be generated separately"), and Figures 2–4 are time-series that the supplied evidence cannot produce (`elapsed_sec` null). |
| 8 | **Citation distribution** | ✅ Strong. Citations appear in Method (Barahona, Lucas, Lam, Inoue & Yoshida, Leleu, Goto, Zhu, Glover, Wu & Hao, Benlic & Hao), Experiments (Akiba/Optuna, etc.), and Discussion (Egger, Xu, Gomes & Selman). |

---

## Overall Recommendation

**Major Revision (borderline reject in current form).** The conceptual contribution — uniform re-scoring and a budget-matched warm-start ablation — is valuable and the supported numbers are clean and correctly reported. However, the paper's identity is "anytime/wall-clock," and the supplied evidence contains **no timing data and no cold-start control numbers**, so the title claim, the abstract's "under one second," and the central "dominates cold-start" thesis are presently unverifiable. These are fixable with data the authors evidently can generate; until the timing traces, the cold-start baseline, and per-trial dispersion are reported, the paper's headline claims outrun its evidence.

**Highest-priority fixes:** (1) report wall-clock time-series with variance and regenerate Figures 2–4 from them; (2) add the cold-start refiner numbers; (3) add CIs / per-trial data; (4) fill Figure 1 and reconcile the 16-trials-vs-4-runs description; (5) restrict the "0.01%" and "changes three times" claims to what the four instances actually show.