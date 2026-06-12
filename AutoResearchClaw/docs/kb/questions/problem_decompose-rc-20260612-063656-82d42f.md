---
created: '2026-06-12T06:39:47+00:00'
evidence:
- stage-02/problem_tree.md
id: problem_decompose-rc-20260612-063656-82d42f
run_id: rc-20260612-063656-82d42f
stage: 02-problem_decompose
tags:
- problem_decompose
- stage-02
- run-rc-20260
title: 'Stage 02: Problem Decompose'
---

# Stage 02: Problem Decompose

# Research Decomposition: CIM ポンプ電力関数 *p(t)* の最適化

## Source

- **Topic:** CIM のポンプ電力関数の最適化 — learning a time-dependent pump-rate / gain schedule *p(t)* for a simulated Coherent Ising Machine (mean-field DOPO network) solving MAX-CUT / Ising problems.
- **Goal context:** "Differentiable Optimization of the CIM Pump-Power Schedule" (my-research, domain: machine-learning, generated 2026-06-12, Asia/Tokyo; quality threshold 4.0).
- **Core novel angle:** Treat *p(t)* as a free, high-dimensional functional object (piecewise / spline / small-MLP), optimize its *entire shape* via **backprop-through-rollout** of a differentiable mean-field CIM/CAC simulator, and **condition the shape on cheap graph spectral features** for per-instance amortization. The dominant assumption being tested: that the hand-set **monotonic linear ramp** is near-optimal.
- **Hypotheses under test:** H1 (learned, possibly non-monotonic schedule beats best linear ramp on p₉₅ / TTS at matched compute); H2 (instance-conditioned schedules generalize to held-out graphs); H3 (the idea transfers on top of CAC's target-amplitude schedule).
- **Benchmark:** G-set (G1, G6, G11, G14, **G22**, G39; 800–2000 nodes) + synthetic generalization suite (random-regular, Barabási–Albert, planted-partition). Metrics: success probability *p₉₅*, TTS / roundtrips-to-solution, mean/best cut, with bootstrap 95% CIs over ≥ 50 seeds/graph.
- **Constraints:** Single consumer GPU, runtime in hours; simulation only (no optical hardware); PyTorch/JAX for autodiff, Numba/CuPy for the batched integrator, CMA-ES/Optuna as the black-box fallback.

## Sub-questions

### SQ1 — Differentiable simulator foundation (enabling)
**Can we build a GPU-batched, differentiable mean-field CIM/CAC integrator whose gradient *∂(cut)/∂(schedule params)* is usable (low-bias, low-variance enough) for schedule optimization?**
- Which integration scheme (Euler–Maruyama vs. higher-order SDE solvers) and step size give a faithful trajectory while staying differentiable and cheap?
- How noisy/unstable is the gradient through the **bistable settling phase** where amplitudes saturate? Where does it vanish or explode?
- What **soft-cut / straight-through surrogate objective** (e.g., tanh-relaxed sign, temperature-annealed) yields informative gradients for the discrete cut value?
- Reproduction check: does the linear-ramp baseline in this simulator match published mean-field CIM p₉₅/TTS on G22 (validating the integrator before any learning)?
- *Why first:* every other sub-question depends on this artifact; if gradients are unusable, the fallback (CMA-ES/Optuna) must be promoted to primary, which reshapes the whole project.

### SQ2 — Schedule parameterization & the monotonicity question (core scientific claim)
**Given a working optimizer, what is the optimal pump-power functional shape, and is it monotonic?**
- Across parameterizations of increasing expressiveness — (i) linear ramp baseline, (ii) free K-knot spline (K ≈ 8–16), (iii) free piecewise — does a **non-monotonic** *p(t)* emerge, and does it beat the best linear ramp by a statistically significant margin in p₉₅ / TTS at matched compute?
- **Knot-count ablation:** how does performance scale with K? Is there a minimal expressiveness that captures the gain?
- If the linear ramp *is* near-optimal within CI, **why** — is the gradient flat along non-linear directions, or does the settling dynamics wash out shape differences? (The publishable negative result.)
- How robust is the learned shape across seeds and across graphs (is there a *universal* shape or is it instance-specific)?
- *Why core:* this is Success Criterion #1 and the falsifiable test against the field's dominant assumption.

### SQ3 — Instance-conditioned amortization & generalization (the "learning" payoff)
**Can a tiny network *g_θ(graph spectral features) → schedule* produce per-instance schedules that generalize to held-out graphs better than a single globally-tuned schedule?**
- Which **cheap instance features** (spectral statistics — Laplacian eigenvalue spread, spectral gap, degree distribution, size) are predictive of the best schedule shape?
- Does *g_θ* beat a single globally-tuned schedule on **held-out** graphs of different sizes/families (true amortization vs. per-instance overfitting)?
- How does the conditioned MLP compare in cost: amortized one-forward-pass schedule vs. per-graph re-tuning (Optuna) — what's the compute trade-off?
- Does generalization hold across graph *families* (G-set → synthetic random-regular / BA / planted-partition) or only within-family?
- *Why high-value-but-later:* depends on SQ2's parameterization being settled; this is Success Criterion #2 and the strongest ML-novelty contribution.

### SQ4 — Transfer to CAC (breadth / generality of the method)
**Does optimizing CAC's underlying target-amplitude schedule (rather than vanilla CIM's pump) yield a measurable improvement over stock CAC?**
- CAC modulates per-spin amplitude via error-variable feedba

... (truncated, see full artifact)
