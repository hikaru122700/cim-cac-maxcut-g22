---
created: '2026-06-12T06:59:50+00:00'
evidence:
- stage-09/exp_plan.yaml
id: experiment_design-rc-20260612-063656-82d42f
run_id: rc-20260612-063656-82d42f
stage: 09-experiment_design
tags:
- experiment_design
- stage-09
- run-rc-20260
title: 'Stage 09: Experiment Design'
---

# Stage 09: Experiment Design

ablations:
- expected_effect: 'If non-monotonicity is the source of gain (FH1), this drops best-of-K
    gap reduction toward the monotonic baselines. If gap is unchanged, the free-form
    win was about smoothness/scaling, not reheat (informative null).

    '
  how_it_differs: 'Identical to differentiable_freeform_pump but knots are passed
    through a cumulative-softplus parameterization so the spline is forced monotonically
    non-decreasing (theta_k = theta_0 + sum softplus(delta_j)). Everything else (Kp,
    opt_steps, loss, budget) unchanged.

    '
  isolates: FH1 (does NON-MONOTONICITY matter?)
  name: monotonic_constrained_pump
  what_is_removed: the freedom for p(t) to decrease (reheat/dip).
- expected_effect: 'If backprop is the differentiator, differentiable_freeform_pump
    beats this at equal compute (faster/better convergence in 12-D). If CMA-ES matches
    it, the contribution is the search SPACE not differentiability — a key, publishable
    distinction.

    '
  how_it_differs: 'SAME 12-knot free-form space and SAME budget, but knots optimized
    by CMA-ES (gradient-free, ~60 evaluations matched to opt_steps*batch compute)
    instead of Adam-on-backprop. Soft-cut still the objective (no gradient used).

    '
  isolates: differentiability value (is backprop-through-rollout the win, or just
    the richer search space?)
  name: gradient_free_cmaes_freeform_pump
  what_is_removed: the gradient / backprop-through-rollout optimizer.
- expected_effect: 'If FH3 holds, this lands within +/-20% of the learned best-of-K
    gap with ZERO search; the spectral_gap_rate_correlation metric then shows |rho|>0.5
    across regimes. If it is >2x off, Gap-1 ''no prescription'' stands and learning
    is genuinely required.

    '
  how_it_differs: 'No optimization at all. A linear ramp whose rate is set by the
    Kibble-Zurek / quasispecies closed form rate* = c * spectral_gap^alpha (c, alpha
    fixed a priori from one calibration cell), using the same Lanczos spectral gap
    as spectral_conditioned_pump_net.

    '
  isolates: FH3 (does the closed-form spectral-gap law match learned optimum?)
  name: kibble_zurek_analytic_pump
  what_is_removed: ALL learning/search — schedule set analytically.
baselines:
- description: 'The universal hand-fixed monotonic linear pump ramp p(t) = p0 + (pf
    - p0) * t/T. This is THE de-facto default the goal targets (Inagaki 2016 Science;
    Honjo 2021 Sci.Adv.). Fair because it is the literal status quo every CIM/CAC
    implementation ships with.

    '
  implementation_spec:
    algorithm_steps:
    - 1. p(t) = p0 + (pf-p0)*t/T over T Euler-Heun steps (no learnable params).
    - '2. Integrate mean-field CIM SDE: dx_i = (-1 + p(t) - x_i^2) x_i + xi*sum_j
      J_ij x_j + sigma*dW.'
    - 3. Clamp |x_i|<=x_clip each step for stability.
    - 4. Run B=K trajectories (different noise seeds) in one GPU batch.
    - 5. Cut from sign(x_i); record mean cut and best-of-K cut.
    class_name: LinearRampPump
    differentiator: Fixed monotonic linear shape; zero schedule learning.
    key_hyperparameters:
      K_trajectories: 20
      T_steps: 400
      dt: 0.05
      p0: -1.0
      pf: 1.0
      sigma: 0.05
      x_clip: 3.0
      xi: 0.5
    key_methods:
    - __init__
    - pump_value
    - solve
    - evaluate_cut
    loss_function: none (no optimization); reports cut only
  name: linear_ramp_baseline
- description: 'Current best-practice TUNING baseline: a 4-scalar sigmoid ramp p(t)=p_end*sigmoid(rate*(t-t_on))+offset
    with {t_on,rate,p_end,offset} optimized by Optuna TPE (Akiba 2019). Represents
    the assumed-shape, scalar-only HPO that practitioners already run — a strong,
    modern, fair comparator the full-shape method must beat.

    '
  implementation_spec:
    algorithm_steps:
    - 1. Define 4-D search space over {t_on,rate,p_end,offset}.
    - 2. TPE proposes scalars; build sigmoid ramp; integrate CIM SDE (B trajectories).
    - 3. Objective = best-of-K soft-cut on S_train graphs.
    - 4. Run N_trials TPE iterations; freeze best scalars.
    - 5. Evaluate frozen ramp on S_test.
    class_name: OptunaScalarSigmoidPump
    differentiator: Optimizes only 4 scalars of an ASSUMED monotonic sigmoid shape;
      no shape freedom, gradient-free.
    key_hyperparameters:
      K_trajectories: 20
      T_steps: 400
      dt: 0.05
      n_trials: 80
      sampler: TPE
      search_seed_split: true
    key_methods:
    - __init__
    - suggest_params
    - pump_value
    - solve
    - fit
    - evaluate_cut
    loss_function: maximize best_of_K soft_cut on S_train (TPE objective)
  name: optuna_scalar_sigmoid_ramp
- description: 'Strong modern near-SOTA CIM variant: Chaotic Amplitude Control (Leleu
    2019 PRL; Reifenstein 2021) with its standard linear pump and error-feedback variable.
    Best-quality solver baseline; fair because CAC is the current high-performance
    reference for CIM MAX-CUT and isolates ''is pump-shape gain redundant once amplitude
   

... (truncated, see full artifact)
