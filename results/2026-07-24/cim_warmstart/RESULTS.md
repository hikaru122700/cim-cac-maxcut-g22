# CIM warm-start experiment (2026-07-24)

## Research question

Does seeding GA, SA, or PT-ICM with solutions produced by a short CIM run
improve MAX-CUT quality at short end-to-end wall-clock times?

Warm-start time is measured as:

`total_time = measured CIM generation time + downstream solver time`

All comparisons use 16 runs, four Numba threads, the same integer seed vector,
and the repository's instance-specific solver parameters. Reported cut values
are recomputed with the common weighted-cut scorer.

## PT-ICM initialization

Two PT-ICM initialization schemes were added:

- `single`: place the CIM solution only in the lowest-temperature A replica.
- `ladder`: initialize every A replica from the CIM solution and flip each spin
  with a probability increasing linearly from 0 at the lowest temperature to
  0.5 at the highest temperature.

The B replicas remain random in both schemes. This preserves A/B disagreement,
which is required for isoenergetic cluster moves, and avoids collapsing the
temperature population to identical states.

## G22

CIM used 600 rounds and produced best/mean cuts of 13270/13247.5 in 0.222 s.
BKS is 13359.

| Solver | Budget | Cold best / mean | CIM-warm best / mean | Cold / warm total time | Mean paired gain |
|---|---:|---:|---:|---:|---:|
| GA | 3 generations | 13357 / 13334.5 | 13358 / 13337.1 | 3.177 / 3.305 s | +2.6 |
| GA | 20 generations | 13358 / 13350.3 | 13358 / 13351.7 | 6.494 / 6.944 s | +1.4 |
| SA | 1,000,000 iterations | 13327 / 13293.7 | 13331 / 13309.3 | 0.273 / 0.548 s | +15.6 |
| SA | 10,000,000 iterations | 13358 / 13342.5 | 13340 / 13320.3 | 2.607 / 3.225 s | -22.2 |
| PT single | 30 sweeps | 13192 / 13157.7 | 13315 / 13294.9 | 1.025 / 1.379 s | +137.3 |
| PT ladder | 30 sweeps | 13192 / 13157.7 | 13328 / 13300.6 | 1.025 / 1.286 s | +142.9 |
| PT single | 200 sweeps | 13313 / 13297.3 | 13330 / 13314.5 | 6.419 / 6.460 s | +17.2 |
| PT ladder | 200 sweeps | 13313 / 13297.3 | 13353 / 13325.1 | 6.419 / 6.714 s | +27.8 |

The PT gains are paired-Wilcoxon significant at every tested budget and for
both initialization modes (`p <= 0.0026`). At 30 sweeps, warm PT beats cold PT
in all 16 paired runs. GA differences are not significant. Short SA improves,
but longer warm SA is significantly worse, showing that its warm-specific
temperature schedule is not suitable for a long tail.

Raw result: [v3_G22_nt16_main/results.json](v3_G22_nt16_main/results.json)

## G55 replication

CIM used 600 rounds and produced best/mean cuts of 10179/10133.8 in 0.303 s.
BKS is 10299.

| Solver | Budget | Cold best / mean | CIM-warm best / mean | Cold / warm total time | Mean paired gain |
|---|---:|---:|---:|---:|---:|
| GA | 3 generations | 10087 / 10035.4 | 10214 / 10178.5 | 3.964 / 5.076 s | +143.1 |
| GA | 20 generations | 10177 / 10137.0 | 10214 / 10180.4 | 12.366 / 12.675 s | +43.4 |
| SA | 1,000,000 iterations | 10187 / 10157.4 | 10228 / 10192.2 | 0.205 / 0.537 s | +34.8 |
| SA | 10,000,000 iterations | 10256 / 10244.2 | 10245 / 10221.9 | 1.866 / 2.015 s | -22.3 |
| PT single | 30 sweeps | 10032 / 9997.5 | 10205 / 10177.2 | 1.446 / 1.717 s | +179.7 |
| PT ladder | 30 sweeps | 10032 / 9997.5 | 10218 / 10182.3 | 1.446 / 1.710 s | +184.8 |
| PT single | 200 sweeps | 10205 / 10185.7 | 10229 / 10197.6 | 9.658 / 10.237 s | +11.9 |
| PT ladder | 200 sweeps | 10205 / 10185.7 | 10243 / 10217.4 | 9.658 / 9.340 s | +31.8 |

The G22 pattern replicates on G55. PT ladder improves all 16 paired runs at
30, 80, and 200 sweeps (`p <= 0.00044`). GA warm-start also improves strongly
at all budgets (`p <= 0.00065`). SA helps at the shortest budget but becomes
harmful at the longest budget.

Raw result:
[v4_G55_nt16_replication/results.json](v4_G55_nt16_replication/results.json)

## Current conclusion

The hypothesis is supported for short-budget search, but not universally:

1. CIM-to-PT-ICM is the clearest result. Quality-diverse ladder seeding
   consistently outperforms random PT-ICM on both instances, even after adding
   CIM generation time.
2. CIM-to-GA is instance dependent: negligible on G22, large on G55.
3. CIM-to-SA is useful as a short refinement tail. A fixed long anneal can
   erase the initial advantage and perform worse than cold SA.
4. These are two-instance, best-of-16 experiments. Claims of generality require
   held-out G-set instances, more independent seeds, and warm-parameter tuning
   separated from final evaluation.

## Reproduction

```bash
uv run python scripts/benchmarks/cim_warmstart_bench.py \
  --dataset G22 --num-trials 16 --cim-budget 600 --tag main

uv run python scripts/benchmarks/cim_warmstart_bench.py \
  --dataset G55 --num-trials 16 --cim-budget 600 --tag replication
```
