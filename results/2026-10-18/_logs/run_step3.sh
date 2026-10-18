#!/bin/bash
# docs/20261018/step3_plan.md の実行:G-set 10 問題 × 開始温度 2 つ
set -e
export NUMBA_NUM_THREADS=24
P=scripts/benchmarks/seed_value_pilot.py
declare -A N=([G1]=800 [G14]=800 [G15]=800 [G22]=2000 [G23]=2000 [G32]=2000 [G39]=2000 [G55]=5000 [G70]=10000 [G77]=14000)
for G in G1 G14 G15 G22 G23 G32 G39 G55 G70 G77; do
  n=${N[$G]}
  for T in 0.5 2.0; do
    if [ "$T" = "0.5" ]; then OFF=6000000; else OFF=7000000; fi
    python $P --dataset $G --candidates 512 --cim-rounds 2400 --cim-seed-start 1000 \
      --sa-seed-offset $OFF --t-start $T --budgets $((50*n)) $((500*n)) $((5000*n)) \
      --trial-runs 2 --trial-sweeps 50 --tag S3
  done
done
echo ALL_DONE
