#!/bin/bash
# ステップ 4:500 掃引ではランダム初期化に 1000 本以上を回すので、512 本では足りない。4096 本を追加で作る。
set -e
export NUMBA_NUM_THREADS=24
for G in G1 G14 G22 G23 G32 G55 G70; do
  python scripts/benchmarks/random_init_sa.py --dataset $G --runs 4096 --sweeps 500 --t-start 2.0
done
echo ALL_DONE
