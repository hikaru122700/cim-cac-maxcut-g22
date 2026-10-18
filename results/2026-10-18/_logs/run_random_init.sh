#!/bin/bash
# ステップ 4 の比較相手:ランダム初期化の SA(開始温度 2.0、500・5000 掃引、512 回)
set -e
export NUMBA_NUM_THREADS=8
for G in G1 G14 G15 G22 G23 G32 G39 G55 G70 G77; do
  python scripts/benchmarks/random_init_sa.py --dataset $G --runs 512 --sweeps 500 5000 --t-start 2.0
done
echo ALL_DONE
