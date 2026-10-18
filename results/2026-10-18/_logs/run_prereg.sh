#!/bin/bash
# 事前登録 prereg_eigen_v1.md の R1〜R5 を順に実行する
set -e
export NUMBA_NUM_THREADS=24
P=scripts/benchmarks/seed_value_pilot.py
python $P --dataset G22 --cim-rounds 600  --cim-seed-start 256 --sa-seed-offset 2000000 --t-start 2.0 --budgets 100000 1000000 10000000 --tag R1
python $P --dataset G22 --cim-rounds 600  --cim-seed-start 256 --sa-seed-offset 3000000 --t-start 4.21405133142366 --t-end 0.05170732376450782 --budgets 100000 1000000 10000000 --tag R2
python $P --dataset G22 --cim-rounds 2400 --cim-seed-start 256 --sa-seed-offset 4000000 --t-start 2.0 --budgets 100000 1000000 10000000 --tag R3
python $P --dataset G55 --cim-rounds 2400 --sa-seed-offset 2000000 --t-start 2.0 --budgets 250000 2500000 25000000 --tag R4
python $P --dataset G70 --cim-rounds 2400 --sa-seed-offset 2000000 --t-start 2.0 --budgets 500000 5000000 50000000 --tag R5
echo ALL_DONE
