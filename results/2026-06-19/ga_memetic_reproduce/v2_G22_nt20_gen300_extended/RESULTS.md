# MAX-CUT 向け GA(メメティック)先行研究の調査と再現実走

**日付**: 2026-06-19  **実験種別**: `ga_memetic_reproduce`  **対象**: G22(G-set, N=2000, 19990 辺, 非重み)

---

## 1. 調査:MAX-CUT に適用された GA / メメティック法の先行研究

MAX-CUT は単純な「ビット列(各頂点の所属側)」で解を表現できるため、古くから GA の
適用対象になってきた。ただし**素朴な GA(ランダム交叉のみ)は MAX-CUT では弱く**、
局所探索を組み込んだ**メメティックアルゴリズム(MA)**が事実上の標準である。本リポジトリの
`modules/GA.py` は、その代表的系譜である **Wu & Hao 系メメティック法**を制約なし
MAX-CUT 向けに具現化したものなので、それを再現対象とした。

### 主要な先行研究

| 研究 | 略称 | 要点 |
|---|---|---|
| Q. Wu, J.-K. Hao, *PPSN XII* (2012), LNCS 7492:297-306 | **MACUT** | グルーピング(共通分割継承)交叉 + 摂動付き Tabu Search。G-set で当時の BKS を多数更新 |
| Q. Wu, J.-K. Hao, *Computers & Operations Research* 40(1):166-179 (2013) | (max-bisection MA) | 距離・品質併用のプール更新 **DisQual** を導入(多様性維持) |
| Q. Wu, Y. Wang, Z. Lü, *Applied Soft Computing* 34:827-837 (2015) | **TSHEA** | 1-flip と限定交換の近傍結合を持つ TS + 均一交叉風結合演算子。G-set 71 例で評価 |
| U. Benlic, J.-K. Hao, *Eng. Appl. AI* 26(3):1162-1173 (2013) | **BLS** | Breakout Local Search。MA ではないが G-set の BKS 基準として広く引用 |

これらが共通して報告する **G22 の最良既知カット(BKS)= 13359**(BLS とも一致)を
再現目標とした。

### メメティック法の骨子(`modules/GA.py` の実装)

- **解表現**: スピン `s ∈ {0,1}^n`、カット `f(s)=Σ_{(i,j)∈E} w_ij·[s_i≠s_j]`。
- **局所探索 = Tabu Search**: 単一フリップ近傍。移動利得 `Δ_v` の増分更新は `O(deg v)`。
  動的 tenure(15 ブロック周期パターン)・aspiration・無改善で摂動。
- **交叉 = グルーピング交叉**: ラベル対称性込みで両親を整列し、合意した側を子に継承、
  残りを等距離になるよう貪欲割当て。
- **集団更新 = DisQual**: 子を仮挿入し、`score = β·norm(品質) + (1−β)·norm(最小ハミング距離)`
  が最小の個体を捨て、品質と多様性を両立。

> 参考(検証済み): [Wu & Hao PPSN 2012(MACUT)](https://link.springer.com/chapter/10.1007/978-3-642-32964-7_30) /
> [Wu & Hao CORWuHao2012(max-bisection MA)](https://leria-info.univ-angers.fr/~jinkao.hao/papers/CORWuHao2012.pdf) /
> [Wu, Wang, Lü 2015(TSHEA)](https://www.researchgate.net/publication/277930600_A_tabu_search_based_hybrid_evolutionary_algorithm_for_the_max-cut_problem)

---

## 2. 再現プロトコル

先行研究にならい、**1 グラフにつき複数の独立 run**を実行して best / mean / worst・
BKS 到達率・平均実時間を集計する。本実装は Numba JIT + trial 並列(`prange`)。

| 設定 | v1(標準予算) | v2(拡張予算 ★本命) |
|---|---|---|
| 独立 run 数 | 20 | 20 |
| 集団サイズ pop_size | 10 | 10 |
| 世代数 generations | 50 | 300 |
| TS 反復 ts_iters | 20000 | 80000 |
| tenure 係数 / cr / β | 15 / 3000 / 0.6 | 15 / 3000 / 0.6 |

`modules/GA.py::simulate_ga_batch` に世代ごとの収束履歴を返す `return_history` を追加し、
収束曲線を取得した(既存呼び出し側は後方互換、回帰テスト 3 件 pass)。

実行コマンド:
```
python scripts/benchmarks/reproduce_ga_memetic.py --dataset G22 --num-trials 20 --generations 50
python scripts/benchmarks/reproduce_ga_memetic.py --dataset G22 --num-trials 20 --generations 300 --ts-iters 80000 --tag extended
```

---

## 3. 結果

### 到達カット(20 run)

| 指標 | v1(50 世代, 2.3s) | v2(300 世代, 42.8s) | 文献 BKS |
|---|---|---|---|
| **best** | 13357(gap **+2**) | **13359(gap 0)** ✅ | 13359 |
| mean | 13330.1(gap +29.0) | 13346.5(gap +12.6) | — |
| worst | 13305 | 13322 | — |
| std | 16.2 | 14.4 | — |
| **BKS 到達率** | 0 / 20 | **5 / 20 = 25%** | — |
| 平均 run 時間 | 0.12 s/run | 2.1 s/run | — |

- **v2 拡張予算で文献値 13359 に正確に到達**(20 run 中 5 run、到達率 25%)。残りの run も
  13322–13358 に密集し、平均で BKS の **99.91%**(gap 12.6)。再現は成功と判断する。
- v1 の標準予算でも 2.3 秒・20 run で best 13357(BKS の 99.985%)に到達しており、
  メメティック法の効率の高さを確認。

### 図

- `convergence.png` — 世代 × カット。全 run 最良・run 平均・最小〜最大の帯。
  v2 では best が早期(〜世代 130 前後)に 13359 へ到達し、平均も単調上昇。
- `hist.png` — 20 run の最終 best カット分布。v2 では BKS=13359 にピーク(5 run)。

`v1_G22_nt20_gen50/` に標準予算版の同名図・`results.json` を併置。

---

## 4. 所見

- **素朴な GA ではなくメメティック法(交叉 + Tabu Search + 多様性維持プール)が鍵**。
  局所探索 TS が解の質を、グルーピング交叉と DisQual が多様性を担い、両者の協調で
  BKS に到達する。
- 計算量ノブは主に **TS 反復数と世代数**。v1→v2 で約 18 倍の計算を投じて gap +2 → 0 を
  詰めた(MAX-CUT 終盤の数辺を詰めるのが指数的に難しいことの実例)。
- 本リポジトリの他ソルバ(CIM/CAC/SA/SB/PT-ICM)との同一実時間軸比較は
  `scripts/benchmarks/anytime_bench.py`、ハイブリッド(物理ソルバ → TS warm-start)は
  `scripts/benchmarks/combo_bench.py` を参照。

---

## 5. 再現に必要なファイル

- 実装: `modules/GA.py`(`simulate_ga_batch` / `tabu_refine_batch`)
- 再現スクリプト: `scripts/benchmarks/reproduce_ga_memetic.py`
- 回帰テスト: `tests/test_ga.py`(3 件)
- 図スタイル: Yu Gothic・目盛り内向き(プロジェクト共通)
