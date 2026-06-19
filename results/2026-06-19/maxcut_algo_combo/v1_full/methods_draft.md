# （草稿）方法論セクション — 後で RESULTS.md に統合

## 1. 研究目的

MAX-CUT 問題に対し、ハイパーパラメータを調整した複数のソルバ
（物理インスパイア型 CIM・CAC、古典ヒューリスティクス SA・SB・PT-ICM・GA）を
**同一の実時間軸**で比較し、

1. 各ソルバ単体の「実時間 × 到達カット」特性（anytime プロファイル）を明らかにし、
2. 物理ソルバ（CIM/CAC）と古典局所探索（SA/GA-TS）を**組み合わせる**ことで、
   単体より効率的に・高品質な解に到達できるかを検証する。

## 2. ベンチマークインスタンス

| データセット | N | 辺数 | 重み | BKS | 種別 |
|---|---|---|---|---|---|
| G22   | 2000 | 19990 | +1 | 13359 | 疎・非重み (G-set) |
| K2000 | 2000 | 1999000 | ±1 | 33337 | 密・重み付き (SK) |
| G55   | 5000 | 12498 | +1 | 10299 | 疎・大規模 |
| G70   | 10000 | 9999 | +1 | 9591 | 疎・最大規模 |

## 3. アルゴリズムと実装

すべて Numba JIT + trial 並列（`prange`）の自家実装（`modules/`）。

- **CIM**（`modules/CIM.py`）: Inoue–Yoshida 進行波モデル（ファイバーループ + PSA）。
  物理パラメータ κ, L, γ, η, ΔP, 結合 J を持つ。
- **CAC**（`modules/CAC.py`）: Leleu 2021 カオス振幅制御。誤差変数 e_i による
  能動補正で全パルスを共通目標 a に揃える。**注意**: 本実装の内部カット計数は
  非重み（辺本数）なので、重み付き K2000 では返り符号から統一採点で再評価した。
- **SA**（`modules/SA.py`）: 指数冷却の焼きなまし。
- **SB**（`modules/SB.py`）: Goto らの Simulated Bifurcation（dSB を採用）。
- **PT-ICM**（`modules/PT_ICM.py`）: 並列焼きなまし + 等エネルギークラスタ移動。
- **GA**（`modules/GA.py`, 本研究で新規実装）: Wu & Hao 2013 の枠組みを
  制約なし MAX-CUT 向けに具現化したメメティックアルゴリズム。
  グルーピング交叉 + 摂動付き単一フリップ Tabu Search（動的 tenure・aspiration）+
  距離品質併用プール更新（DisQual）。参考: Wu & Hao, *Comput. Oper. Res.* 40 (2013) 166;
  Wu & Hao, PPSN 2012; Wu, Wang, Lü, *Appl. Soft Comput.* 34 (2015) 827。

## 4. ハイパーパラメータ調整

- **G22**: CIM・CAC は既存の Optuna 探索結果（CIM 1000 trial, CAC 250 trial）を使用。
- **K2000・G55・G70**: G22 の物理パラメータは転移しない（特に CIM）ため、
  各データセットで CIM・CAC を Optuna（TPESampler, 25–30 trial）で再調整し、
  目的関数＝統一重み付き平均カットを最大化（`scripts/tuning/tune_anytime_params.py`、
  `results/anytime_tuned_params.json` に保存）。
- SA・SB・PT-ICM・GA は全データセットで頑健に動くため、文献推奨の
  インスタンス適応設定（SB の auto_c0、PT の幾何温度ラダー等）を共通使用。

## 5. 評価プロトコル（anytime 予算スイープ）

各ソルバの主要計算量ノブ（CIM=rounds, CAC=steps, SA=iters, SB=steps,
PT=sweeps, GA=generations）を幾何グリッドで振り、各点で
`num_trials` バッチ（既定 16）を固定 seed で実行。バッチの実時間 t と、
**全手法共通の重み付きカット関数**（`ctx.score()`、返り符号から再計算）で採点した
カット統計（最良・平均）を記録。x = 実時間（log）、y = 最大カット。
あるバッチが時間上限を超えたら以後の大予算はスキップ（適応打ち切り）。
JIT コンパイルは計測前にウォームアップで除外。

## 6. 組合せ（ハイブリッド + 並列ポートフォリオ）

- **ウォームスタート・ハイブリッド**: 物理ソルバ（explorer ∈ {CIM, CAC}）を
  短い固定予算で走らせ良い初期スピン配置を得て、それを局所探索
  （refiner ∈ {TS = GA のメメティック局所探索, warm-start SA}）の初期値に与える。
  同一の精錬予算で、乱数初期からの cold-start と比較（`scripts/benchmarks/combo_bench.py`）。
  総時間 = explorer 時間 + refiner 時間。
- **並列ポートフォリオ**: 単体 anytime データから、各時刻で全手法の最良を採る包絡線
  （K コア並列）と、1 コアを K 分割する時間分割版を計算（`scripts/benchmarks/analyze_results.py`）。

## 7. 再現

- ハーネス: `scripts/benchmarks/{algo_registry,anytime_bench,combo_bench,analyze_results}.py`
- 調整: `scripts/tuning/tune_anytime_params.py`
- 図は Yu Gothic・目盛り内向き（プロジェクト共通スタイル）。
