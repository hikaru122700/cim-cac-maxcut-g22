# MAX-CUT on G-Set — CIM / CAC / SB / SA の実装と比較

Stanford G-Set ベンチマークの MAX-CUT 問題を 4 つの手法で解いて比較するリポジトリです。
中心評価は G22 (N=2000, K=19990, 既知最良解 **13359**) で、入力グラフは
`input/G1.txt`, `G14`, `G15`, `G22`, `G23`, `G32`, `G39`, `G55`, `G70`, `G77`, `G81`
を同梱しています。

実装はすべて Python + Numba で、外ループを JIT コンパイルし
trial 並列(`prange`)で CPU コアに分散実行します。

## 実装した手法

| 手法 | 元論文 | アルゴリズムの概要 |
|---|---|---|
| **CIM** | Inoue & Yoshida, *Opt. Commun.* **522**, 128642 (2022) | 位相感応増幅器(PSA)を持つファイバーループ型コヒーレントイジングマシンの進行波モデル。各パルスの in-phase 振幅の符号からスピンを決める。 |
| **CAC** | Leleu et al., *Comm. Phys.* **4**, 266 (2021) | Chaotic Amplitude Control。パルスごとのエラー変数と目標振幅の動的変調、時変な結合ランプを組み合わせて準最適解を脱出する拡張 CIM。 |
| **SB**  | Goto et al., *Sci. Adv.* **5**, eaav2372 (2019); **7**, eabe7953 (2021); Kanao & Goto, arXiv (2022) | Simulated Bifurcation。非線形ハミルトニアン系の断熱分岐を模倣する古典力学的アルゴリズム。`modules/SB.py` に aSB / bSB / dSB の 3 バリアントを実装。 |
| **SA** | Kirkpatrick et al., *Science* **220**, 671 (1983) | 1-flip 指数冷却の古典的シミュレーテッドアニーリング(ベースライン)。 |

## 実行結果 (G22 で各 100 trial)

| 手法 | 平均 | 最良 | 最悪 | 標準偏差 | 壁時計時間 |
|---|---|---|---|---|---|
| CIM | 13275.3 | 13326 | 13220 | 20.5 | **3.3 秒** |
| CAC | 13284.8 | **13358** | 13214 | 25.8 | 206.5 秒 |
| SA  | 13224.8 | 13314 | 13048 | 50.1 | 200.3 秒 |

- **CAC** が最良 13358 まで到達(既知最良解 13359 にわずか 1 差)
- **CIM** が最速。3.3 秒で平均 13275 の安定結果(既知最良解の 99.75%)
- **SA** はこの時間スケールでは他の 2 手法に劣る

結果の可視化(`results/` は **日付サブフォルダ** で管理。詳細は `CLAUDE.md`):

| 画像 | 内容 |
|---|---|
| `results/2026-04-23/compare_histogram.png` | 各手法の cut 値の分布(3 パネル) |
| `results/2026-04-23/compare_running_best.png` | trial 進行に対する running best |
| `results/2026-04-23/compare_bar.png` | 平均・最良の棒グラフ比較 |
| `results/2026-04-24/compare_all_*.png` | G-Set 全グラフでの 4 手法横断比較 |
| `results/2026-05-12/v1_sb_all_variants.png` | SB の 3 バリアント比較 |

### 論文再現の程度

- **CIM**: 100 trial の平均 **13275.3** は論文 (Inoue & Yoshida 2022, Fig.8) の平均
  13275 と完全一致。最良 13326 は論文値 13321 を若干上回る。
- **CAC**: 論文は **FPGA 実装**で p₀ = 0.11 (100 run 中 11%) で 13359 に到達すると
  報告。本リポジトリの CPU 実装は 100 trial・約 200 秒で最良 13358(1 cut 不足)。
  論文の成功率を再現するには外ループ数を 1〜2 桁増やす必要があります。

## セットアップ

依存解決には [uv](https://docs.astral.sh/uv/) を使っています。

```bash
uv sync
```

主要依存: `numpy`, `scipy`, `numba`, `wandb`, `matplotlib`, `pymupdf` (Python 3.13+)。

## 使い方

スクリプトはすべて **プロジェクトルートから** `python -m <パッケージパス>` の
形で実行します。出力は `results/YYYY-MM-DD/v{N}_*.png` 形式で日付フォルダに
バージョン採番して保存されます(`CLAUDE.md` 参照)。

### CIM を単発で実行(ラウンドごとに wandb にログ)

```bash
uv run python -m modules.CIM
```

### CIM を 100 trial 並列実行(wandb に統計をログ)

```bash
uv run python -m scripts.runners.CMI_multi_run
```

### CAC を 100 trial 並列実行

```bash
uv run python -m modules.CAC
```

### SA ベースラインを実行

```bash
uv run python -m modules.SA
```

### SB(aSB / bSB / dSB)を実行

```bash
uv run python -m scripts.benchmarks.compare_sb_all
uv run python -m scripts.benchmarks.compare_sb_cim
```

### CIM / CAC / SA の比較スクリプト(`results/<date>/` に図を出力)

```bash
uv run python -m scripts.benchmarks.compare
```

### G-Set 全グラフでのベンチマーク

```bash
uv run python -m scripts.benchmarks.benchmark_gset
uv run python -m scripts.benchmarks.plot_benchmark_gset
```

### CIM のハイパーパラメータ調整(Optuna)

```bash
uv run python -m scripts.tuning.tune_cim_optuna
uv run python -m scripts.tuning.verify_optuna_best
```

### CAC ハイパーパラメータ調整 (Hanyu 2025 Method B)

`α, ρ, δ, γ (β_inj 成長率), τ (β_inj リセット窓)` の 5 パラメータを
**感度順に逐次最適化** する Method B を実装 (参考: Hanyu et al., arXiv:2507.20295)。

```bash
# 既定: lex 目的関数 + τ 拡張グリッド (G22 BKS 追跡に推奨)
uv run python -m scripts.tuning.tune_cac

# 旧挙動 (mean_cut 目的 + 対称 τ グリッド)
uv run python -m scripts.tuning.tune_cac --objective mean --tau-standard
```

処理の流れ:

1. **Phase 1 (感度評価)**: 各パラメータを独立に 5 段階の倍率で評価し、
   主要メトリック(`max` / `lex` は `max_cut`, `mean` は `mean_cut`)の
   `max − min` スプレッドで感度を定量化。
   - 非 τ パラメータ: `(0.5, 0.75, 1.0, 1.5, 2.0)` の対称乗数
   - τ (既定で拡張): `(0.1, 0.25, 0.5, 1.0, 2.0)` — τ = 9N が外ループ
     step 数に近く、下側の感度を見るために非対称化
2. **Phase 2 (逐次最適化)**: 感度の高い順に 1 パラメータずつ最適化し、
   勝った値を以降の評価にロック。目的関数は以下から選択:
   - **`lex`** (既定, G22 BKS 追跡用): `(max_cut, mean_cut)` の辞書式比較。
     まず `max_cut` を最大化し、同値時は `mean_cut` で tie-break。
   - **`max`**: `max_cut` のみ。ピーク志向。
   - **`mean`**: `mean_cut` のみ (旧デフォルト、安定志向)。
3. **Final**: GSET 既定値と調整後 config を、full budget
   (100 trial × 50000 outer step) で再評価して
   `Δ max_cut`, `Δ mean_cut`, `Δ optimal_hits` を表示。

評価予算は軽量スクリーニング (20 trial × 20000 step) で Phase1+Phase2 合計約 50 eval、
加えて Final で full budget (100 trial × 50000 step) の 2 eval を行う構成です。
実行時間はマシン性能 (主に CPU コア数と numba JIT キャッシュの有無) に依存します。

CLI フラグ:

| フラグ | 既定 | 説明 |
|---|---|---|
| `--objective {max,mean,lex}` | `lex` | 目的関数 |
| `--tau-standard` | off | τ グリッドを対称 (0.5〜2.0) に戻す |
| `--graph PATH` | `input/G22.txt` | 入力グラフ |
| `--screen-trials N` | 20 | Phase 1/2 の trial 数 |
| `--screen-steps N` | 20000 | Phase 1/2 の外ループ step 数 |
| `--final-trials N` | 100 | Final 評価の trial 数 |
| `--final-steps N` | 50000 | Final 評価の外ループ step 数 |
| `--seed-base N` | 0 | シード基点 |
| `--output-csv PATH` | `results/tune_cac_log.csv` | ログ出力先 |

ログは `results/tune_cac_log.csv` に phase 付きで書き出されます。

単体テスト (numba を起動せず純粋ロジックのみ):

```bash
uv run pytest tests/test_tune_cac.py -v
```

### CAC ビジュアライザ (AHC 風プレーヤー付き HTML)

AtCoder Heuristic Contest のビジュアライザ風のインタラクティブ
プレーヤーで、CAC の 1 run 実行時の内部状態を時間軸に沿って再生できます。
単一の自己完結 HTML ファイル (Plotly.js CDN のみ外部依存) を生成します。

```bash
# 既定: 100 trial x 50000 step, G22
uv run python -m scripts.plotting.run_cac_viz

# 軽量実行 (動作確認用)
uv run python -m scripts.plotting.run_cac_viz --num-trials 10 --outer-steps 5000

# 出力先指定
uv run python -m scripts.plotting.run_cac_viz --output results/viz/my_run.html
```

出力先 (既定): `results/viz/cac_<timestamp>.html`

実装構成:

- **バッチ実行 (JIT, 高速)** で `--num-trials` 件の final cut を集計
- **代表 trial のトレース (純粋 Python + numpy, 低速)** で最高 cut を
  出した seed を再実行し、内部状態を周期的に記録
  - `snapshot_interval` ごと + `β_inj` リセット時 + 改善時に必ず記録
    (集計メトリクス用)
  - `spin_frame_interval` ごと + 改善時に per-spin 振幅を int8 量子化で
    記録 (AHC プレーヤーの空間ビュー用)
- 両者を 1 つの `RunRecord` に統合し、HTML にレンダリング

AHC 風プレーヤー:

- **▶/⏸ 再生・停止** (速度 0.5×〜10× 可変)
- **seek バー** でタイムライン任意地点へジャンプ
- **◀ ▶| ⏮ ⏭ ボタン** で ±1 frame / 先頭・末尾
- **キーボード**: ← / → (±1), Shift+← / → (±10), Space (再生/停止)
- **空間ビュー 1**: 2000 スピンを格子配置 (index 順) で色分け
  (青=x&gt;0, 赤=x&lt;0, 明度=|x|) — 時間に沿って解が結晶化していく様子
- **空間ビュー 2**: |x| 降順ソート済みレーン — 振幅の成長順序
- **振幅分布**: この時点の x 値ヒストグラム
- **再生ヘッド同期**: 全時系列チャートに縦線を描画、seek に応じて移動

表示チャート (Plotly):

1. **Cut 値の推移**: 現在 cut / best cut / 改善点 / β_inj リセット点
2. **振幅 |x| 動態**: 平均・標準偏差の推移
3. **エラー変数 e 動態**: 平均・標準偏差の推移
4. **β_inj と目標振幅 a(t)**: 時変結合強度と目標のダブル軸
5. **スピン符号バランス**: +1 スピン数の推移 (N/2 からの偏り)
6. **最終 cut のヒストグラム**: 全 trial の分布と既知最良解マーカー
7. **ハイパーパラメータ一覧表**: 実行時 config

CLI フラグ:

| フラグ | 既定 | 説明 |
|---|---|---|
| `--graph PATH` | `input/G22.txt` | 入力グラフ |
| `--num-trials N` | 100 | バッチ trial 数 |
| `--outer-steps N` | 50000 | 外ループ step 数 |
| `--seed-base N` | 0 | シード基点 |
| `--snapshot-interval N` | 100 | 集計スナップショット周期 |
| `--spin-frame-interval N` | 500 | AHC プレーヤー用 per-spin フレーム周期 (小さすぎると HTML 肥大化) |
| `--output PATH` | `results/viz/cac_<ts>.html` | HTML 出力先 |
| `--target-cut N` | 13359 | 目標 cut (既知最良解) |

単体テスト (純粋ロジックのみ, numba 不要):

```bash
uv run pytest tests/test_visualize.py -v
```

## 実装上のポイント

### 1. Numba JIT + `prange` による trial 並列化

`_simulate_cim_batch` と `_simulate_cac_batch` は `@njit(cache=True, fastmath=True,
parallel=True)` でネイティブコードにコンパイルされており、`prange` で trial を
CPU コアに分散実行します。初回のみ 3〜5 秒の JIT コンパイル時間がかかりますが、
2 回目以降は `__pycache__` のキャッシュから即起動します。

### 2. スパース結合行列

G22 は N=2000, K=19990 で非零要素はわずか約 1% です。`scipy.sparse.csr_matrix` で
保持し、Numba 内では手書きの CSR matvec ループで scipy のラッパーオーバーヘッド
を回避しています。

### 3. 独立した検算モジュール

`modules/verify.py` が SA/CIM/CAC とは独立にカット数を再計算します。
具体的には辺リスト版と隣接リスト版の 2 種類の方法でカウントして突き合わせ、
符号規約や二重カウントのバグを検出します。

### 4. CIM のノイズ式に ℏω 補正

Inoue & Yoshida 2022 の Eq.(6) は真空ゆらぎ分散 `σ² = (2-η)·G/4·BW` を
「one-photon energy 単位」で書いています。一方 Eq.(14) の飽和係数
`γ = 42.09 W⁻¹` は物理単位(ワット)です。両者を整合させるには **ℏω**
(1550 nm で約 1.28×10⁻¹⁹ J)を乗じる必要があります。

`modules/CIM.py` では `photon_energy_J = 1.28e-19` を config に入れてノイズ分散に
掛けています。この補正を入れないと初期ノイズが信号を吹き飛ばし、飽和項
`γ·I_in` が暴走して系が病的な状態でロックされます。

## リポジトリ構成

```
.
├── modules/                          # アルゴリズム実装本体
│   ├── CIM.py                        # CIM (Inoue & Yoshida 2022) + 単発 main()
│   ├── CAC.py                        # CAC (Leleu et al. 2021) + 100 trial main()
│   ├── SB.py                         # Simulated Bifurcation (aSB/bSB/dSB)
│   ├── SA.py                         # シミュレーテッドアニーリング
│   ├── rulebase.py                   # 1パス貪欲法
│   └── verify.py                     # 独立した検算モジュール
├── scripts/
│   ├── benchmarks/                   # 比較・ベンチマーク
│   │   ├── compare.py                # CIM / CAC / SA の 100 trial 比較
│   │   ├── compare_sb_all.py         # SB 3 バリアント横断比較
│   │   ├── compare_sb_cim.py         # SB vs CIM
│   │   ├── benchmark_gset.py         # G-Set 全グラフでのベンチマーク
│   │   └── plot_benchmark_gset.py    # benchmark_gset の結果可視化
│   ├── runners/                      # 単発/バッチ実行ランナー
│   │   ├── CMI_multi_run.py          # CIM 100 trial + wandb
│   │   ├── rulebase.py               # rulebase 1パス実行
│   │   └── run_no_wandb.py           # wandb なしバッチ実行
│   ├── tuning/                       # ハイパーパラメータ探索
│   │   ├── tune_cac.py               # CAC Hanyu 2025 Method B
│   │   ├── tune_cim_optuna.py        # CIM の Optuna 探索
│   │   └── verify_optuna_best.py     # 最良 config の再評価
│   ├── plotting/                     # 可視化・ビジュアライザ
│   │   ├── visualize.py              # RunRecord → HTML レンダリング
│   │   ├── trace_cac.py              # CAC 1-trial トレーサー
│   │   ├── run_cac_viz.py            # CAC + HTML ビジュアライザ CLI
│   │   ├── plot_amplitude_8nodes.py  # 8 頂点の振幅軌道
│   │   ├── plot_cim_amplitudes.py    # CIM 振幅ダイナミクス
│   │   ├── plot_cim_8vertices.py     # 8 頂点の個別軌道
│   │   ├── zoom_cim_tail.py          # 末端ラウンドの拡大
│   │   ├── investigate_tail_oscillation.py  # 末端振動の詳細解析
│   │   ├── diag_cim_tail.py          # 末端振動の数値診断
│   │   ├── plot_cut_distribution.py  # cut 分布
│   │   ├── plot_cim_only_histogram.py
│   │   └── plot_rulebase_score.py
│   └── utils/                        # ユーティリティ
│       ├── save_assignment.py
│       ├── download_sb_papers.py
│       └── verify_sb_determinism.py
├── tests/
│   ├── test_tune_cac.py
│   └── test_visualize.py
├── input/                            # Stanford G-Set ベンチマーク
│   ├── G1.txt, G14.txt, G15.txt, G22.txt, G23.txt,
│   ├── G32.txt, G39.txt, G55.txt, G70.txt, G77.txt, G81.txt
├── papers/                           # 参照論文 PDF (SB 系)
├── results/                          # 日付サブフォルダ管理 (CLAUDE.md 参照)
│   ├── YYYY-MM-DD/                   # 日付ごとに v{N}_*.png でバージョン採番
│   ├── benchmark_gset.csv            # 全期間共通の集計マスタ
│   ├── tune_cac_log.csv
│   └── viz/                          # ビジュアライザ HTML 出力
├── CLAUDE.md                         # プロジェクト共通ルール
├── README.md
├── LICENSE                           # MIT
├── pyproject.toml
├── uv.lock
└── .python-version
```

## CAC の仕組み(簡単に)

通常の CIM は閾値を超えると全パルスが一斉に ±c_sat に飽和するため、
「1 つだけスピンを反転すれば cut が増える」ような準最適解にトラップされがちです。
CAC はこれを以下の 3 機構で解決します:

1. **パルスごとのエラー変数 e_i** — 振幅の不均一性に応じて結合強度を動的に変える。
   振幅が目標 a より大きければ e_i は小さくなり、そのパルスは結合から離脱。
   逆に小さければ e_i が成長して結合を強める。

2. **目標振幅 a(t) の動的変調** — 現在解の品質と最良解の差 ΔH に応じて目標振幅
   を `a(t) = α + ρ·tanh(δ·ΔH)` で変化させる。停滞していると a が上がって
   強い圧力がかかる。

3. **結合ランプ β_inj(t) の周期的リセット** — 結合スケールを 0 から線形に
   成長させ、一定時間改善がなければ 0 にリセット。焼きなましの再加熱に相当し、
   異なる初期ノイズから別経路で解を探す。

この 3 つが組み合わさることで、(x, e) 結合ダイナミクスが構造的な情報を使った
「カオス軌道」を描き、SA の単純ランダム探索より効率的に解空間を訪問します。

## ライセンス

MIT — 詳細は `LICENSE` を参照。

## 引用

このリポジトリを使う場合は元論文を引用してください。

```bibtex
@article{Inoue2022,
  author  = {Kyo Inoue and Kazuhiro Yoshida},
  title   = {Traveling-wave model of coherent Ising machine based on fiber loop with pulse-pumped phase-sensitive amplifier},
  journal = {Optics Communications},
  volume  = {522},
  pages   = {128642},
  year    = {2022},
  doi     = {10.1016/j.optcom.2022.128642}
}

@article{Leleu2021,
  author  = {Timothée Leleu and Farad Khoyratee and Timothée Levi and Ryan Hamerly and Takashi Kohno and Kazuyuki Aihara},
  title   = {Scaling advantage of chaotic amplitude control for high-performance combinatorial optimization},
  journal = {Communications Physics},
  volume  = {4},
  pages   = {266},
  year    = {2021},
  doi     = {10.1038/s42005-021-00768-0}
}
```
