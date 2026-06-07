---
name: experiment-management
description: 実験の一連の管理ワークフロー(cim-cac-maxcut-g22 用)。新しい実験を始めるとき、実験を回すとき、結果をまとめるときに使う。手順は①実験コードを日付バージョン付きで作成 → ②出力を日付×実験種別×バージョンのディレクトリへ格納 → ③実験完了後に RESULTS.md へ解説を執筆 → ④md_to_pdf.py で PDF 化。Use when starting/running an experiment, organizing its outputs, or writing up and exporting results.
---

# 実験管理ワークフロー

このプロジェクトの実験は **4 ステップ** で管理する。日付とバージョンを軸に、
コード・出力・解説・PDF を 1 実験 = 1 まとまりとして残す。**過去の成果は上書き
せず必ず残す**(比較・再現のため)。すべてプロジェクトルートから実行する。

関連: [[cim-cac-maxcut-patterns]](results 規約・プロット規約・Numba 規約の詳細)。

---

## ステップ 1 — 実験コードを日付バージョン付きで作成

新しい実験ロジックは `modules/<YYYY-MM-DD>_<NAME>[_v{N}][_<variant>].py` で作る。
ファイル名そのものに **実験日**と**版**を刻む。

実例(既存):
```
modules/2026-05-29_CIM_PT.py
modules/2026-06-06_CIM_PT_v2.py
modules/2026-06-06_CIM_PT_v3.py
modules/2026-06-08_CIM_PT_v3_swap_ablation.py
```

- `<YYYY-MM-DD>` … 着手日(`date.today().isoformat()`)。
- `<NAME>` … 手法名(`CIM_PT` 等、UpperSnake)。
- `_v{N}` … 同手法の改良版。前版から派生したら番号を上げる(`_v2`, `_v3`)。
- `_<variant>` … 派生実験(`_swap_ablation` 等)。

> **注意**: 日付始まりのファイル名は `import` 不可。他スクリプトから使うときは
> `importlib` でロードする(詳細は [[cim-cac-maxcut-patterns]] の「Date-prefixed module」)。
> ```python
> _spec = importlib.util.spec_from_file_location("mod", ROOT / "modules" / "2026-06-06_CIM_PT_v3.py")
> mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mod)
> ```

スクリプト冒頭で `EXPERIMENT_KIND` 定数を必ず定義する(ステップ2の出力先に使う):
```python
EXPERIMENT_KIND = "cim_pt_v3"   # 英小文字 snake_case、出力フォルダ名になる
```

---

## ステップ 2 — 出力を日付バージョンのディレクトリへ格納

すべての出力(PNG / JSON / CSV / .npz / DB / HTML)は **3 段構造**へ保存する。
出力フォルダもコードと同じく日付とバージョンで切る。

```
results/<YYYY-MM-DD>/<EXPERIMENT_KIND>/v{N}_<description>/<file>
```

実例:
```
results/2026-06-06/cim_pt_v3/v1_rounds1500_swap10_perreplica_ramp/
    ├── data.npz
    ├── hist.png
    ├── running_best.png
    ├── ramps_amplitude.png
    ├── RESULTS.md      ← ステップ3
    └── RESULTS.pdf     ← ステップ4
```

- `v{N}` は **実験種別ごと**に自動採番(日付が変われば再び v1)。
- `<description>` は **CLI 引数から自動生成**(`rounds1500_swap10_...`)。`--tag` で任意の
  サフィックスを足せるようにする。
- **ファイル名から実験名・グラフ名の prefix を外す**(フォルダ名で表現済み)。
  良い例 `hist.png` / 悪い例 `cim_pt_v3_G22_hist.png`。

採番ヘルパー(コピーして使う):
```python
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[N]   # 階層に応じて調整

def next_version(kind_root: Path) -> int:
    max_v = 0
    if kind_root.exists():
        for p in kind_root.iterdir():
            if p.is_dir() and p.name.startswith("v"):
                head = p.name.split("_", 1)[0]
                if head[1:].isdigit():
                    max_v = max(max_v, int(head[1:]))
    return max_v + 1

kind_root = ROOT / "results" / date.today().isoformat() / EXPERIMENT_KIND
v = next_version(kind_root)
out_dir = kind_root / f"v{v}_{description}"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "hist.png")
```

プロット保存時は共通スタイルを守る(Agg / Yu Gothic / `unicode_minus=False` /
内向き目盛り)。詳細は [[cim-cac-maxcut-patterns]]。

---

## ステップ 3 — 実験完了後、RESULTS.md に解説を執筆

実験が終わったら、その出力ディレクトリ直下に `RESULTS.md` を書く。図は **相対パス**
(`./hist.png`)で参照する(ステップ4の PDF 化で絶対 URI に変換される)。

数式は `$...$`(インライン)/ `$$...$$`(ディスプレイ)で書く。MathJax で描画される。

**RESULTS.md の標準構成**(既存 `cim_pt_v3/v1_*/RESULTS.md` に準拠):

```markdown
# <実験タイトル> — <グラフ名>

実験日: <YYYY-MM-DD> / グラフ: **G22**(N=2000, 辺19990) / 既知ベスト: **13359**
再現コード: `modules/<YYYY-MM-DD>_<NAME>.py`
データ: `results/<YYYY-MM-DD>/<KIND>/v{N}_<desc>/`

---

## 1. この実験で何を試したか(一言で)
> 端的な要約(引用ブロックで強調)

## 2. 数式の変更点(必須)

このプロジェクトの実験は CIM/CAC の**基準更新方程式**のどこかを改変するもの。
レポートには **基準式の画像を必ず貼り**、「どの記号・項を、なぜ、どう変えたか」を
基準式 → 変更後の順で明記する。

![基準更新方程式](../../../../docs/assets/base_update_equations.png)

$$\mathbf{F}(n) = a\mathbf{E}(n-1) + b\mathbf{J}\mathbf{E}(n-1) + c\mathbf{n}_0$$
$$\mathbf{E}(n) = \mathbf{F}(n)\exp\!\left[\alpha\,p(n)\bigl(1 - \beta|\mathbf{F}(n)|^2\bigr)\right]$$

**この実験での変更:**

| 記号/項 | 基準式 | 変更後 | 変更理由 |
|---|---|---|---|
| `p(n)`(ポンプ) | 全レプリカ共通の固定/単一ランプ | `p_r(n)=mult_r·p_ramp(n)` の各レプリカ独立ランプ | 各レプリカに焼きなまし(時間発展)を持たせるため |
| `(該当記号)` | ... | ... | ... |

> 変更がない記号は表に載せない。変更した式は変更後の形を $$...$$ で再掲する。

## 3. 実験条件
| 項目 | 値 |
|---|---|
| グラフ | G22(N=2000, 辺19990) |
| ラウンド数 | ... |
| 試行数 | ... |

## 3. 結果サマリー
| 手法 | 平均カット | 最良カット | 標準偏差 | 既知ベスト比 |
|---|---|---|---|---|
| baseline | ... | ... | ... | ... |
| **本手法** | **...** | **...** | **...** | **...** |

## 4. 図と読み取り
![累積最良カット](./running_best.png)
- 横軸/縦軸の説明と、図から読み取れること

## 5. 考察 — 何が効いたのか
（なぜその結果になったかの物理的・アルゴリズム的理由）

## 6. 注意点・今後の課題
- 統計的限界(単一インスタンス・seed 数など)
- 次に試すこと
```

執筆のコツ:
- 解説・ラベルは**日本語**。物理コードは論文の式番号(`Eq.(3)`)を併記。
- 結論を最初に(セクション1で一言要約)、根拠は図と表で示す。
- baseline との差分を必ず数値で書く(「平均 +17.7・最良 +11」のように)。
- 過大評価しない注意書き(セクション6)を入れる。

---

## ステップ 4 — RESULTS.md を PDF 化

`scripts/utils/md_to_pdf.py` で PDF を生成する。Python-Markdown → MathJax →
Chrome ヘッドレス印刷の経路(LaTeX 不要・日本語/数式対応・Windows 専用)。

```powershell
# プロジェクトルートから。出力先を省略すると同じ場所に .pdf を作る
python scripts/utils/md_to_pdf.py results/<date>/<kind>/v{N}_<desc>/RESULTS.md
```

挙動:
- 図の相対パス `./hist.png` を絶対 `file://` URI に自動変換するので PDF に画像が埋まる。
- `$...$` / `$$...$$` は MathJax-SVG で描画(`_` の `<em>` 誤変換は内部で退避済み)。
- 完成すると `RESULTS.pdf` が同じディレクトリに出力される。

> 中間生成物の `RESULTS_tmp.html` がコミットされている既存例があるが、PDF さえ
> 残れば一時 HTML は不要。

---

## チェックリスト(実験 1 サイクル)

- [ ] 実験コードを `modules/<YYYY-MM-DD>_<NAME>[_v{N}].py` で作成、`EXPERIMENT_KIND` 定義
- [ ] 出力を `results/<date>/<kind>/v{N}_<desc>/` へ(`next_version` で採番、**上書き禁止**)
- [ ] ファイル名から実験名・グラフ prefix を外す / プロットは共通スタイル
- [ ] 実験完了後、出力ディレクトリ直下に `RESULTS.md`(標準構成・相対パス画像)を執筆
- [ ] `python scripts/utils/md_to_pdf.py .../RESULTS.md` で `RESULTS.pdf` を生成
