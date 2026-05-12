# プロジェクト共通ルール

このファイルは Claude Code が自動で読み込む、本リポジトリ固有の作業ルールです。

## ディレクトリ構成

```
modules/        # アルゴリズム実装 (CIM, CAC, SB, SA, rulebase, verify)
scripts/        # 実行スクリプト (benchmarks/ tuning/ plotting/ runners/ utils/)
input/          # ベンチマーク入力 (G-set)
results/        # 実行結果 — 年月日サブフォルダで管理 ★
papers/         # 参照論文 PDF
tests/          # ユニットテスト
web/            # Web ビジュアライザ
```

## results/ 配下の管理ルール

**すべての出力 (PNG / JSON / CSV / DB / HTML / .npz 等) は `results/YYYY-MM-DD/` 配下に保存する**。

### Why
- ファイル数が増えると平坦な `results/` では履歴が追えなくなる
- 「いつの実験結果か」を一目で識別できるようにするため
- 同じファイル名 (`v1_cut_distribution.png` 等) が日をまたぐと衝突する

### How to apply

1. **新しいスクリプトを書くときは、保存先を `results/<実行日>/` にする**:

   ```python
   from datetime import date
   out_dir = Path(f"results/{date.today().isoformat()}")
   out_dir.mkdir(parents=True, exist_ok=True)
   fig.savefig(out_dir / "v1_cut_distribution.png")
   ```

2. **既存スクリプトを再実行するときは、出力パスを上記形式に書き換えてから走らせる**。

3. **バージョン番号 (`v1_, v2_, …`) は日付フォルダ内でリセットする**(別日なら別の v1 を切る)。日付フォルダ内では従来通り上書き禁止・自動採番を継続。

4. **ファイル名から日付プレフィックスは外す**。日付はフォルダ名で表現する。

   悪い例: `results/2026-05-12_v1_optuna_history.png`
   良い例: `results/2026-05-12/v1_optuna_history.png`

5. **`results/` 直下に置いてよいのは「日付フォルダ」と「全期間共通の永続資産」(`benchmark_gset.csv` 等の集計マスタ) のみ**。それ以外の単発実験成果物は必ず日付フォルダに入れる。

### 既存ファイルの扱い

整理前から `results/` 直下に置かれているファイルは、必要に応じて作成日のフォルダに移動する。Git の commit 日や `ls -la` の mtime を参考に作業日を決める。

## 図のスタイル(再掲)

`~/.claude/projects/.../memory/feedback_plot_style.md` で定義されているプロジェクト共通ルール:

- 目盛り(ひげ)は四角の **内向き** — `ax.tick_params(direction="in", which="both", top=True, right=True)`
- ラベル・タイトル・凡例は **日本語** — Windows なら `plt.rcParams["font.family"] = "Yu Gothic"` を冒頭で設定
- `plt.rcParams["axes.unicode_minus"] = False` で「−」の文字化けを抑止

## 出力上書きの禁止(再掲)

`~/.claude/projects/.../memory/feedback_no_overwrite.md`:

- 同名ファイルを上書きしない。`v{N}_…` の自動採番でバージョン分けする(本プロジェクトでは「日付フォルダ × バージョン番号」の二段構え)
- 過去の実験成果は比較・参照のために残す

## スクリプト実行規約

すべて **プロジェクトルートから** `python scripts/<カテゴリ>/<スクリプト>.py` の形で実行する。CWD = プロジェクトルートを前提とした相対パス (`input/G22.txt` 等) を採用。

## 対応プラットフォーム

**Windows のみ動作保証**。Linux / macOS との互換は考慮しない。

- 改行コード差異 (CRLF / LF) は気にしない。`.gitattributes` で `eol=lf` を強制する等の対応も不要
- パス区切り文字に `\\` を含むハードコードがあっても OK(ただし `Path` ベースで書くのが基本)
- 文字コードは UTF-8 を基本にしつつ、Windows コンソール(cp932)で見ると一部 mojibake が出る点は許容
