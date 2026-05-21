# プロジェクト共通ルール

このファイルは Claude Code が自動で読み込む、本リポジトリ固有の作業ルールです。

## ディレクトリ構成

```
modules/        # アルゴリズム実装 (CIM, CAC, SB, SA, rulebase, verify)
scripts/        # 実行スクリプト (benchmarks/ tuning/ plotting/ runners/ utils/)
input/          # ベンチマーク入力 (G-set, K2000)
results/        # 実行結果 — 年月日サブフォルダで管理 ★
papers/         # 参照論文 PDF
docs/           # アルゴリズム解説・設計メモなど(md ファイル)
tests/          # ユニットテスト
web/            # Web ビジュアライザ
tools/          # ポータブル外部ツール (gcc, rudy など)
```

## results/ 配下の管理ルール

**すべての出力 (PNG / JSON / CSV / DB / HTML / .npz 等) は次の 3 段階構造で保存する**:

```
results/<YYYY-MM-DD>/<実験種別>/v{N}_<簡潔な説明>/<ファイル>
```

### Why
- ファイル数が増えると平坦な `results/` では履歴が追えなくなる
- 「いつ・何の実験・どんな設定か」を一目で識別できるようにするため
- 同じファイル名 (`hist.png` 等) が衝突せず、1 run = 1 ディレクトリで完結する
- v 採番は日付フォルダ × 実験種別 × バージョン番号の 3 軸でリセットされる

### 構成要素

1. **`<YYYY-MM-DD>`**: 実行日(date.today().isoformat())。
2. **`<実験種別>`**: スクリプトに `EXPERIMENT_KIND` 定数で固定する短い英小文字 snake_case 名。同じスクリプトの全 run はこの直下に並ぶ。例:
   - `cim_vs_pticm`
   - `cim_optuna_rounds_sweep`
   - `cim_optuna_reduced`
3. **`v{N}_<説明>`**: その実験種別内でのバージョン番号 + run の内容を表す簡潔な説明 (snake_case)。説明は CLI 引数から自動生成すること(例: `v5_sweeps1500_NT16_trajectory`、`v2_4cond_300trial`)。`--tag` で追加サフィックスを付けられるようにする。

### How to apply (新規スクリプト)

```python
EXPERIMENT_KIND = "my_experiment"

def get_kind_root() -> Path:
    out = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    out.mkdir(parents=True, exist_ok=True)
    return out

def next_version(kind_root: Path) -> int:
    max_v = 0
    for p in kind_root.iterdir():
        if p.is_dir() and p.name.startswith("v"):
            head = p.name.split("_", 1)[0]
            if head[1:].isdigit():
                max_v = max(max_v, int(head[1:]))
    return max_v + 1

def build_description(args) -> str:
    parts = [f"sweeps{args.sweeps}"]            # 主要パラメータ
    if args.tag: parts.append(args.tag)         # 任意タグ
    return "_".join(parts)

kind_root = get_kind_root()
v = next_version(kind_root)
out_dir = kind_root / f"v{v}_{build_description(args)}"
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / "hist.png")               # ファイル名から実験名 prefix を外す
```

### 命名規約

- **ファイル名から実験名・graph 名の prefix を外す**。それらはフォルダ名で表現する。
  - 悪い例: `cim_vs_pticm/v5_*/v5_compare_cim_pticm_G22_hist.png`
  - 良い例: `cim_vs_pticm/v5_sweeps1500_NT16_trajectory/hist.png`
- **`<説明>` は半角英数字 + アンダースコアのみ**。スペース・全角・記号は避ける。日本語ファイル名は使わない。
- **v 採番は `<実験種別>` 単位**で 1 から始める。日が変われば別の日付フォルダ配下で再度 v1 から始まる。

### 既存ファイルの扱い

整理前から平坦に置かれているファイルは、`git mv` で `<実験種別>/v{N}_<説明>/` 配下へ移行する。`<説明>` は当時の主要パラメータから事後的に決める(例: smoke test の v1 → `v1_smoke`)。

### `results/` 直下に置いてよいもの

「日付フォルダ」と「全期間共通の永続資産」(`benchmark_gset.csv` 等の集計マスタ、`optuna_cim_study.db` 等の永続ストレージ) のみ。それ以外の単発実験成果物は必ず日付フォルダ + 実験種別フォルダの下に入れる。

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
