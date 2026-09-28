# G22・G55・G70の構造からCIMの解をどう理解するか

2026-09-22、v2。ユーザー指定の `Print (8).pdf` の赤字を修正指示として反映した資料。説明順序を組み直した前版を照合し、文章に偏っていた説明を具体的な図へ置き換えた。全16ページ・図21点。

## 閲覧・編集

- `report_20260922_v2.pdf`：閲覧用。日付と版をファイル名にも明記。
- `report.tex` と `preamble.tex`：本文とLuaLaTeX設定。
- `latex_sources.zip`：本文、設定、使用図、生成コード、データ、対応記録。
- `redline_response.md`：手書き指摘と反映ページの一覧。

## 構成

1. 研究目的と読む道順。
2. Max-Cutの得点、記号、全反転の意味。
3–4. 橋・関節点と、4+4+1から理解する最適値の分解。
5. G55・G70の中心部分、付いた木、離れた木、孤立点。
6–7. 木の除去から2-coreを導入し、次数とcoreを区別。
8–11. 実測の比較、奇数閉路、固有値、CIM。
12–14. MQLibとECO-DQNの方法・結果・今回の使い道。
15–16. 当面の研究課題、文献と再現情報。

## 新たに確認した内訳

G55は、中心部分4,789頂点・12,318辺、付いた木180頂点・180辺、孤立点31個。離れた木はない。

G70は、中心部分4,798頂点・6,394辺、付いた木3,215頂点・3,215辺、離れた木243個（計633頂点・390辺）、孤立点1,354個。橋の合計は3,215+390=3,605本。入力からNetworkX 3.6.1で確認し、`data/decomposition.json`に保存した。

図6はこの内訳の図解で、数値は実測、形や描く枝の本数は説明用。図7–7は小グラフの説明例。既存GA結果は9月14日の記録を参照し、今回ソルバーの性能を再測定していない。

## ビルド

Windows / MiKTeX LuaLaTeX。`C:/Windows/Fonts/`のYu Mincho、Yu Gothic、Times New Roman、Arialを使用する。図PDFを同梱しているため、再ビルドだけならPython不要。このフォルダで次を実行する。

```powershell
lualatex -jobname=report_20260922_v2 -interaction=nonstopmode -halt-on-error report.tex
lualatex -jobname=report_20260922_v2 -interaction=nonstopmode -halt-on-error report.tex
```

公開済み版を改訂するときは、新しい日付・v番号のフォルダを作り、そちらで作業する。

## 図とデータの再現

Pythonのnumpy、networkx、matplotlib、およびYu Gothicが必要。入力を再集計する第3スクリプトは、リポジトリルートを作業ディレクトリとして実行する。

1. `make_figures_base.py`：引き継いだMax-Cut例・閉路の図など。
2. `make_revision_figures.py`：次数分布、橋の反転、固有値の例、MQLibの報告値の図など。
3. `make_second_revision_figures.py`：今回の分解の再集計と、分解図・除去の途中経過の図。
4. `make_instruction_figures.py`：L・R・wを図に記した橋の反転、辺を共有する奇数閉路、1頂点の反転gain。最後に実行する。小グラフの得点も検算する。

図のスクリプトは各版のフォルダに図を生成する。生成先はスクリプトの場所から決まり、旧版の図を変更しない。ZIPは本文で参照する図だけを収録しているが、元スクリプトを実行すると未使用の旧版図も生成される。

`data/summary.json`は既存の構造監査（入力SHA-256を含む）、`degree_and_cycle.json`は実入力の次数分布と確認したG70の奇数閉路、`toy_spectrum.json`は説明用グラフの固有値。`decomposition.json`だけが今回新規の実入力集計である。

## 品質確認

LuaLaTeXを2回実行し、ページ数・図番号・参照ファイル・ログを確認。PopplerでPDFを画像化し、全ページの文字・図・改ページを目視確認した。G55・G70の分解は頂点数・辺数の合計と橋の本数を照合した。

元の赤字PDFと旧版は変更していない。`review_input/`、`qa/`、LaTeXログは作業用で、ソースZIPに含めない。
