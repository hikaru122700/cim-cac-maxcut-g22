# グラフ構造とCIM候補評価の図解レビュー

- `report.pdf`: 完成版、10ページ、図11点。
- `report.tex`: LuaLaTeXの本文ソース。
- `figures/`: PDF形式のベクトル図。TikZ図はTeX本文に含む。
- `make_figures.py`: グラフ・模式図の生成スクリプト。
- `data/summary.json`: 図に用いた構造集計値、入力SHA256付き。

## TeXからPDFを生成

Windows上のLuaLaTeXを使用。本文の日本語は游明朝、見出しは游ゴシック、欧文はTimes New RomanとArialを使用する。フォントはWindows標準の `C:/Windows/Fonts/` を参照する。図のPDFは同梱しているため、Pythonを実行せずにTeXをコンパイルできる。

成果物を保持するため、編集する場合はこのフォルダを別のバージョンへコピーしてから作業する。コピー先で次を2回実行する。

```powershell
lualatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex
lualatex -interaction=nonstopmode -halt-on-error -file-line-error report.tex
```

必要な主なTeXパッケージ: luatexja, fontspec, geometry, amsmath, amssymb, graphicx, booktabs, tabularx, pgf/TikZ, tcolorbox, fancyhdr, enumitem, hyperref。

図を修正する場合は、Pythonのmatplotlib、numpy、networkxを利用して `python make_figures.py` を実行し、再コンパイルする。図は実測値を使う比較図と、説明用の模式図に分けて本文で明記している。学習モデルの性能実験は行っていない。

## 確認

全ページをPopplerでPNG化し、文字・図・表・改ページを目視確認した。LaTeXのOverfull/Underfull警告と欠落文字をチェックした。PDF内の文献リンクと日本語テキスト抽出を確認した。

調査の元資料: `docs/20260921/graph_structure_literature_v1.md`。
入力の構造計算: `scripts/utils/graph_structure_audit.py`。
確認したリポジトリHEAD: `9053ca4`（2026-09-14）。
