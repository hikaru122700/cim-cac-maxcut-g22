# グラフ構造とCIMの解を理解する — 赤字反映版

2026-09-21作成。`Print (7).pdf` の手書き指摘を反映した、16ページ・図18点の日本語資料。

## ファイル

- `report.pdf`: 閲覧用PDF。
- `report.tex`: LuaLaTeXソース。図1・13・15・17・18はTikZで記述。
- `latex_sources.zip`: TeX、使用図、図生成コード、データ、対応記録のソース一式。
- `redline_response.md`: 赤字への対応一覧。
- `make_figures_base.py`: 旧版から引き継いだ説明図と構造集計図の生成コード。
- `make_revision_figures.py`: 今回追加した説明図、次数分布、既存実験・文献の再描画。
- `data/`: 構造集計、次数分布と確認済み閉路、説明用グラフの固有値。

## 主な改訂

橋・関節点の除去前後と、片側全反転によるMax-Cutの加法分解を図解した。次数とcore numberを区別し、実際の入力から次数分布を数え直した。奇数閉路と固有値の説明を追加し、G70に存在する5頂点閉路を確認した。MQLibとECO-DQNは公開本文を読んで詳説し、論文で確認された結果と本資料からの応用案を分けた。候補の将来価値・選別器の詳細設計は、今回の本文から外した。

## ページ案内

1: 表紙・読み方。2: Max-Cutの基本。3–4: 橋・関節点と分解。5–6: 次数・coreと色の意味。7–8: 構造統計・次数分布・既存GA結果。9: 奇数閉路。10–11: 固有値・CIM。12: MQLib。13–14: ECO-DQN。15: 当面の研究範囲。16: 文献・再現情報。

## PDFの再ビルド

Windows、MiKTeXのLuaLaTeXを使用。Windows標準フォントのYu Mincho、Yu Gothic、Times New Roman、Arialを`C:/Windows/Fonts/`から参照する。図PDFは同梱しているため、本文の再ビルドにはPythonは不要。

このフォルダで次を2回実行する。

```powershell
lualatex -interaction=nonstopmode -halt-on-error report.tex
lualatex -interaction=nonstopmode -halt-on-error report.tex
```

公開済み版の変更時は、このフォルダを新しいv番号へコピーしてから作業する。図を再生成する場合は、Pythonのnumpy、networkx、matplotlibとYu Gothicフォントが必要。リポジトリルートから、新版フォルダ内の`make_figures_base.py`、続いて`make_revision_figures.py`を実行する。元コードは未使用の旧版図も生成するが、ソースZIPには本文で使用する図だけを収録した。

## データと解釈

- `data/summary.json`は既存の構造監査結果。入力ファイルのSHA-256も記録されている。`python scripts/utils/graph_structure_audit.py`で別バージョンに再集計できる。
- `data/degree_and_cycle.json`は`input/G22.txt`、`G55.txt`、`G70.txt`から今回数えた次数分布。次数の合計が辺数の2倍であることを確認。G70の閉路は318 → 5286 → 3866 → 8561 → 6210 → 318で、5辺すべてが入力に存在する。
- 図10は`results/2026-09-14/ga_diversity_trajectory/v1_G22_G55_G70_nt16_gen300_measured/REPORT.md`の平均最終gapをBKSで割った値。GAの既存結果であり、CIM単独の難易度測定ではない。
- 図14の固有値は説明用の小グラフの計算値。G-setのスペクトルを新たに測定した結果ではない。
- 図16はMQLib論文Section 6の報告値を再描画したもの。本リポジトリでの再実験ではない。
- 新たなCIM・GA・SA・学習モデルの性能実験は行っていない。

## 確認

LuaLaTeXで2回ビルドし、PDF全16ページを画像として目視確認。ページからの文字抽出と図番号1–18を確認。元PDFとv1は保持した。`qa/`、`review_input/`、ビルドログは作業用で、配布ZIPには含めない。
