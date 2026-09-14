# CIM出力解の多様性：既存実験の再解析

report.pdf：A4横、5ページの説明資料。
report.tex：編集可能なLuaLaTeXソース。
assets/：資料で使うグラフと提供スクリーンショット。

## 再コンパイル
このディレクトリで以下を実行してください。

lualatex --interaction=nonstopmode --halt-on-error report.tex
lualatex --interaction=nonstopmode --halt-on-error report.tex

LuaLaTeX、luatexja、luatexja-fontspecなどが必要です。
フォントはWindowsのArialと游ゴシックを使用しています。
別環境ではreport.tex冒頭のフォント設定を調整してください。

## 内容の位置づけ
保存済みのCIM生成解（共通Tabu Search後）を再解析した資料です。
同品質部分集合でも多様性が異なることは確認できますが、
多様性がGA性能を改善する因果効果はこの資料からは結論できません。
GA収束図は提供画像をそのまま掲載しています。

## 検証
LuaLaTeXでコンパイルし、全5ページを画像化して確認しました。
グラフの集計・抽出データと再解析スクリプトは、同日付の
cim_diversity_evidence/v2_G22_G55_G70_matched_r1000_discussion/
に保存されています。
