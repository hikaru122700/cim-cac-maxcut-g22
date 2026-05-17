# MAX-CUT Visualizer (Vercel 静的 Web アプリ)

Gset 形式のグラフファイル + N 行 0/1 割当ファイルをブラウザ上にアップロードすると、
cut 値、パーティション、局所最適性を可視化します。

**全てクライアント側で処理**されるため、ファイルは外部に送信されません。

## ローカル開発

```bash
cd web
npm install
npm run dev
```

Vite dev server (http://localhost:5173) が立ち上がります。

## ビルド

```bash
npm run build
# 出力: web/dist/
```

## Vercel へのデプロイ

### 方法 1: CLI

```bash
cd web
npx vercel            # プレビューデプロイ
npx vercel --prod     # 本番デプロイ
```

Vercel は `vercel.json` を読み、Vite として自動認識します。

### 方法 2: GitHub 連携

1. https://vercel.com/new で GitHub リポジトリを import
2. Root directory を `web` に設定
3. Framework Preset: Vite (自動検出される)
4. Deploy

## 入力ファイル

### 1. Graph file (Gset 形式)

```
N K
u1 v1 w1
u2 v2 w2
...
```

- 1 行目: 頂点数 N と辺数 K
- 続く K 行: 辺 (u, v) と重み w (省略時は 1)
- **1-indexed** (Gset 慣習)
- 例: リポジトリ同梱の `input/G22.txt`

### 2. Assignment file (0/1 N 行)

```
0
1
1
0
...
```

- 各行に 0 または 1 (もしくは -1/+1)
- **0-indexed** (頂点 i の割当が i 行目)
- 空白区切りで 1 ファイルに並べても可
- `scripts/save_assignment.py` の出力形式

### Assignment ファイルの生成

リポジトリルートで:

```bash
uv run python -m scripts.save_assignment
# → results/assignments/G22_cac_seed0.txt
```

生成された txt をこの Web アプリにアップロード。

## 表示される指標

| 指標 | 意味 |
|---|---|
| Cut value | 2 パーティションを跨ぐ辺の本数 |
| Cut ratio | cut / 総辺重み |
| +1 / −1 | 各パーティションの頂点数 |
| Balance | 2 パーティションの頂点差 (\|+1 − −1\|) |
| Local improvable | フリップで cut が純増する頂点数 (局所最適からの距離) |
| Gap to BKS | 既知最良解との差 (G22 のとき BKS=13359 を自動適用) |

## 視覚化モード

### Spin Grid — 2D グリッド (index 順)

- **partition**: 青 = +1 側, 赤 = −1 側
- **cut-degree**: 明度 = 各頂点の cut 次数 / 次数 (境界の頂点ほど明るい)
- **improvable**: 黄色 = フリップで cut が増える頂点 (局所最適でない)

### Cut-degree distribution

各頂点が相手パーティションへ伸ばす辺の本数のヒストグラム。
右に裾が長いほど「境界」の頂点が多く、左に寄るほど内側で閉じた解。

## 技術スタック

- Vite + React 18 + TypeScript (strict)
- Plotly は使わず、HTML5 Canvas2D のみで描画 (バンドル最小化)
- 外部依存なし (CDN フォールバック含めて zero network call)
