"""K2000 相当の SK 模型インスタンスを生成して input/K2000.txt に保存する。

仕様(Inagaki 2016 / Goto 論文と同じクラス):
  N = 2000
  all-to-all 結合 (1,999,000 辺)
  各辺の重み w_ij ∈ {-1, +1} を等確率で振る (Ising 換算で J_ij ∈ ∓1)
  seed = 0 で再現可能

注意:
  Inagaki 2016 のオリジナル K2000 ファイルは公開配布されておらず、
  これは「同クラスの独立インスタンス」である。論文の best=33337 とは
  別の値を持つ。手元で dSB を heavy run して best を推定して内部基準とする。

ファイル形式 (Gset 互換):
  1 行目: "N K"
  2 行目以降: "i j w_ij"  (i, j は 1-indexed)

サイズ: 約 20 MB(規定通り .gitignore で除外推奨、seed 同じなら再生成可能)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import time
import numpy as np


N = 2000
SEED = 0


def main():
    rng = np.random.default_rng(SEED)
    out_path = Path(__file__).resolve().parents[2] / "input" / "K2000.txt"

    t0 = time.time()

    # 上三角だけ ±1 ランダム生成。memory: N*(N-1)/2 個の int8 で約 2 MB
    n_edges = N * (N - 1) // 2
    print(f"Generating N={N}, K={n_edges} edges, seed={SEED}...")
    weights = rng.choice([-1, 1], size=n_edges).astype(np.int8)

    # (i, j) ペアを上三角で列挙してファイルに書き込む
    print(f"Writing to {out_path} ...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", buffering=1024 * 1024) as f:
        f.write(f"{N} {n_edges}\n")
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                f.write(f"{i+1} {j+1} {int(weights[idx])}\n")
                idx += 1

    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    pos = int((weights == 1).sum())
    neg = int((weights == -1).sum())
    print(f"\n[Done] {elapsed:.1f}s  size={size_mb:.1f} MB  edges={n_edges}")
    print(f"  weight balance: +1={pos} ({pos/n_edges*100:.2f}%)  "
          f"-1={neg} ({neg/n_edges*100:.2f}%)")
    print(f"\n[Path] {out_path}")
    print(f"既知最良値は未知。dSB 等で heavy run して内部基準を決めること。")


if __name__ == "__main__":
    main()
