"""Optuna study から評価値の推移を折れ線で描く。

env:
  STUDY_NAME  : (必須) optuna study 名
  STORAGE     : sqlite URL (既定: sqlite:///results/optuna_cim_study.db)
  OUT_PATH    : 出力 PNG (既定: results/<study>_line.png)
  TITLE       : グラフタイトル (既定: study 名)
  REF_PAPER   : 参照線(論文 mean) (既定: 13275)
  REF_BEST    : 参照線(既知最良) (既定: 13359)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna

plt.rcParams["font.family"] = "Yu Gothic"
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    study_name = os.environ.get("STUDY_NAME")
    if not study_name:
        print("STUDY_NAME を指定してください。例: STUDY_NAME=cim_g22_reduced_nr3000_paperfix_nr3000")
        sys.exit(2)

    storage = os.environ.get("STORAGE", "sqlite:///results/optuna_cim_study.db")
    title = os.environ.get("TITLE", study_name)
    ref_paper = float(os.environ.get("REF_PAPER", 13275))
    ref_best = float(os.environ.get("REF_BEST", 13359))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = sorted(study.trials, key=lambda t: t.number)
    values = np.array(
        [t.value if t.value is not None else np.nan for t in trials], dtype=float
    )
    idx = np.arange(1, len(values) + 1)

    running_best = np.fmax.accumulate(np.where(np.isnan(values), -np.inf, values))
    running_best = np.where(np.isfinite(running_best), running_best, np.nan)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=130)

    ax.plot(
        idx, values, color="#1f77b4", linewidth=0.9, alpha=0.65,
        label="各試行の mean_cut",
    )
    ax.plot(
        idx, running_best, color="#d62728", linewidth=2.2,
        label="これまでの最良 mean_cut",
    )
    ax.axhline(
        ref_paper, color="black", linestyle=":", linewidth=1.2,
        label=f"論文 Fig.8 平均 {ref_paper:.0f}",
    )
    ax.axhline(
        ref_best, color="goldenrod", linestyle="--", linewidth=1.3,
        label=f"既知最良値 {ref_best:.0f}",
    )

    best_val = float(np.nanmax(values))
    ax.set_xlabel("Optuna 試行番号")
    ax.set_ylabel("Optuna に返したスコア (mean best_cut)")
    ax.set_title(f"{title}\n最終 best = {best_val:.2f} / {len(values)} trials")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.tight_layout()

    default_out = f"results/{study_name}_line.png"
    out_path = Path(os.environ.get("OUT_PATH", default_out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
