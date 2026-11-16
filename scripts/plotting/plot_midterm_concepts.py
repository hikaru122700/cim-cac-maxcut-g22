"""卒論中間報告の資料に入れるイメージ図（概念図）を描く。

  ising.png    スピンの向き＝頂点の集合。向きが逆の辺がカットされる（6 頂点の例）
  sa.png       焼きなまし法（SA）：エネルギーの山谷を、温度を下げながら動く 1 つの解（実際に Metropolis 法で動かした軌跡）
  ga.png       遺伝的アルゴリズム（GA）：集団 → 親を選ぶ → 交叉 → 突然変異・局所探索 → 入れ替え
  goal.png     同じ総時間での比較：ランダム開始と CIM 開始の時間配分と、解の質の推移（概念図）
  cim_loop.png CIM の仕組み：光パルスがループを周回し、増幅・測定・結合の注入をくり返す
  valley.png   考察：最良解（BKS 解）との距離 d。CIM の解・ランダム解と SA 後の位置（G22 の実測の d を半径に使う）

出力先は results/<今日>/midterm_concept_figs/v{N}_figs/（上書きしない）。

usage:
  python scripts/plotting/plot_midterm_concepts.py
"""

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "mathtext.fontset": "stix",
    "axes.labelsize": 20, "axes.titlesize": 20, "legend.fontsize": 18,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "font.size": 18,
})
EXPERIMENT_KIND = "midterm_concept_figs"
BLUE, RED, GRAY, BLACK, ORANGE, GREEN = "#1f5fbf", "#d62728", "#9a9a9a", "#222222", "#ff7f0e", "#2ca02c"
# G22 の実測（卒論中間報告 v34 の結果③）：最良解（BKS 解）までの距離 d の平均
D_CIM_START, D_CIM_SA, D_RANDOM_START, D_RANDOM_SA = 0.16, 0.14, 0.49, 0.28


def out_dir() -> Path:
    root = Path("results") / date.today().isoformat() / EXPERIMENT_KIND
    root.mkdir(parents=True, exist_ok=True)
    vs = [int(p.name.split("_", 1)[0][1:]) for p in root.iterdir()
          if p.is_dir() and p.name.split("_", 1)[0][1:].isdigit()]
    out = root / f"v{max(vs, default=0) + 1}_figs"
    out.mkdir()
    return out


def save(fig, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


def arrow(ax, a, b, color=BLACK, lw=2.5, style="-|>", ms=18, **kw) -> None:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, **kw))


# ---------------------------------------------------------------- イジング
def fig_ising(out: Path) -> None:
    pos = {0: (0, 1), 1: (1.2, 1.5), 2: (2.4, 1), 3: (0, -0.2), 4: (1.2, -0.7), 5: (2.4, -0.2)}
    spin = {0: 1, 1: -1, 2: 1, 3: -1, 4: 1, 5: -1}
    edges = [(0, 1), (1, 2), (0, 3), (2, 5), (3, 4), (4, 5), (1, 4), (0, 4), (2, 4)]
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    for i, j in edges:
        cut = spin[i] != spin[j]
        (x1, y1), (x2, y2) = pos[i], pos[j]
        ax.plot([x1, x2], [y1, y2], color=RED if cut else GRAY, lw=4 if cut else 2, ls="-" if cut else "--", zorder=1)
    for v, (x, y) in pos.items():
        col = BLUE if spin[v] > 0 else RED
        ax.add_patch(Circle((x, y), 0.28, color="white", ec=col, lw=3, zorder=2))
        arrow(ax, (x, y - 0.2 * spin[v]), (x, y + 0.2 * spin[v]), color=col, lw=3.5, ms=20, zorder=3)
    tx = 3.2
    ax.text(tx, 1.45, "↑ スピン +1 ＝ 集合 A", color=BLUE, fontsize=20, va="center")
    ax.text(tx, 0.85, "↓ スピン −1 ＝ 集合 B", color=RED, fontsize=20, va="center")
    ax.plot([tx, tx + 0.5], [0.2, 0.2], color=RED, lw=4)
    ax.text(tx + 0.65, 0.2, "向きが逆の辺 ＝ カット", color=RED, fontsize=20, va="center")
    ax.plot([tx, tx + 0.5], [-0.4, -0.4], color=GRAY, lw=2, ls="--")
    ax.text(tx + 0.65, -0.4, "向きが同じ辺", color=BLACK, fontsize=20, va="center")
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylim(-1.1, 1.9)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out / "ising.png")


# ---------------------------------------------------------------- SA
def landscape(x: np.ndarray) -> np.ndarray:
    return (0.9 * np.sin(x / 9.0) + 0.55 * np.sin(x / 3.3 + 1.0) + 0.25 * np.sin(x / 1.6)
            - 1.6 * np.exp(-((x - 128) / 16.0) ** 2) + 0.0015 * (x - 100) ** 2 / 10)


X0 = 30


# seed は山を越えて一番深い谷に着く例（0〜39 のうち 0・6・9・10・11・23・31 が着く）
def sa_walk(seed: int = 0, steps: int = 3000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    grid = np.arange(0, 200)
    e = landscape(grid.astype(float))
    t_sched = 1.5 * (0.02 / 1.5) ** (np.arange(steps) / (steps - 1))
    x, path = X0, []
    for t in t_sched:
        y = int(np.clip(x + rng.choice([-3, -2, -1, 1, 2, 3]), 0, 199))
        d = e[y] - e[x]
        if d <= 0 or rng.random() < np.exp(-d / t):
            x = y
        path.append(x)
    return np.array(path), t_sched


def fig_sa(out: Path) -> None:
    grid = np.linspace(0, 199, 800)
    path, t_sched = sa_walk()
    fig = plt.figure(figsize=(7.6, 4.4))
    ax = fig.add_axes([0.0, 0.0, 0.66, 1.0])
    ax.plot(grid, landscape(grid), color=BLACK, lw=2.5, zorder=1)
    idx = np.arange(0, len(path), 25)
    xs = path[idx].astype(float)
    sc = ax.scatter(xs, landscape(xs) + 0.12, c=idx, cmap="coolwarm_r", s=40, zorder=2, edgecolor="none")
    def e(v: float) -> float:
        return float(landscape(np.array([float(v)]))[0])

    ax.plot(path[-1], e(path[-1]) + 0.12, "o", ms=16, color=BLUE, mec=BLACK, zorder=3)
    ax.plot(X0, e(X0) + 0.12, "o", ms=16, color=RED, mec=BLACK, zorder=3)
    ax.text(X0, e(X0) + 0.5, "初期解", ha="center", fontsize=20, color=RED)
    ax.text(path[-1], e(path[-1]) - 0.6, "最後の解", ha="center", fontsize=20, color=BLUE)
    ax.annotate("悪くなる反転も確率で\n受け入れ → 山を越える", xy=(64, e(64) + 0.2), xytext=(120, 2.55), fontsize=16,
                ha="center", va="center", arrowprops=dict(arrowstyle="-|>", color=RED, lw=2), color=RED)
    ax.set_xlabel("解（スピン配置）")
    ax.set_ylabel("エネルギー H（−カット値）")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 199)
    ax.set_ylim(-2.4, 3.3)
    cax = ax.inset_axes([0.73, 0.05, 0.22, 0.035])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=[])
    cb.outline.set_visible(False)
    cax.text(-0.03, 0.5, "序盤", transform=cax.transAxes, ha="right", va="center", fontsize=15, color=RED)
    cax.text(1.03, 0.5, "終盤", transform=cax.transAxes, ha="left", va="center", fontsize=15, color=BLUE)
    ax2 = fig.add_axes([0.78, 0.55, 0.22, 0.4])
    ax2.plot(np.arange(len(t_sched)), t_sched, color=BLACK, lw=2.5)
    ax2.set_title("温度 T を下げる", fontsize=18)
    ax2.set_xlabel("ステップ", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(0.12, 0.88, "高温：大きく動く", transform=ax2.transAxes, fontsize=13, color=RED, va="top")
    ax2.text(0.97, 0.2, "低温：谷に\n落ち着く", transform=ax2.transAxes, fontsize=13, color=BLUE, ha="right")
    fig.text(0.89, 0.27, r"$P_{\mathrm{accept}}=\min\left(1,\ e^{-\Delta H/T}\right)$", ha="center", fontsize=19)
    fig.text(0.89, 0.13, "ΔH：反転による H の増分", ha="center", fontsize=14)
    save(fig, out / "sa.png")


# ---------------------------------------------------------------- GA
def bits(ax, x, y, s, w=0.32, h=0.42, mark=None, edge=BLACK) -> None:
    for k, b in enumerate(s):
        col = BLUE if b == "1" else (RED if b == "0" else "#dddddd")
        ax.add_patch(Rectangle((x + k * w, y), w, h, facecolor=col, edgecolor="white", lw=1.5))
        if mark and k in mark:
            ax.add_patch(Rectangle((x + k * w, y), w, h, facecolor="none", edgecolor=ORANGE, lw=3.5, zorder=3))
    ax.add_patch(Rectangle((x, y), w * len(s), h, facecolor="none", edgecolor=edge, lw=1.5))


def fig_ga(out: Path) -> None:
    pop = ["1101001011", "0110110100", "1100011011", "0011101100", "1011001010"]
    cuts = [13, 11, 14, 10, 12]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    # 集団
    ax.text(1.6, 4.35, "① 集団（個体 = 解）", ha="center", fontsize=18)
    for i, (s, c) in enumerate(zip(pop, cuts)):
        sel = i in (0, 2)
        y = 3.5 - i * 0.7
        bits(ax, 0.0, y, s, edge=ORANGE if sel else BLACK)
        ax.text(3.35, y + 0.2, f"{c}", va="center", fontsize=15, color=ORANGE if sel else BLACK)
    ax.text(3.35, 3.95, "カット", fontsize=12, ha="left")
    # 親を選ぶ → 交叉
    arrow(ax, (3.95, 2.3), (4.65, 2.3))
    ax.text(6.25, 4.35, "② 交叉", ha="center", fontsize=18)
    p1, p2 = pop[0], pop[2]
    bits(ax, 4.65, 3.5, p1)
    bits(ax, 4.65, 2.8, p2)
    ax.text(8.0, 3.7, "親1", ha="left", va="center", fontsize=14)
    ax.text(8.0, 3.0, "親2", ha="left", va="center", fontsize=14)
    child = "".join(a if a == b else "?" for a, b in zip(p1, p2))
    arrow(ax, (6.25, 2.7), (6.25, 2.15))
    bits(ax, 4.65, 1.65, child)
    ax.text(6.25, 1.2, "一致する頂点を\n受け継ぐ", ha="center", va="top", fontsize=14)
    # 突然変異・局所探索
    arrow(ax, (8.0, 1.86), (8.6, 1.86))
    filled = "".join(c if c != "?" else "01"[k % 2] for k, c in enumerate(child))
    flip = 6
    mutated = filled[:flip] + ("0" if filled[flip] == "1" else "1") + filled[flip + 1:]
    bits(ax, 8.6, 1.65, mutated, mark=[flip])
    ax.text(10.2, 1.2, "③ 突然変異・改良", ha="center", va="top", fontsize=18)
    ax.text(10.2, 0.72, "一部を反転し局所探索で磨く", ha="center", va="top", fontsize=14)
    # 入れ替え
    ax.text(8.85, 3.3, "④ 集団の悪い個体と\n　 入れ替える", ha="left", va="center", fontsize=16)
    arrow(ax, (10.2, 2.2), (10.2, 2.85))
    ax.plot([10.2, 10.2, 1.6], [3.8, 5.0, 5.0], color=GREEN, lw=2.5)
    arrow(ax, (1.6, 5.0), (1.6, 4.72), color=GREEN)
    ax.text(6.25, 5.08, "世代をくり返す", color=GREEN, fontsize=16, ha="center", va="bottom")
    ax.add_patch(Rectangle((0.0, -0.05), 0.32, 0.3, facecolor=BLUE))
    ax.text(0.4, 0.1, "集合 A", va="center", fontsize=14)
    ax.add_patch(Rectangle((1.6, -0.05), 0.32, 0.3, facecolor=RED))
    ax.text(2.0, 0.1, "集合 B", va="center", fontsize=14)
    ax.add_patch(Rectangle((3.2, -0.05), 0.32, 0.3, facecolor="#dddddd"))
    ax.text(3.6, 0.1, "親で異なる（どちらかを選ぶ）", va="center", fontsize=14)
    ax.set_xlim(-0.1, 12.5)
    ax.set_ylim(-0.2, 5.5)
    ax.axis("off")
    save(fig, out / "ga.png")


# ---------------------------------------------------------------- 目的（同じ総時間）
def fig_goal(out: Path) -> None:
    fig = plt.figure(figsize=(6.4, 5.0))
    ax = fig.add_axes([0.05, 0.68, 0.9, 0.3])
    total = 10.0
    ax.add_patch(Rectangle((0, 1.0), total, 0.6, facecolor=BLUE, alpha=0.85))
    ax.text(total / 2, 1.3, "SA", color="white", ha="center", va="center", fontsize=18)
    ax.add_patch(Rectangle((0, 0.0), 2.5, 0.6, facecolor=RED, alpha=0.85))
    ax.text(1.25, 0.3, "CIM", color="white", ha="center", va="center", fontsize=18)
    ax.add_patch(Rectangle((2.5, 0.0), total - 2.5, 0.6, facecolor=BLUE, alpha=0.55))
    ax.text(6.25, 0.3, "SA", color="white", ha="center", va="center", fontsize=18)
    ax.text(-0.2, 1.3, "ランダム開始", ha="right", va="center", fontsize=17)
    ax.text(-0.2, 0.3, "CIM 開始", ha="right", va="center", fontsize=17)
    ax.annotate("", xy=(total, -0.35), xytext=(0, -0.35), arrowprops=dict(arrowstyle="<->", lw=1.8))
    ax.text(total / 2, -0.75, "同じ総時間", ha="center", va="center", fontsize=17)
    ax.set_xlim(-3.6, total + 0.2)
    ax.set_ylim(-1.0, 1.75)
    ax.axis("off")
    ax2 = fig.add_axes([0.2, 0.08, 0.76, 0.48])
    t = np.linspace(0, total, 300)
    g_rand = 0.15 + 2.6 * np.exp(-t / 1.6)
    tc = 2.5
    tt = t[t >= tc]
    g_cim = 0.1 + 0.35 * np.exp(-(tt - tc) / 2.0)
    ax2.plot(t, g_rand, color=BLUE, lw=3, label="ランダム開始")
    ax2.plot(tt, g_cim, color=RED, lw=3, label="CIM 開始")
    ax2.plot([tc, tc], [g_cim[0], 3.0], color=RED, lw=1.5, ls=":")
    ax2.axhline(0.3, color=BLACK, ls="--", lw=1.8)
    ax2.text(total, 0.36, "目標の品質", ha="right", va="bottom", fontsize=16)
    ax2.set_xlabel("経過時間")
    ax2.set_ylabel("BKS との差")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_ylim(0, 3.0)
    ax2.legend(loc="upper right", frameon=False, fontsize=16)
    ax2.text(3.0, 1.5, "どちらが先に\n目標に届くか？", fontsize=17, color=BLACK)
    save(fig, out / "goal.png")


# ---------------------------------------------------------------- 考察（谷）
def fig_valley(out: Path) -> None:
    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.add_patch(Circle((0, 0), 0.3, color="#e6e6e6", zorder=0))
    ax.text(0.0, -0.245, "谷（d < 0.3）", ha="center", fontsize=15)
    for r, col in ((D_CIM_START, RED), (D_RANDOM_SA, BLUE), (D_RANDOM_START, BLUE)):
        ax.add_patch(Circle((0, 0), r, fill=False, ls=":", lw=1.3, color=col))
    ax.plot(0, 0, "*", ms=26, color="#ffcc00", mec=BLACK, zorder=5)
    ax.text(0.03, 0.035, "最良解", fontsize=16, zorder=6)

    def ring(r, n, a0, a1):
        a = rng.uniform(a0, a1, n)
        rr = r * (1 + rng.normal(0, 0.06, n))
        return np.c_[rr * np.cos(a), rr * np.sin(a)]

    rs, rf = ring(D_RANDOM_START, 7, 0.3, 2.9), None
    rf = rs / np.linalg.norm(rs, axis=1, keepdims=True) * D_RANDOM_SA * (1 + rng.normal(0, 0.05, (len(rs), 1)))
    cs = ring(D_CIM_START, 7, 3.6, 5.9)
    cf = cs / np.linalg.norm(cs, axis=1, keepdims=True) * D_CIM_SA * (1 + rng.normal(0, 0.04, (len(cs), 1)))
    for a, b in zip(rs, rf):
        arrow(ax, a, b, color=BLUE, lw=1.6, ms=12)
    for a, b in zip(cs, cf):
        arrow(ax, a, b, color=RED, lw=1.6, ms=12)
    ax.scatter(rs[:, 0], rs[:, 1], s=90, facecolor="white", edgecolor=BLUE, lw=2.2, zorder=4)
    ax.scatter(rf[:, 0], rf[:, 1], s=90, color=BLUE, zorder=4)
    ax.scatter(cs[:, 0], cs[:, 1], s=90, facecolor="white", edgecolor=RED, lw=2.2, zorder=4)
    ax.scatter(cf[:, 0], cf[:, 1], s=90, color=RED, zorder=4)
    ax.text(0.0, 0.54, f"ランダム解　d = {D_RANDOM_START}", color=BLUE, ha="center", fontsize=16)
    ax.text(-0.66, -0.36, f"SA 後\nd = {D_RANDOM_SA}", color=BLUE, ha="left", va="center", fontsize=15)
    ax.text(-0.05, -0.52, f"CIM の解　d = {D_CIM_START}\n→ SA 後　d = {D_CIM_SA}", color=RED, ha="center",
            va="top", fontsize=16)
    ax.scatter([0.4], [-0.25], s=70, facecolor="white", edgecolor=BLACK, lw=2)
    ax.text(0.44, -0.25, "開始", va="center", fontsize=14)
    ax.scatter([0.4], [-0.33], s=70, color=BLACK)
    ax.text(0.44, -0.33, "SA 後", va="center", fontsize=14)
    ax.set_xlim(-0.66, 0.62)
    ax.set_ylim(-0.76, 0.62)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out / "valley.png")


# ---------------------------------------------------------------- CIM の仕組み
def box(ax, xy, w, h, text, fc="#f2f2f2", ec=BLACK, fs=17, color=BLACK) -> None:
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=2, zorder=4))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs, color=color, zorder=5)


def fig_cim_loop(out: Path) -> None:
    rng = np.random.default_rng(4)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(1.6 * np.cos(th), 1.6 * np.sin(th), color="#7f7f7f", lw=7, zorder=1, alpha=0.6)
    n = 16
    for k in range(n):
        a = 2 * np.pi * k / n + 0.1
        if abs(np.sin(a)) > 0.9 and np.sin(a) > 0:
            continue
        amp = rng.uniform(0.18, 0.4) * (1 if rng.random() < 0.5 else -1)
        x, y = 1.6 * np.cos(a), 1.6 * np.sin(a)
        ux, uy = np.cos(a), np.sin(a)
        col = BLUE if amp > 0 else RED
        arrow(ax, (x, y), (x + amp * ux, y + amp * uy), color=col, lw=3, ms=14, zorder=3)
    ax.text(0, 0.1, "光パルス N 個\n（= スピン）", ha="center", va="center", fontsize=17)
    ax.text(0, -0.8, "（くり返し周回）", ha="center", fontsize=16, color="#555555")
    box(ax, (0, 1.6), 2.9, 0.75, "増幅器（ポンプ光）\n利得 αp(n)・飽和 β", fc="#fde9e9", ec=RED, fs=15)
    box(ax, (1.6, -0.2), 1.45, 0.7, "取り出して\n測る", fs=15)
    box(ax, (-1.6, -0.2), 1.45, 0.7, "注入\n（結合 bJ）", fs=15, fc="#e8f0fb", ec=BLUE)
    box(ax, (0, -2.45), 4.4, 0.62, "計算機：各パルスへの結合 Σ J x を計算", fs=15, fc="#e8f0fb", ec=BLUE)
    ax.plot([2.35, 2.7, 2.7, 2.25], [-0.2, -0.2, -2.45, -2.45], color=BLUE, lw=2.5, zorder=2)
    ax.plot([-2.25, -2.7, -2.7], [-2.45, -2.45, -0.2], color=BLUE, lw=2.5, zorder=2)
    arrow(ax, (-2.7, -0.2), (-2.4, -0.2), color=BLUE)
    ax.text(2.0, 1.95, "ループ損失 a・雑音 c", fontsize=15, color="#555555", ha="left")
    ax.text(-3.2, 2.45, "青（外向き）：振幅 +（位相 0）　赤（内向き）：振幅 −（位相 π）", fontsize=14, ha="left")
    ax.set_xlim(-3.3, 3.9)
    ax.set_ylim(-2.85, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, out / "cim_loop.png")


def main() -> None:
    out = out_dir()
    fig_cim_loop(out)
    fig_ising(out)
    fig_sa(out)
    fig_ga(out)
    fig_goal(out)
    fig_valley(out)


if __name__ == "__main__":
    main()
