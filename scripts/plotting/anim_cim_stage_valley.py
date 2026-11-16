"""実験 D(cim_stage_valley)の D1 を動画にする:CIM 1024 本の途中の解が、ラウンドとともにどこへ動くか。

  左:各試行の点(横 = E = ov1 − ov2、右ほど u1 の側。縦 = 最も近い BKS 解までの距離 d、下ほど近い)。
      d < 0.3(最良解の谷の中)の点は赤、それ以外は灰色。
  右:1024 本の d と E の中央値の推移(線形段の範囲を灰色で塗る)。

出力(既存のファイルは上書きしない):
  d1_anim.mp4         1280×720、15 fps
  d1_anim_poster.png  スライドの代わりの静止画(線形段の終わりのコマ)

usage:
  python scripts/plotting/anim_cim_stage_valley.py <cim_stage_valley の出力ディレクトリ> <ffmpeg.exe のパス>
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

plt.rcParams.update({
    "font.family": "Yu Gothic", "axes.unicode_minus": False, "axes.labelsize": 20, "axes.titlesize": 20,
    "legend.fontsize": 20, "xtick.labelsize": 16, "ytick.labelsize": 16, "font.size": 16,
})
RED, BLUE, GRAY, ORANGE = "#d62728", "#1f77b4", "#9a9a9a", "#ff7f0e"
VALLEY_EDGE = 0.3
FIRST_ROUND = 40
FPS, HOLD_SEC = 15, 4


def load(out: Path) -> dict:
    z = np.load(out / "raw.npz")
    r = z["d1_rounds"]
    keep = r >= FIRST_ROUND
    d, e = z["d1_d_bks"][:, keep], (z["d1_ov1"] - z["d1_ov2"])[:, keep]
    return {"r": r[keep], "d": d, "e": e, "k_lin": float(np.median(z["d1_k_lin"])),
            "d_med": np.median(d, axis=0), "e_med": np.median(e, axis=0)}


def stage_text(rnd: int, k_lin: float) -> str:
    if rnd <= k_lin:
        return "線形段:振幅が小さく、行列 J をかけるだけ → J の最大固有ベクトル $u_1$ の向きが育つ"
    return "非線形段:振幅が大きくなり利得が飽和 → 各スピンが ±1 に決まる"


def setup(data: dict):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.07, 0.19, 0.47, 0.64])
    bx = fig.add_axes([0.63, 0.19, 0.34, 0.64])
    ax.axhline(VALLEY_EDGE, color="black", ls="--", lw=1.5)
    ax.text(0.97, VALLEY_EDGE + 0.008, "谷の縁 d = 0.3", ha="right", va="bottom", fontsize=16)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_xlim(-0.6, 1.0)
    ax.set_ylim(0.0, 0.55)
    ax.set_xlabel("E = ov$_1$ − ov$_2$(右ほど $u_1$ の側)")
    ax.set_ylabel("最良解までの距離 d(下ほど近い)")
    ax.tick_params(direction="in", which="both", top=True, right=True)
    bx.axvspan(FIRST_ROUND, data["k_lin"], color="#e8e8e8", zorder=0)
    bx.text(np.sqrt(FIRST_ROUND * data["k_lin"]), 0.53, "線形段", ha="center", fontsize=18)
    bx.text(800, 0.53, "非線形段", ha="center", fontsize=18)
    bx.set_xscale("log")
    bx.set_xlim(FIRST_ROUND, 2400)
    bx.set_ylim(-0.05, 0.57)
    bx.axhline(0, color="black", lw=0.8)
    bx.set_xlabel("CIM のラウンド")
    bx.set_title("1024 本の中央値", fontsize=20)
    bx.tick_params(direction="in", which="both", top=True, right=True)
    art = {
        "scat": ax.scatter([], [], s=9, alpha=0.7),
        "title": fig.text(0.5, 0.92, "", ha="center", fontsize=21),
        "round": ax.text(-0.57, 0.52, "", fontsize=20, va="top", bbox={"fc": "white", "ec": "none", "alpha": 0.85}, zorder=5),
        "frac": ax.text(-0.57, 0.46, "", fontsize=20, va="top", color=RED, bbox={"fc": "white", "ec": "none", "alpha": 0.85}, zorder=5),
        "dline": bx.plot([], [], color=RED, lw=3, label="距離 d")[0],
        "eline": bx.plot([], [], color=BLUE, lw=3, label="E")[0],
        "now": bx.axvline(FIRST_ROUND, color=ORANGE, lw=2),
        "note": fig.text(0.5, 0.025, "", ha="center", fontsize=20, color=RED),
    }
    bx.legend(loc="center right", frameon=False)
    return fig, art


def draw(i: int, data: dict, art: dict) -> None:
    j = min(i, len(data["r"]) - 1)
    rnd = int(data["r"][j])
    d, e = data["d"][:, j], data["e"][:, j]
    inside = d < VALLEY_EDGE
    art["scat"].set_offsets(np.column_stack([e, d]))
    art["scat"].set_color(np.where(inside, RED, GRAY))
    art["title"].set_text(stage_text(rnd, data["k_lin"]))
    art["round"].set_text(f"ラウンド {rnd}")
    art["frac"].set_text(f"谷の中 {inside.mean():.0%}")
    art["dline"].set_data(data["r"][: j + 1], data["d_med"][: j + 1])
    art["eline"].set_data(data["r"][: j + 1], data["e_med"][: j + 1])
    art["now"].set_xdata([rnd, rnd])
    art["note"].set_text("1 本ずつでは谷の中は 32%。256 本からカット値の上位 16 本を選ぶと 99.7%"
                         if i >= len(data["r"]) else "")


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    out, ffmpeg = Path(sys.argv[1]), Path(sys.argv[2])
    if not (out / "raw.npz").exists() or not ffmpeg.exists():
        sys.exit("raw.npz or ffmpeg not found")
    mp4, poster = out / "d1_anim.mp4", out / "d1_anim_poster.png"
    for f in (mp4, poster):
        if f.exists():
            raise FileExistsError(f"{f} already exists")
    plt.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    data = load(out)
    fig, art = setup(data)
    n_frames = len(data["r"]) + FPS * HOLD_SEC
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, str(mp4), dpi=100):
        for i in range(n_frames):
            draw(i, data, art)
            writer.grab_frame()
    k = int(np.searchsorted(data["r"], data["k_lin"]))
    draw(k, data, art)
    fig.savefig(poster, dpi=100)
    plt.close(fig)
    print("saved", mp4, poster)


if __name__ == "__main__":
    main()
