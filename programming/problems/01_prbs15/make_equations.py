"""解説 md 用の数式画像を生成する。

PDF 変換ツールが LaTeX 数式 ($$...$$) を描画できない環境でも数式が見えるよう、
matplotlib の mathtext で数式を PNG 画像にして埋め込めるようにする。
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = os.path.dirname(os.path.abspath(__file__))

# (出力ファイル名, mathtext 文字列, フォントサイズ)
EQUATIONS = [
    ("eq_polynomial.png", r"$g(x) = x^{15} + x^{14} + 1$", 26),
    (
        "eq_sinc.png",
        r"$|H(f)|^2 \propto \mathrm{sinc}^2(fT) = "
        r"\left(\dfrac{\sin(\pi f T)}{\pi f T}\right)^2$",
        26,
    ),
]


def render(filename: str, mathtext: str, fontsize: int) -> None:
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0, 0, mathtext, fontsize=fontsize)
    out = os.path.join(HERE, filename)
    # bbox_inches="tight" で数式部分だけを切り出す
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1, transparent=False)
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    for name, text, size in EQUATIONS:
        render(name, text, size)
