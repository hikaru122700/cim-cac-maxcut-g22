"""Markdown → PDF 変換ユーティリティ(数式 MathJax / 日本語対応)。

LaTeX を入れずに、Python-Markdown で HTML 化 → MathJax で数式描画 →
Chrome のヘッドレス印刷で PDF 化する。Windows 専用。

使い方(プロジェクトルートから):
    python scripts/utils/md_to_pdf.py <input.md> [output.pdf]
output 省略時は入力と同じ場所に拡張子 .pdf で出力する。
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
<style>
  body {{ font-family: "Yu Gothic", "Meiryo", sans-serif; font-size: 11pt;
         line-height: 1.7; color: #1a1a1a; max-width: 920px; margin: 0 auto;
         padding: 24px; }}
  h1 {{ font-size: 20pt; border-bottom: 3px solid #2c5f8a; padding-bottom: 6px; }}
  h2 {{ font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 4px;
        margin-top: 28px; color: #2c5f8a; }}
  h3 {{ font-size: 12.5pt; margin-top: 20px; }}
  table {{ border-collapse: collapse; margin: 12px 0; font-size: 10pt; }}
  th, td {{ border: 1px solid #bbb; padding: 5px 10px; text-align: left; }}
  th {{ background: #eef3f8; }}
  img {{ max-width: 100%; height: auto; display: block; margin: 10px 0; }}
  code {{ background: #f3f3f3; padding: 1px 5px; border-radius: 3px;
          font-family: "Consolas", monospace; font-size: 9.5pt; }}
  blockquote {{ border-left: 4px solid #2c5f8a; margin: 12px 0; padding: 4px 16px;
                background: #f7fafc; color: #333; }}
  hr {{ border: none; border-top: 1px solid #ddd; margin: 20px 0; }}
  mjx-container {{ overflow-x: auto; overflow-y: hidden; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    raise FileNotFoundError("Chrome / Edge が見つかりません: " + str(CHROME_CANDIDATES))


def md_to_html(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    # 画像の相対パスを絶対 file URI に変換(PDF から参照できるように)
    base = md_path.parent.resolve()

    def fix_img(m: re.Match) -> str:
        alt, src = m.group(1), m.group(2).strip()
        if src.startswith(("http://", "https://", "file:", "data:")):
            return m.group(0)
        abs_path = (base / src).resolve()
        uri = abs_path.as_uri()
        return f"![{alt}]({uri})"

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", fix_img, text)

    # --- 数式を Markdown 処理から保護する ---
    # Markdown は数式内の `_`(P_r 等)を <em> に誤変換するため、
    # $$...$$ / $...$ をプレースホルダに退避し、変換後に復元する。
    math_store: list[str] = []

    def stash(m: re.Match) -> str:
        math_store.append(m.group(0))
        return f"\x00MATH{len(math_store) - 1}\x00"

    # display ($$...$$) を先に、次に inline ($...$) を退避
    text = re.sub(r"\$\$.*?\$\$", stash, text, flags=re.S)
    text = re.sub(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)", stash, text)

    body = markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
    )

    # 数式を復元(プレースホルダが <p> で囲まれていても元の式に戻す)
    def restore(m: re.Match) -> str:
        return math_store[int(m.group(1))]

    body = re.sub(r"\x00MATH(\d+)\x00", restore, body)
    return HTML_TEMPLATE.format(body=body)


def html_to_pdf(html: str, pdf_path: Path) -> None:
    chrome = find_chrome()
    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "doc.html"
        html_file.write_text(html, encoding="utf-8")
        cmd = [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=20000",  # MathJax の描画完了を待つ
            f"--print-to-pdf={pdf_path}",
            html_file.as_uri(),
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        # Chrome が書き出すまで僅かに待つ
        for _ in range(20):
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                break
            time.sleep(0.2)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    md_path = Path(sys.argv[1]).resolve()
    if not md_path.exists():
        print(f"入力が見つかりません: {md_path}")
        sys.exit(1)
    pdf_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else md_path.with_suffix(".pdf")

    html = md_to_html(md_path)
    html_to_pdf(html, pdf_path)
    size = pdf_path.stat().st_size if pdf_path.exists() else 0
    print(f"saved: {pdf_path}  ({size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
