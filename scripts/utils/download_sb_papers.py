"""Cloudflare バイパス込みで SB 3 論文を取得。"""
import sys
from pathlib import Path

import cloudscraper

OUT_DIR = Path(__file__).parent / "papers"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = [
    ("Goto2019_aSB_SciAdv.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.aav2372"),
    ("Goto2021_bSB_dSB_SciAdv.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.abe7953"),
    ("KanaoGoto2022_thermalSB_arXiv.pdf",
     "https://arxiv.org/pdf/2203.08361"),
]

scraper = cloudscraper.create_scraper(
    browser={"browser": "chrome", "platform": "windows", "mobile": False}
)

for name, url in TARGETS:
    out = OUT_DIR / name
    print(f"Fetching {name}...")
    try:
        r = scraper.get(url, timeout=60, allow_redirects=True)
        print(f"  status={r.status_code}, size={len(r.content)} bytes, "
              f"content-type={r.headers.get('Content-Type', '?')}")
        if r.status_code == 200 and len(r.content) > 50_000:
            out.write_bytes(r.content)
            print(f"  -> Saved: {out}")
        else:
            head = r.content[:200].decode("utf-8", errors="replace")
            print(f"  -> Failed. First 200 bytes: {head!r}")
    except Exception as exc:
        print(f"  -> Exception: {exc}")
    print()
