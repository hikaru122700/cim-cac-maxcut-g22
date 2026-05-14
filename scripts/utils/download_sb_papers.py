"""Cloudflare バイパス込みで Ising マシン関連の主要論文をダウンロードする。

- SB 関連 3 本(初期取得分)
- CIM 関連最新 (CAC, 100k-spin CIM, 各種実機・アルゴリズム拡張)
- ベンチマーク・サーベイ

すでに同名 PDF (50KB 超)がある場合はスキップ。出力先は papers/ 直下。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cloudscraper

OUT_DIR = Path(__file__).resolve().parents[2] / "papers"
OUT_DIR.mkdir(exist_ok=True)

TARGETS = [
    # ===== SB 系(既存)=====
    ("Goto2019_aSB_SciAdv.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.aav2372"),
    ("Goto2021_bSB_dSB_SciAdv.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.abe7953"),
    ("KanaoGoto2022_thermalSB_arXiv.pdf",
     "https://arxiv.org/pdf/2203.08361"),

    # ===== CIM 系アルゴリズム =====
    # CAC (Leleu et al. 2021 Comm. Phys.) — SB と並ぶ CIM 系の主役
    ("Leleu2021_CAC_arXiv.pdf",
     "https://arxiv.org/pdf/2009.04084"),
    # CAC のハイパラ最適化 (2025)
    ("Sao2025_CIM_PortfolioTuning_arXiv.pdf",
     "https://arxiv.org/pdf/2507.20295"),
    # L0 compressed sensing with CAC-CIM (Sci. Rep. 2023)
    ("L0CompressedSensing_CAC_CIM_SciRep2023.pdf",
     "https://www.nature.com/articles/s41598-023-43364-8.pdf"),

    # ===== CIM 物理機 =====
    # 100,000-spin CIM (Honjo et al. 2021 Sci. Adv.)
    ("Honjo2021_100k_spin_CIM_SciAdv.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.abh0952"),
    # CIM for independent sets (2025 Sci. Adv.)
    ("CIM_IndependentSets_SciAdv2025.pdf",
     "https://www.science.org/doi/pdf/10.1126/sciadv.ads7223"),
    # Versatile multi-wavelength CIM (Light: Sci. Appl. 2026)
    ("MultiWavelength_CIM_Light2026.pdf",
     "https://www.nature.com/articles/s41377-026-02225-5.pdf"),
    # Polarization symmetry breaking Kerr CIM (Nat. Comm. 2026)
    ("PolarizationKerr_CIM_NatComm2026.pdf",
     "https://www.nature.com/articles/s41467-026-68794-6.pdf"),

    # ===== サーベイ・ベンチマーク =====
    # CIM "The Good, The Bad, The Ugly" (2025) — レビュー的展望
    ("CIM_GoodBadUgly_arXiv2025.pdf",
     "https://arxiv.org/pdf/2507.14489"),
    # 包括的 Max-Cut ベンチマーク (DA vs SBM vs SA vs HS) 2025
    ("DAMB_ComprehensiveBenchmark_arXiv2025.pdf",
     "https://arxiv.org/pdf/2507.22117"),

    # ===== 別パラダイム(参考)=====
    # 200 GOPS Hopfield-inspired photonic Ising machine (Nature 2025)
    ("Hopfield_Photonic_200GOPS_Nature2025.pdf",
     "https://www.nature.com/articles/s41586-025-09838-7.pdf"),
]


def main():
    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "windows", "mobile": False}
    )

    ok, skip, fail = 0, 0, 0
    for name, url in TARGETS:
        out = OUT_DIR / name
        if out.exists() and out.stat().st_size > 50_000:
            print(f"[SKIP] {name} (already {out.stat().st_size//1024} KB)")
            skip += 1
            continue
        print(f"Fetching {name} ...")
        try:
            r = scraper.get(url, timeout=90, allow_redirects=True)
            size = len(r.content)
            ctype = r.headers.get("Content-Type", "?")
            is_pdf = "pdf" in ctype.lower() or r.content[:4] == b"%PDF"
            if r.status_code == 200 and size > 50_000 and is_pdf:
                out.write_bytes(r.content)
                print(f"  OK    status={r.status_code} size={size//1024} KB")
                ok += 1
            else:
                head = r.content[:120].decode("utf-8", errors="replace")
                print(f"  FAIL  status={r.status_code} size={size} ctype={ctype}")
                print(f"        preview: {head!r}")
                fail += 1
        except Exception as exc:
            print(f"  EXCEPTION: {exc}")
            fail += 1

    print(f"\nSummary: ok={ok}  skip={skip}  fail={fail}  total={len(TARGETS)}")


if __name__ == "__main__":
    main()
