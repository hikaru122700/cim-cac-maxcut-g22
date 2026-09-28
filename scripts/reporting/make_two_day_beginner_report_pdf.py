"""Create a beginner-friendly Japanese report of the two-day CIM warm-start study."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reporting.make_sa_ga_matched_report_pdf import (
    BODY,
    BODY_SMALL,
    FONT,
    H1,
    PALE_BLUE,
    PALE_GRAY,
    PALE_GREEN,
    REPORT_MID,
    SUBTITLE,
    TITLE,
    callout,
    chart,
    configure_matplotlib,
    data_table,
    p,
    three_boxes,
)

OUT = ROOT / "output" / "pdf" / "cim_warmstart_two_day_beginner_guide_ja.pdf"
FIG = ROOT / "results" / "2026-07-26" / "cim_warmstart" / "beginner_report_figures"

G22_MAIN = ROOT / "results" / "2026-07-24" / "cim_warmstart" / "v3_G22_nt16_main" / "results.json"
G55_MAIN = ROOT / "results" / "2026-07-24" / "cim_warmstart" / "v4_G55_nt16_replication" / "results.json"
SA_MATCHED = ROOT / "results" / "2026-07-26" / "cim_warmstart" / "sa_matched_v2_G22_G55_nt16_to300m" / "results.json"
GA_MATCHED = ROOT / "results" / "2026-07-26" / "cim_warmstart" / "ga_matched_v1_G22_G55_nt16" / "results.json"
GA_FULL = ROOT / "results" / "2026-07-26" / "cim_warmstart" / "ga_full_cim_population_v1_G22_G55_nt16" / "results.json"

NAVY = "#17324d"
COLD = "#687780"
WARM = "#d24a43"
FULL = "#2c6f9e"
PT = "#6d54a3"
GREEN = "#16836c"
GRID = "#d7e0e6"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_all() -> dict:
    return {
        "G22_main": read(G22_MAIN),
        "G55_main": read(G55_MAIN),
        "sa": read(SA_MATCHED),
        "ga": read(GA_MATCHED),
        "full": read(GA_FULL),
    }


def save_maxcut_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.8), dpi=180)
    pos = {
        0: (0.8, 2.7), 2: (2.0, 1.0), 5: (3.1, 2.7), 7: (3.8, 1.0),
        1: (6.2, 2.7), 3: (7.0, 1.0), 4: (8.2, 2.7), 6: (9.2, 1.0),
    }
    left = {0, 2, 5, 7}
    edges = [(0,1),(0,2),(1,2),(1,3),(2,3),(2,4),(3,5),(4,5),(4,6),(5,6),(5,7),(6,7)]
    for a, b in edges:
        cross = (a in left) != (b in left)
        ax.plot(
            [pos[a][0], pos[b][0]],
            [pos[a][1], pos[b][1]],
            color=GREEN if cross else "#b8c2c9",
            lw=3.2 if cross else 1.5,
            zorder=1,
        )
    for node, (x, y) in pos.items():
        side_a = node in left
        ax.scatter(
            x, y, s=650, color=FULL if side_a else WARM,
            edgecolor="white", linewidth=2, zorder=3,
        )
        ax.text(x, y, str(node + 1), ha="center", va="center", color="white",
                fontsize=12, fontweight="bold", zorder=4)
    ax.text(2.2, 3.35, "グループA", color=FULL, ha="center", fontsize=14, fontweight="bold")
    ax.text(7.8, 3.35, "グループB", color=WARM, ha="center", fontsize=14, fontweight="bold")
    ax.text(5.0, 0.05, "緑の線 = 2グループをまたぐ線。これをできるだけ多くする", ha="center",
            fontsize=13, color=NAVY)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.2, 3.7)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def short_gain_figure(data: dict, path: Path) -> None:
    g22_pt = data["G22_main"]["solvers"]["PT"]["ladder"][0]
    g55_pt = data["G55_main"]["solvers"]["PT"]["ladder"][0]
    values = {
        "G22": [
            data["ga"]["datasets"]["G22"]["points"][2]["paired"]["delta_mean"],
            data["sa"]["datasets"]["G22"]["points"][0]["paired"]["delta_mean"],
            g22_pt["paired"]["warm_minus_cold_mean"],
        ],
        "G55": [
            data["ga"]["datasets"]["G55"]["points"][2]["paired"]["delta_mean"],
            data["sa"]["datasets"]["G55"]["points"][0]["paired"]["delta_mean"],
            g55_pt["paired"]["warm_minus_cold_mean"],
        ],
    }
    times = {
        "G22": [
            data["ga"]["datasets"]["G22"]["points"][2]["warm"]["total_time"],
            data["sa"]["datasets"]["G22"]["points"][0]["warm"]["total_time"],
            g22_pt["warm"]["total_time"],
        ],
        "G55": [
            data["ga"]["datasets"]["G55"]["points"][2]["warm"]["total_time"],
            data["sa"]["datasets"]["G55"]["points"][0]["warm"]["total_time"],
            g55_pt["warm"]["total_time"],
        ],
    }
    labels = ["GA\n3世代", "SA\n10万反復", "PT-ICM\n30 sweep"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=190)
    for ax, ds in zip(axes, ("G22", "G55")):
        x = np.arange(3)
        bars = ax.bar(x, values[ds], color=[FULL, WARM, PT], width=0.64)
        for bar, value, seconds in zip(bars, values[ds], times[ds]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values[ds]) * 0.035,
                f"+{value:.1f}\n総時間 {seconds:.2f}秒",
                ha="center", va="bottom", fontsize=9,
            )
        ax.axhline(0, color=COLD, lw=1)
        ax.set_xticks(x, labels)
        ax.set_ylabel("CIMを使ったときの平均cut改善量")
        ax.set_title(ds)
        ax.grid(True, axis="y", color=GRID, ls=":")
        ax.set_ylim(0, max(values[ds]) * 1.32)
    fig.suptitle("短時間側では、CIMから始めると平均的に良い解へ進みやすい",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def pt_figure(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=190)
    for ax, ds, source in zip(
        axes, ("G22", "G55"), (data["G22_main"], data["G55_main"])
    ):
        pt = source["solvers"]["PT"]
        for mode, label, color, marker in (
            ("single", "CIM解を1か所に入れる", WARM, "o"),
            ("ladder", "CIM解を少しずつ崩して広げる", PT, "s"),
        ):
            x = [row["budget"] for row in pt[mode]]
            y = [row["paired"]["warm_minus_cold_mean"] for row in pt[mode]]
            ax.plot(x, y, marker=marker, ms=7, lw=2, color=color, label=label)
            for xv, yv in zip(x, y):
                offset = 4 if mode == "ladder" else -9
                ax.text(xv, yv + offset, f"+{yv:.1f}", ha="center", fontsize=8)
        ax.axhline(0, color=COLD, lw=1)
        ax.set_xlabel("PT-ICMのsweep数")
        ax.set_ylabel("平均cut改善量")
        ax.set_title(ds)
        ax.set_xticks([30, 80, 200])
        all_gains = [
            row["paired"]["warm_minus_cold_mean"]
            for mode in ("single", "ladder")
            for row in pt[mode]
        ]
        ax.set_ylim(-5, max(all_gains) * 1.18)
        ax.grid(True, axis="y", color=GRID, ls=":")
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("PT-ICM：CIM初期解の効果が最も一貫していた",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ga_figure(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3), dpi=190)
    for ax, ds in zip(axes, ("G22", "G55")):
        pts = data["ga"]["datasets"][ds]["points"]
        x = [row["budget"] for row in pts]
        y = [row["paired"]["delta_mean"] for row in pts]
        ax.plot(x, y, marker="D", ms=7, lw=2, color=FULL)
        ax.fill_between(x, 0, y, color=FULL, alpha=0.08)
        ax.axhline(0, color=COLD, lw=1)
        ax.set_xlabel("GA世代数")
        ax.set_ylabel("平均cut改善量")
        ax.set_title(ds)
        ax.set_xticks(x)
        ax.grid(True, axis="y", color=GRID, ls=":")
        for xv, yv in zip(x, y):
            ax.text(xv, yv + (5 if ds == "G55" else 0.8), f"{yv:+.1f}", ha="center", fontsize=8)
    fig.suptitle("GA：G22では差が小さく、G55では長く残った",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def sa_figure(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=190)
    for ax, ds in zip(axes, ("G22", "G55")):
        pts = data["sa"]["datasets"][ds]["points"]
        x = np.array([row["warm"]["total_time"] for row in pts])
        y = np.array([row["paired"]["delta_mean"] for row in pts])
        labels = ["10万", "30万", "100万", "300万", "1000万", "3000万", "1億", "3億"]
        ax.plot(x, y, marker="o", ms=6, lw=2, color=WARM)
        ax.axhspan(0, max(y.max(), 1) * 1.15, color=GREEN, alpha=0.08)
        ax.axhspan(min(y.min(), -1) * 1.25, 0, color=WARM, alpha=0.06)
        ax.axhline(0, color=COLD, lw=1.2)
        for xv, yv, label in zip(x, y, labels):
            ax.annotate(label, (xv, yv), xytext=(3, 5), textcoords="offset points", fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("CIM生成を含む総時間（秒、対数目盛）")
        ax.set_ylabel("平均cut改善量")
        ax.set_title(ds)
        ax.grid(True, color=GRID, ls=":")
    fig.suptitle("SA：短時間では有効だが、長く回すと差は消える",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def ga_full_figure(data: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.5), dpi=190)
    for ax, ds in zip(axes, ("G22", "G55")):
        points = data["full"]["datasets"][ds]["points"]
        selected = [points[0], points[3], points[-1]]
        x = np.arange(3)
        width = 0.24
        for offset, key, label, color in (
            (-width, "cold", "全てrandom", COLD),
            (0, "single", "CIM×1個体", WARM),
            (width, "full", "CIM×全個体", FULL),
        ):
            vals = [row[key]["gap_best"] for row in selected]
            bars = ax.bar(x + offset, vals, width=width, label=label, color=color)
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, max(value, 0) + 1.2,
                        f"{value:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, ["0世代", "8世代", "50世代"])
        ax.set_ylabel("最良gap（小さいほど良い）")
        ax.set_title(ds)
        ax.grid(True, axis="y", color=GRID, ls=":")
        ax.legend(fontsize=8)
        if ds == "G55":
            ax.set_yscale("symlog", linthresh=10)
    fig.suptitle("GAの初期集団を全てCIM解にすると、最初の品質がさらに上がる",
                 fontsize=16, fontweight="bold", color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(data: dict) -> dict[str, Path]:
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {
        "maxcut": FIG / "01_maxcut_for_beginner.png",
        "overall": FIG / "02_short_gain_overview.png",
        "pt": FIG / "03_pt_icm_story.png",
        "ga": FIG / "04_ga_story.png",
        "sa": FIG / "05_sa_story.png",
        "full": FIG / "06_ga_full_population_story.png",
    }
    save_maxcut_figure(paths["maxcut"])
    short_gain_figure(data, paths["overall"])
    pt_figure(data, paths["pt"])
    ga_figure(data, paths["ga"])
    sa_figure(data, paths["sa"])
    ga_full_figure(data, paths["full"])
    return paths


def footer(canvas, doc):
    canvas.saveState()
    width, _ = landscape(A4)
    canvas.setStrokeColor(colors.HexColor("#d8dee4"))
    canvas.line(15 * mm, 11 * mm, width - 15 * mm, 11 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(REPORT_MID)
    canvas.drawString(15 * mm, 6.5 * mm, "CIM warm-start研究 - 初学者向け2日間まとめ")
    canvas.drawRightString(width - 15 * mm, 6.5 * mm, str(doc.page))
    canvas.restoreState()


def compact_image(path: Path, width_mm: float) -> Image:
    probe = Image(str(path))
    width = width_mm * mm
    return Image(str(path), width=width, height=width * probe.imageHeight / probe.imageWidth,
                 hAlign="CENTER")


def build_pdf(data: dict, figures: dict[str, Path]) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=landscape(A4),
        leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm, bottomMargin=15 * mm,
        title="CIM warm-start研究 - 初学者向け2日間まとめ",
        author="cim-cac-maxcut-g22 project",
    )
    story = []

    story += [
        Spacer(1, 12 * mm),
        p("CIMを「良いスタート地点を作る装置」として使う研究", TITLE),
        Spacer(1, 3 * mm),
        p("CIM・Max-Cutを知らない人のための、ここ2日間の研究まとめ", SUBTITLE),
        Spacer(1, 12 * mm),
        callout("結論を一言で：CIMで先に良い候補を作ってから別の探索法へ渡すと、"
                "特に短時間では、randomから始めるより良い解を得やすい。"),
        Spacer(1, 9 * mm),
        three_boxes([
            ("最も安定した結果", "CIM → PT-ICM。G22とG55の両方で、短時間から長めの計算まで改善しました。", PALE_GREEN),
            ("問題によって差が出た結果", "CIM → GA。G22では差が小さく、G55では大きな改善が残りました。", PALE_BLUE),
            ("使い方に注意が必要", "CIM → SA。短時間では有効ですが、長く回すとrandom開始との差が消えました。", PALE_GRAY),
        ]),
        Spacer(1, 8 * mm),
        p("この資料では数式を使わず、「何を比べたか」「何が分かったか」「なぜそうなったか」を説明します。", BODY),
        PageBreak(),
    ]

    story += [
        p("1. まず、何を解いているのか", H1),
        Spacer(1, 2 * mm),
        p("Max-Cutは、点を2グループに分け、グループをまたぐ線の合計を最大にする問題です。"
          "配送網、回路配置、クラスタリングなどに現れる組合せ最適化問題の代表例です。", BODY),
        Spacer(1, 2 * mm),
        compact_image(figures["maxcut"], 225),
        Spacer(1, 3 * mm),
        three_boxes([
            ("cut値", "2グループをまたぐ線の合計。大きいほど良い解です。", PALE_GREEN),
            ("BKS", "Best Known Solutionの略。現在知られている最良値です。", PALE_BLUE),
            ("gap", "BKS - 今回のcut値。0が最良で、小さいほど良いと読みます。", PALE_GRAY),
        ]),
        Spacer(1, 3 * mm),
        callout("この資料のグラフでは、基本的に「gapが下がる」または「改善量がプラス」なら良い結果です。"),
        PageBreak(),
    ]

    story += [
        p("2. CIMをどう使う研究なのか", H1),
        Spacer(1, 5 * mm),
        data_table(
            ["段階", "担当", "たとえ", "この研究での役割"],
            [
                ["1", "CIM", "先に良さそうな出発地点を探す人", "短時間でMax-Cutの候補解を作る"],
                ["2", "GA / SA / PT-ICM", "候補を受け取り、さらに磨く人", "CIM解の近くや別方向を探索する"],
                ["3", "評価", "完成品を採点する人", "cut値、gap、必要時間を測る"],
            ],
            [20, 46, 89, 111],
        ),
        Spacer(1, 9 * mm),
        three_boxes([
            ("cold start", "GA・SA・PT-ICMをrandomな解から開始します。比較の基準です。", PALE_GRAY),
            ("CIM warm-start", "最初の解をCIMに作らせ、その解をGA・SA・PT-ICMへ渡します。", PALE_GREEN),
            ("研究の問い", "CIMを使う準備時間まで含めても、短時間でより良い解が得られるか？", PALE_BLUE),
        ]),
        Spacer(1, 9 * mm),
        callout("駅伝で言えば、CIMが第1走者、GA・SA・PT-ICMが第2走者です。"
                "速い第1走者から良い位置でタスキを受け取れるかを調べています。"),
        PageBreak(),
    ]

    story += [
        p("3. 実験条件 - 公平な比較にするために", H1),
        Spacer(1, 3 * mm),
        data_table(
            ["項目", "設定", "初心者向けの意味"],
            [
                ["対象問題", "G22、G55", "形と大きさが違う2種類のMax-Cut問題"],
                ["試行回数", "各条件16回", "偶然の当たり外れだけで判断しない"],
                ["CIM計算量", "600 rounds", "CIMは全条件で同じ長さだけ動かす"],
                ["乱数", "同じseedを対応させる", "比較する2条件へ同じ難しさの偶然を与える"],
                ["時間A", "後段だけ", "CIM解が既に用意済みの場合"],
                ["時間B", "CIM生成 + 後段", "実際に最初から最後まで使う場合"],
                ["各点", "独立した実験", "一本の実行が時間とともに動いた軌跡ではない"],
            ],
            [42, 72, 152],
        ),
        Spacer(1, 5 * mm),
        three_boxes([
            ("世代・反復・sweep", "各アルゴリズムをどれだけ長く動かしたかを表す、それぞれの計算量の単位です。", PALE_BLUE),
            ("平均改善量", "同じseedのcold結果をwarm結果が平均でどれだけ上回ったか。プラスならCIMが有利です。", PALE_GREEN),
            ("注意", "GA、SA、PT-ICMでは1単位の重さが違うので、最後は秒単位でも比較します。", PALE_GRAY),
        ]),
        PageBreak(),
    ]

    story += [
        p("4. 2日間の結果を最初に全体像で見る", H1),
        Spacer(1, 1 * mm),
        chart(figures["overall"], width_mm=235),
        Spacer(1, 3 * mm),
        p("棒が高いほど、同じ条件のrandom開始よりCIM開始の平均cutが高かったことを示します。"
          "棒の上の秒数はCIM生成を含む16試行バッチの実測総時間です。", BODY),
        Spacer(1, 3 * mm),
        callout("短時間では3手法すべてに改善が見えました。ただし、改善が長く残るかどうかは手法と問題で異なります。"),
        PageBreak(),
    ]

    story += [
        p("5. PT-ICMの結果 - 最も一貫して効果が出た", H1),
        Spacer(1, 1 * mm),
        chart(figures["pt"], width_mm=235),
        Spacer(1, 3 * mm),
        three_boxes([
            ("何をしたか", "CIM解をPT-ICMの1か所へ入れる方法と、少しずつ崩して複数の場所へ広げる方法を比較しました。", PALE_BLUE),
            ("分かったこと", "G22・G55の両方で全ての計算量において平均改善量がプラスでした。30 sweepでは全16組でwarmが勝ちました。", PALE_GREEN),
            ("なぜ効くか", "PT-ICMは複数の探索を行き来させます。良いCIM解と多様な周辺解を同時に置くと、良い領域から広く探せます。", PALE_GRAY),
        ]),
        Spacer(1, 3 * mm),
        callout("現時点で最も強く言える結果：CIMはPT-ICMの初期解作成役として有望です。"),
        PageBreak(),
    ]

    story += [
        p("6. GAの結果 - G22とG55で差が出た", H1),
        Spacer(1, 1 * mm),
        chart(figures["ga"], width_mm=235),
        Spacer(1, 3 * mm),
        three_boxes([
            ("G22", "GAだけでも早くBKS近くへ到達します。CIMから始めても改善できる余地が小さく、長く回すと差が消えました。", PALE_GRAY),
            ("G55", "random開始のGAが最初はBKSから遠く、CIM解が良い出発点として働きました。50世代でも平均+26.8でした。", PALE_GREEN),
            ("重要な解釈", "「CIMがG22に効かない」のではなく、「強いGAがG22で既に十分良く、上積みが見えにくい」です。", PALE_BLUE),
        ]),
        Spacer(1, 3 * mm),
        callout("初期解の価値は、問題そのものだけでなく、後段アルゴリズムがrandom開始でどれだけ苦戦するかで変わります。"),
        PageBreak(),
    ]

    story += [
        p("7. SAの結果 - 短い仕上げ役として使う", H1),
        Spacer(1, 1 * mm),
        chart(figures["sa"], width_mm=235),
        Spacer(1, 3 * mm),
        three_boxes([
            ("短時間", "10万反復ではG22で平均+111.2、G55で+217.5。CIM解を少し磨く使い方は有効でした。", PALE_GREEN),
            ("長時間", "3億反復ではG22で+0.3、G55で-1.5。差はほぼなくなり、明確な優位とは言えません。", PALE_GRAY),
            ("なぜか", "長く動かすと、cold側も十分に良い領域へ到達します。また同じ温度設定を長く使うと初期解の記憶が薄れます。", PALE_BLUE),
        ]),
        Spacer(1, 3 * mm),
        callout("SAでは「CIMの後に短く仕上げる」が自然です。長いSA専用の温度設計はまだ調整していません。"),
        PageBreak(),
    ]

    story += [
        p("8. GAの全初期集団をCIM解にした追加実験", H1),
        Spacer(1, 1 * mm),
        chart(figures["full"], width_mm=235),
        Spacer(1, 3 * mm),
        three_boxes([
            ("比較した3条件", "全てrandom、1個体だけCIM、全個体をCIM。全個体CIMではそれぞれ異なるrandom seedを使いました。", PALE_BLUE),
            ("品質", "全CIMは初期から最良gapを押し下げました。G22は0世代でgap 1、G55はgap 82でした。", PALE_GREEN),
            ("時間", "全個体のCIM生成にG22約2.95秒、G55約1.95秒。1個体だけの場合より追加費用がかかります。", PALE_GRAY),
        ]),
        Spacer(1, 3 * mm),
        callout("全CIM集団は短期品質を最優先するときに有効。ただし費用対効果では、1個体だけCIMの方が有利な場面が多いです。"),
        PageBreak(),
    ]

    story += [
        p("9. なぜG22では小さく、G55やK2000では大きく見えるのか", H1),
        Spacer(1, 3 * mm),
        data_table(
            ["問題", "特徴", "random開始の後段", "CIM初期解の価値", "今回の位置づけ"],
            [
                ["G22", "2000点・疎な問題", "GAがすぐBKS近くへ行く", "GAでは上積み余地が小さい。ただし短いPT-ICMでは大きい", "この2日で再計測"],
                ["G55", "5000点・より大規模で疎", "短いGAでは良い領域へ入りにくい", "良い出発点の差が長く残る", "この2日で再計測"],
                ["K2000", "2000点・非常に密で重み付き", "局所探索の開始位置が結果を左右しやすい", "CIMが良い盆地を示し、局所探索が仕上げやすい", "過去の関連結果"],
            ],
            [30, 55, 65, 79, 37],
        ),
        Spacer(1, 7 * mm),
        three_boxes([
            ("天井効果", "すでにBKS近くなら、それ以上改善できる数値がほとんど残っていません。G22のGAが該当します。", PALE_GRAY),
            ("問題の大きさ・密度", "大きい問題や密な問題では探索空間が複雑になり、良い開始位置の価値が上がる場合があります。", PALE_BLUE),
            ("後段との相性", "同じCIM解でもGA、SA、PT-ICMで使い方が違います。効果は「問題×後段手法」で決まります。", PALE_GREEN),
        ]),
        Spacer(1, 5 * mm),
        callout("K2000のgap 794 → 120というCIM→局所探索の結果は過去データです。今回の16試行・同一条件比較には含めていません。"),
        PageBreak(),
    ]

    story += [
        p("10. 時間・パラメータ調整・結果の限界", H1),
        Spacer(1, 3 * mm),
        data_table(
            ["論点", "今回行ったこと", "まだ行っていないこと"],
            [
                ["時間", "CIM生成込みと除外の両方を測定", "別PC・GPU・実機CIMでの再測定"],
                ["CIM", "全て600 roundsで統一", "問題ごとのrounds最適化"],
                ["GA", "coldとwarmで同一の既存パラメータ", "warm専用の集団サイズ・交叉率調整"],
                ["SA", "coldとwarmで温度・反復数・乱数位置を一致", "warm専用の温度・冷却スケジュール調整"],
                ["PT-ICM", "singleとladderの2方式を実装", "レプリカ数・温度列・崩し率の広い探索"],
                ["統計", "2問題×各16試行の対応比較", "多数の未使用問題での最終評価"],
            ],
            [44, 111, 111],
        ),
        Spacer(1, 7 * mm),
        three_boxes([
            ("チューニング量", "warm-start専用の本格的なパラメータ探索はしていません。既存設定を揃えた初期検証です。", PALE_BLUE),
            ("言えること", "G22・G55のこの条件では、短時間warm-startが有望。特にPT-ICMで一貫しています。", PALE_GREEN),
            ("まだ言えないこと", "全Max-Cut問題で必ず有効、または全時間域で必ず速いとはまだ言えません。", PALE_GRAY),
        ]),
        Spacer(1, 4 * mm),
        callout("現在の結果は「研究仮説を支持する初期証拠」です。一般化を主張するには、未調整の別問題で再現させる必要があります。"),
        PageBreak(),
    ]

    story += [
        p("11. この2日間で分かったことと、次にやるべきこと", H1),
        Spacer(1, 5 * mm),
        data_table(
            ["優先度", "次の実験", "理由"],
            [
                ["1", "未使用のG-set問題を複数選び、同じ設定のまま評価", "G22・G55だけに合った結果ではないことを確認する"],
                ["2", "PT-ICMのladder方式を中心に試行数を30～50へ増やす", "現在もっとも一貫した結果の信頼性を高める"],
                ["3", "GA集団中のCIM個体割合を1、25%、50%、100%で比較", "品質とCIM生成時間のちょうど良い点を探す"],
                ["4", "SAのwarm専用温度・短時間スケジュールを調整", "長時間化ではなく、短い仕上げ性能を伸ばす"],
                ["5", "K2000を今回と同じ16試行・時間込み条件で再検証", "過去結果を現在の公平な比較方法で確認する"],
            ],
            [28, 143, 95],
        ),
        Spacer(1, 8 * mm),
        callout("研究としての中心主張案：CIMは単独の最終解法としてだけでなく、"
                "古典的探索法へ良い初期状態を供給する前処理として価値がある。"
                "効果は短時間で大きく、後段手法ではPT-ICMが最も安定している。"),
        Spacer(1, 8 * mm),
        three_boxes([
            ("発表で最初に言うこと", "CIMが答えを完成させるのではなく、後の探索を有利な場所から始めさせる研究です。", PALE_BLUE),
            ("結果を一言で", "短時間では効く。PT-ICMで最も安定、GAは問題依存、SAは短い仕上げ向きです。", PALE_GREEN),
            ("正直に付け加えること", "2問題・16試行の初期検証で、warm専用チューニングと汎化確認はこれからです。", PALE_GRAY),
        ]),
        Spacer(1, 5 * mm),
        p("データ保存先：results/2026-07-24/cim_warmstart/ および results/2026-07-26/cim_warmstart/。"
          "全結果は共通のweighted-cut採点で再計算しています。", BODY_SMALL),
    ]

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    configure_matplotlib()
    data = load_all()
    figures = make_figures(data)
    build_pdf(data, figures)
    print(OUT)


if __name__ == "__main__":
    main()
