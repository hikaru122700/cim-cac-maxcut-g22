"""反強磁性検証の生データから日本語図とRESULTS.mdを作成する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False,
                     "font.size": 11, "axes.titleweight": "bold"})
COLORS = ["#808993", "#D29620", "#328BBC", "#C34450"]


def style(ax):
    """全方向の内向き目盛りを適用する。"""
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(axis="y", alpha=.18)


def spinplot(ax, state, edges, title, note, highlight=()):
    """固定の格子座標にスピンと不満足な結合を描画する。"""
    xy = np.array([(c, 3-r) for r in range(4) for c in range(4)])
    for i, j in edges:
        bad = state[i] == state[j]
        ax.plot(xy[[i,j],0], xy[[i,j],1], lw=3.4 if bad else 1.7,
                c="#D54738" if bad else "#CED4DA", zorder=1)
    ax.scatter(xy[:,0],xy[:,1],s=380,c=["#176B9A" if s>0 else "#E6AB35" for s in state],
               edgecolors="white",linewidths=1.3,zorder=2)
    for i,(x,y) in enumerate(xy):
        ax.text(x,y,"+" if state[i]>0 else "−",ha="center",va="center",fontsize=16,
                c="white" if state[i]>0 else "#40300F",zorder=3)
    if highlight:
        ax.scatter(xy[list(highlight),0],xy[list(highlight),1],s=660,
                   facecolors="none",edgecolors="#273746",linestyle="--",zorder=4)
    ax.set_title(title,pad=12,fontsize=14)
    ax.text(.5,-.085,note,ha="center",va="top",transform=ax.transAxes,linespacing=1.5)
    ax.set_aspect("equal")
    ax.set_xlim(-.45,3.45)
    ax.set_ylim(-.35,3.35)
    ax.axis("off")


def main():
    """既存成果を上書きせず図と報告書を作る。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    out=args.run_dir
    names=["states.png","path_length.png","sa.png","ga.png","control.png","RESULTS.md"]
    if any((out/x).exists() for x in names):
        raise FileExistsError("図・報告書が存在します。上書きしません")
    summary=json.loads((out/"summary.json").read_text(encoding="utf-8"))
    diag=json.loads((out/"diagnostics.json").read_text(encoding="utf-8"))
    data=np.load(out/"triangular.npz")
    exact=np.load(out/"diagnostics.npz")
    edges=summary["graphs"]["triangular"]["edges"]
    state_fig,axes=plt.subplots(1,2,figsize=(11,5.2))
    for ax,ex in zip(axes,diag["examples"]):
        state=data["source_states"][3,ex["cim_seed"]]
        spinplot(ax,state,edges,f"CIM seed {ex['cim_seed']}：cut = 23",
                 f"非悪化経路の最短反転数：{ex['distance']}回\n"
                 f"SA 10 sweepの最適到達率：{100*ex['success']:.1f}%（{ex['successes']}/256）",
                 highlight=(ex["flip_vertices"][0],))
    state_fig.suptitle("同じCIM品質でも、最適解への進みやすさが異なる",fontsize=17,y=.97)
    state_fig.text(.5,.07,"青＋／黄−：スピン　赤線：不満足辺（両例とも10本）　点線円：最短経路の最初の反転点",
                   ha="center",fontsize=9)
    state_fig.subplots_adjust(top=.79,bottom=.27,left=.04,right=.96,wspace=.15)
    state_fig.savefig(out/"states.png",dpi=180)
    plt.close(state_fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout="constrained")
    gs=diag["groups"]
    axes[0].bar([x['distance'] for x in gs],[100*x['success'] for x in gs],color="#328BBC",width=.72)
    for x in gs:
        axes[0].text(x['distance'],100*x['success']+1.4,f"{100*x['success']:.1f}%",ha="center",fontsize=9)
    axes[0].set(xlabel="非悪化経路の最短反転数",ylabel="SA 10 sweepの最適到達率（%）",ylim=(0,65),xticks=range(1,9))
    axes[0].set_title("初期cut = 23の233候補を比較")
    for ex,color in zip(diag["examples"],["#176B9A","#C34450"]):
        axes[1].plot(range(len(ex['route_cuts'])),ex['route_cuts'],"o-",color=color,
                     label=f"seed {ex['cim_seed']}（最短{ex['distance']}反転）")
    axes[1].set(xlabel="反転回数",ylabel="カット値",ylim=(22.8,24.4),yticks=[23,24])
    axes[1].set_title("全列挙から求めた非悪化の最短経路")
    axes[1].legend(loc="upper left",fontsize=9)
    for ax in axes: style(ax)
    fig.savefig(out/"path_length.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    for ax,key,title in zip(axes,["square","triangular"],["正方格子：最適cut 24 / 24辺","対角辺付き格子：最適cut 24 / 33辺"]):
        g=summary['graphs'][key]
        for source,label,color in [("random","ランダム初期解",COLORS[0]),("cim128","CIM 128 round",COLORS[2]),("cim1500","CIM 1500 round",COLORS[3])]:
            rows=sorted([x for x in g['sa_rows'] if x['source']==source],key=lambda x:x['sweeps'])
            ax.plot(range(4),[100*x['success'] for x in rows],"o-",label=label,color=color)
        ax.set(title=title,xlabel="追加SAのsweep数（間隔は等しく表示）",ylabel="最適到達率（%）",
               xticks=range(4),xticklabels=[0,1,10,100],ylim=(-3,105))
        ax.legend(fontsize=9,loc="lower right")
        style(ax)
    fig.savefig(out/"sa.png",dpi=180)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.1),layout="constrained")
    for ax,ts in zip(axes,[4,32]):
        for source,label,color in [("random","ランダム集団",COLORS[0]),("cim1500","全CIM集団",COLORS[3])]:
            rows=[x for x in summary['graphs']['triangular']['ga_rows'] if x['source']==source and x['ts_iters']==ts]
            ax.plot([x['generation'] for x in rows],[100*x['success'] for x in rows],"o-",label=label,color=color)
        ax.set(title=f"TS 1回あたり{ts}反復",xlabel="観測段階",ylabel="集団で最適解を得た割合（%）",ylim=(-3,105),
               xticks=[-1,0,2,4,6,8],xticklabels=["生集団","初期TS後","2世代","4世代","6世代","8世代"])
        ax.tick_params(axis="x",labelsize=8)
        ax.legend(fontsize=9)
        style(ax)
    fig.savefig(out/"ga.png",dpi=180)
    plt.close(fig)

    control=np.load(out/"controls.npz")
    fig,axes=plt.subplots(1,3,figsize=(12,4.5))
    sq=summary['graphs']['square']['edges']
    for ax,si,title in [(axes[0],0,"A：内点1個の欠陥"),(axes[1],1,"B：逆位相の領域境界")]:
        p=float(np.mean(control['cuts'][1,1,si]==24))
        spinplot(ax,control['initial'][si],sq,title,f"初期cut = 20\nSA 10 sweep：{100*p:.1f}%が最適",highlight=(5,) if si==0 else ())
    for ti,t,col in [(0,.1,"#328BBC"),(1,.5,"#C34450"),(2,2.,"#D29620")]:
        axes[2].plot(range(3),[100*np.mean(control['cuts'][ti,j,1]==24) for j in range(3)],"o-",label=f"初期温度 {t}",color=col)
    axes[2].set(title="Bは温度・予算で脱出率が変わる",xlabel="SAのsweep数",ylabel="最適到達率（%）",
                xticks=range(3),xticklabels=[1,10,100],ylim=(-3,105))
    axes[2].legend(fontsize=8)
    style(axes[2])
    fig.subplots_adjust(top=.82,bottom=.24,left=.035,right=.98,wspace=.38)
    fig.savefig(out/"control.png",dpi=180)
    plt.close(fig)

    def sa(key,source,sweeps):
        return next(x for x in summary['graphs'][key]['sa_rows'] if x['source']==source and x['sweeps']==sweeps)
    def ga(ts,source,gen):
        return next(x for x in summary['graphs']['triangular']['ga_rows'] if x['ts_iters']==ts and x['source']==source and x['generation']==gen)
    exa,exb=diag['examples']
    sa_table=[]
    for key,name in [('square','正方格子'),('triangular','対角辺付き')]:
        for src,label in [('random','ランダム'),('cim1500','CIM 1500 round')]:
            cells=[f"{sa(key,src,b)['mean']:.3f} / {100*sa(key,src,b)['success']:.1f}%" for b in [0,1,10,100]]
            sa_table.append(f"| {name} | {label} | "+" | ".join(cells)+" |")
    ga_table=[]
    for ts in [4,32]:
        for src,label in [('random','ランダム'),('cim1500','全CIM')]:
            cells=[f"{100*ga(ts,src,g)['success']:.1f}%" for g in [-1,0,8]]
            ga_table.append(f"| {ts} | {label} | "+" | ".join(cells)+" |")
    group_table="\n".join(f"| {g['distance']} | {g['candidates']} | {g['successes']}/{g['attempts']} | {100*g['success']:.1f}% |" for g in gs)
    cross=diag['crossover']['triangular']
    report=f'''<style>
@page {{ size: A4; margin: 14mm 15mm; }}
body {{ padding: 0 !important; font-size: 9.5pt !important; line-height: 1.45 !important; }}
h1 {{ font-size: 18pt !important; }}
h2 {{ font-size: 14pt !important; margin-top: 12px !important; }}
h3 {{ font-size: 11pt !important; margin-top: 8px; }}
table {{ font-size: 8.5pt !important; width: 100%; }}
td, th {{ padding: 4px 6px !important; }}
img {{ max-height: 100mm; object-fit: contain; }}
tr, img {{ break-inside: avoid; }}
.pagebreak {{ break-before: page; }}
</style>

# 反強磁性モデルでのCIM解の後段評価

実験日：2026-09-28。数値CIM・SA・メメティックGAの実測。全て16スピン、単位正重み、外場なし、開放境界。

## 1. 結論と実際の配置

**同じカット値でも、解の配置によって短いSAでの改善率が変わった。** 対角辺付き格子で、CIMの初期cutが23の233候補を比較すると、非悪化経路が最短1反転の群では最適到達率52.8%、8反転の群では10.0%だった。各候補256回の独立乱数追試による。

![実CIM出力の比較](./states.png)

両図はCIM 1500 roundの実出力から選択。両方ともcut=23、不満足辺10本。正の辺重みを反強磁性結合として埋め込むため、隣同士が逆符号の辺が得点になる。最適値は24で、不満足辺9本が残る。三角形があり全結合の同時満足はできない。

- **左（seed {exa['cim_seed']}）**：1点の反転で24へ到達できる。SA 160回の反転提案で、127/256回が最適へ到達。
- **右（seed {exb['cim_seed']}）**：単独反転で即改善できず、cut=23のまま移動してから24へ進む経路が必要。27/256回が最適へ到達。
- 例は成功率を見て選ばず、最短経路長1と8の各群で最小のCIM seedを選んだ。最短経路は実際のSA軌跡ではなく、全列挙に基づく解析結果。

この小規模例では、**不満足辺の本数だけでなく、改善へつながる配置変更の経路を見る意義**を確認できた。

<div class="pagebreak"></div>

## 2. 同じcut=23の解を、別のSA乱数で追試

![経路長と追試結果](./path_length.png)

全65,536配置を列挙。cutが23以上の配置だけを通る1スピン反転経路について、最適解から幅優先探索で最短反転数を求めた。対象の233候補は全て悪化なしで最適解へ行けた。長い経路の例は「エネルギーを上げなければ脱出できない」型ではなく、**同じ品質の状態を何度も移動する必要がある**型である。

| 最短反転数 | CIM候補数 | 成功回数 / SA試行数 | 最適到達率 |
|---:|---:|---:|---:|
{group_table}

SAの条件は初期温度0.5、終端パラメータ0.01、10 sweep=160反転提案。探索中の最良cutを保持する。主実験を見て経路長という特徴を選び、その後、主実験と重複しない乱数（2,000,000以降）で各候補256回追試した。

**解釈の範囲：** 同じ固定配置での再現確認であり、未見グラフへの予測性能を検証したものではない。また経路長は最適解を知った全列挙解析であり、この計算法をそのまま大規模問題の安価な選別器には使えない。非悪化経路長以外の配置特性も群間で異なり得るため、経路長だけの因果効果とは主張しない。

<div class="pagebreak"></div>

## 3. CIMからSAへ渡した全体結果

![SAでの最適到達率](./sa.png)

| 問題 | 初期解 | 追加SAなし | 1 sweep | 10 sweep | 100 sweep |
|---|---|---:|---:|---:|---:|
{chr(10).join(sa_table)}

表の各セルは「平均最良cut / 最適到達率」。最適cutは両問題とも24。追加SAなしは256個の初期解、SAありは256候補×32乱数=8,192試行。候補生成のseedを揃えたCIMの32/128/1500 roundは同じ軌跡の異なる探索予算であり、独立な768試行とは数えない。

- **正方格子（24辺）**：CIM 1500 roundは256/256試行で最適。ここでは後段改善の余地がない。
- **対角辺付き格子（33辺）**：CIMだけでは23/256試行が最適。CIM→SA 100 sweepでは6,643/8,192試行が最適、平均cutは23.090から23.811へ+0.721。
- 同じ追加SA 100 sweepで、ランダム初期解は78.1%、CIM初期解は81.1%。差は短いSAほど大きいが、**この比較はCIM生成時間を含む総時間の比較ではない**。

CIMの返却解は「最後のroundの状態」ではなく、各runの全roundで記録した最良cutの状態。最適値24は既知ベストの引用でなく、全列挙と独立な隣接行列計算で検証した値である。

<div class="pagebreak"></div>

## 4. GA：初期局所探索と交叉を分ける

![GAの段階別比較](./ga.png)

対角辺付き格子で、8個体×32集団、8世代を実行。全CIM集団は各個体が異なるCIM seed。表は「集団内に最適解を得たrunの割合」であり、前ページのSAの1候補当たりの率とは母数が違う。

| TS反復 / 回 | 初期集団 | 生集団 | 初期TS後 | 8世代後 |
|---:|---|---:|---:|---:|
{chr(10).join(ga_table)}

TS 32反復では、CIM集団は**交叉前の初期TSだけで全32集団が最適解を獲得**。この成功を交叉の効果と解釈することはできない。TS 4反復でも24/32集団から26/32集団への追加改善に留まった。

### 同じ親にTSだけをかける対照

CIM 128 roundの実候補から、親同士のcutが同じ128ペアを抽出。各ペア32回、現行交叉後にTSを4反復する場合と、親を交互に選んでTSだけを4反復する場合を比較した。これは初期TS前の親を用いる診断で、全GAの置換実験ではない。

| 条件 | 平均cut | 親のcutを超えた割合 |
|---|---:|---:|
| 親そのもの | {cross['parent_mean']:.3f} | - |
| 交叉直後 | {cross['raw_child_mean']:.3f} | {100*cross['raw_child_improvement']:.1f}% |
| 交叉＋TS 4反復 | {cross['child_ts_mean']:.3f} | {100*cross['child_ts_improvement']:.1f}% |
| 親へTS 4反復 | {cross['parent_ts_mean']:.3f} | {100*cross['parent_ts_improvement']:.1f}% |

今回の診断では、交叉＋TSは親へのTSだけを平均で上回らなかった。**「異なるCIM解を交叉すれば良くなる」という仮説は、この条件では支持されなかった。** より大きな問題や構造を保存する交叉へ一般化した結論ではない。

<div class="pagebreak"></div>

## 5. 前回の模式例A/Bも実際にSAで検証

![模式例からSAを実行](./control.png)

ここだけは実CIM出力ではなく、前回作成した手作り配置を初期解にした対照実験。両者は同じ正方格子・cut=20・不満足辺4本。初期温度0.5、10 sweep、各1,024試行で、Aは1,023/1,024（99.9%）、Bは216/1,024（21.1%）が最適へ到達した。

Aには+4の単独反転利得がある。Bは全16個の単独反転が悪化し、最大利得も−1。Bの最適到達率は初期温度2.0、100 sweepでは99.5%まで上がった。したがって「Bは解けない」ではなく、**温度と探索予算によって、同じcutの初期解の価値が変わる**という結果である。

## 6. 今回分かったことと次の検証

1. 正方格子は基準動作の確認に適するが、CIMが簡単に解き切るため、後段の改善可能性には対角辺付きの問題が適していた。
2. SA向けには、不満足辺の数に加え、すぐ改善する反転の有無と、同じ品質の状態を移動して改善へ至る構造を調べる価値がある。
3. GA向けには、初期TSで解がどう変わるかと、交叉にTS単独以上の効果があるかを別々に測る必要がある。
4. 次は対角辺の配置を変えた複数の小問題で再現性を確認し、短い局所探索から得られる安価な指標を比較する。その後、G55/G70へ進む。

制約：2種類の固定小規模問題、固定CIM設定での数値実験。実機の性能、総時間の高速化、大規模問題への汎化は未検証。GAは小問題に合わせTS予算を4/32反復に設定しており、既存G-setベンチマークの20,000反復等とは異なる。

<div class="pagebreak"></div>

## 7. 更新式・実験条件・再現性

![プロジェクト共通の基準更新式](../../../../docs/assets/base_update_equations.png)

**更新式の変更なし。** `modules/CIM.py` の既存進行波モデルをそのまま使用した。基準図の結合行列Jへ、今回は正方格子または対角辺付き格子を入れ、各辺を−0.03とした。ポンプの増加量や増幅・飽和・雑音処理に新しい項は加えていない。図は共通の説明用基準式であり、実際の物理定数から利得・雑音を計算する手順の正本は保存したCIM.py。

| 項目 | 設定 |
|---|---|
| 頂点・辺・境界 | 16頂点、正方24辺 / 対角付き33辺、開放境界 |
| 最適性検証 | 65,536配置の全列挙。両問題とも最適cut 24、最適配置2個（全反転対） |
| CIM | 256 seed（0〜255）、32/128/1500 round、各runの最良状態を返す |
| CIM物理定数 | kappa=130、L=0.05、gamma=42.09、損失11 dB |
| CIMポンプ・雑音 | 増加量0.05 mW/round、帯域1 GHz、光子エネルギー1.28e−19 J |
| SA主比較 | 1/10/100 sweep、1 sweep=16回のランダム反転提案、初期温度0.5、終端0.01 |
| SA繰り返し | 主比較は各候補32回。経路長追試は各候補256回。手作り対照は1,024回 |
| GA | 8個体×32 run、8世代、TS 4/32反復、cr=32、摂動2、tenure係数2、品質重み0.6 |
| 対照 | ランダム初期解、手作りA/B、親へのTSのみ |

GA.pyの引数`init_population`に既存の二重定義があり読み込めなかったため、重複する宣言1行のみ削除した。GA・SA warm-startの既存テスト7件が通過。今回の出力は全て辺リストから再計算し、最適値を超えていないこととソルバの返却値の一致を検証した。

### 再実行（プロジェクトルートから）

```powershell
python scripts/benchmarks/antiferro_seed_validation.py
python scripts/benchmarks/antiferro_seed_diagnostics.py <新しい出力先>
python scripts/reporting/antiferro_seed_report.py <新しい出力先>
python scripts/utils/md_to_pdf.py <新しい出力先>/RESULTS.md
```

出力先は日付×実験種別×版番号で自動採番。今回の本実験はv2、v1は32候補のsmoke試験。summary.jsonに設定・ソースSHA-256、各npzにCIM配置・SA結果・GA集団・全列挙結果を保存。diagnostics.json/npzに独立乱数追試と最短経路を保存。ソルバと実験コードのスナップショットも同梱。
'''
    (out/"RESULTS.md").write_text(report,encoding="utf-8")
    print(out/"RESULTS.md")


if __name__=="__main__":
    main()
