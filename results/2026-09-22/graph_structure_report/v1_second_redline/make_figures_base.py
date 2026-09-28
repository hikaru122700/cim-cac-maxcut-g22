"""Vector figures for the Japanese LaTeX research review."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)
DATA_FILE = HERE / 'data' / 'summary.json'
if not DATA_FILE.exists():
    DATA_FILE = ROOT / 'results/2026-09-21/graph_connectivity/v2_structure_G22_G55_G70/summary.json'
DATA = json.loads(DATA_FILE.read_text())['graphs']
NAVY, TEAL, ORANGE, GREY = '#203C55', '#138B86', '#D77932', '#AAB5BF'
plt.rcParams.update({'font.family': 'Yu Gothic', 'font.size': 15, 'axes.unicode_minus': False,
                     'pdf.fonttype': 42, 'axes.spines.top': True, 'axes.spines.right': True,
                     'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.edgecolor': GREY})

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)

def graph(ax, g, pos, colors=None, special=(), labels=False):
    special = {frozenset(e) for e in special}
    normal = [e for e in g.edges if frozenset(e) not in special]
    highlighted = [e for e in g.edges if frozenset(e) in special]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=normal, edge_color=GREY, width=2)
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=highlighted, edge_color=ORANGE, width=3.5)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors or TEAL, node_size=330,
                           edgecolors='white', linewidths=1.5)
    if labels:
        nx.draw_networkx_labels(g, pos, ax=ax, font_color='white', font_size=10)
    ax.set_aspect('equal'); ax.axis('off'); ax.margins(0.2)

# An identical graph under two assignments; all edge counts are exact.
g = nx.Graph([(0,1),(1,2),(2,3),(3,0),(0,2)])
pos = {0:(0,1),1:(1,1),2:(1,0),3:(0,0)}
fig, axes = plt.subplots(1, 2, figsize=(9,3.4))
for ax, s, title in zip(axes, [[0,0,0,1],[0,1,0,1]], ['配置A：切れている辺は2本','配置B：切れている辺は4本']):
    cut = [(u,v) for u,v in g.edges if s[u] != s[v]]
    graph(ax,g,pos,[TEAL if s[u]==0 else NAVY for u in g],cut)
    ax.set_title(title,fontsize=19,pad=12)
fig.text(.5,.015,'点の色＝二つの組　／　オレンジの辺＝cut',ha='center',fontsize=17)
save(fig,'cut_example')

# A bridge joining cyclic blocks, with a tail and an isolated vertex.
g = nx.Graph([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,6),(6,3),(4,7),(7,8)])
g.add_node(9)
pos={0:(0,1),1:(0,-1),2:(1.2,0),3:(3,0),4:(4.2,1),5:(5.4,0),6:(4.2,-1),7:(5.5,1.8),8:(6.8,1.8),9:(6.8,-1)}
fig,ax=plt.subplots(figsize=(10,4))
graph(ax,g,pos,[TEAL if u not in (7,8,9) else GREY for u in g],nx.bridges(g))
ax.text(.35,-1.65,'三角形のブロック',ha='center')
ax.text(4.2,-1.65,'四角形のブロック',ha='center')
ax.text(2.1,.55,'橋',ha='center',color=ORANGE,fontweight='bold')
ax.text(6.5,2.35,'木状の枝',ha='center',color=ORANGE)
ax.text(6.8,-1.65,'孤立点',ha='center')
ax.set_ylim(-2.1,2.8)
save(fig,'decomposition')

fig, axes=plt.subplots(1,3,figsize=(10,3.1))
g=nx.cycle_graph(4);g.add_edges_from([(1,4),(4,5),(2,6)]);g.add_node(7)
pos={0:(0,0),1:(0,1),2:(1,1),3:(1,0),4:(-.8,1.5),5:(-1.6,2),6:(1.8,1.7),7:(2,0)}
for ax,nodes,title in zip(axes,[list(g),[0,1,2,3,4],[0,1,2,3]],['最初のグラフ','1回目：孤立点と葉を除く','2回目：四角形が残る']):
    h=g.subgraph(nodes)
    graph(ax,h,pos,[TEAL if n<4 else GREY for n in h])
    ax.set_xlim(-2,2.5);ax.set_ylim(-.5,2.5);ax.set_title(title,fontsize=15)
save(fig,'core_peeling')

fig, axes=plt.subplots(1,2,figsize=(10,4),gridspec_kw={'width_ratios':[1.05,1]})
y=np.arange(3)
parts=[np.array([d['isolated_vertices'] for d in DATA]),
       np.array([d['vertices']-d['isolated_vertices']-d['two_core_vertices'] for d in DATA]),
       np.array([d['two_core_vertices'] for d in DATA])]
left=np.zeros(3)
for a,c,label in zip(parts,[GREY,ORANGE,TEAL],['孤立点','それ以外のcore外頂点','2-core']):
    axes[0].barh(y,a,left=left,color=c,label=label,height=.5)
    for i,val in enumerate(a):
        if val>800:axes[0].text(left[i]+val/2,i,f'{val:,}',ha='center',va='center',color='white',fontsize=14)
    left+=a
axes[0].set_yticks(y,['G22','G55','G70']);axes[0].invert_yaxis()
axes[0].set_xlabel('頂点数');axes[0].set_xlim(0,10500)
axes[0].set_title('頂点の内訳',fontsize=17)
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.25),frameon=False,fontsize=13,ncol=1)
bridges=[d['bridges']/d['edges']*100 for d in DATA]
axes[1].barh(y,bridges,color=ORANGE,height=.5)
axes[1].set_yticks(y,['G22','G55','G70']);axes[1].invert_yaxis();axes[1].set_xlim(0,43)
axes[1].set_xlabel('全辺に占める橋の割合（%）');axes[1].set_title('橋の割合',fontsize=17)
for i,val in enumerate(bridges):axes[1].text(val+.8,i,f'{val:.2f}%',va='center',fontsize=14)
for ax in axes:ax.tick_params(direction='in',which='both',top=True,right=True)
fig.tight_layout(w_pad=2)
save(fig,'comparison')

fig,axes=plt.subplots(1,3,figsize=(10,3))
for ax,d in zip(axes,DATA):
    hist={int(k):v for k,v in d['core_number_histogram'].items()}
    x=list(hist); h=[hist[k]/d['vertices']*100 for k in x]
    ax.bar(x,h,color=TEAL,width=.75)
    ax.set_ylim(0,100);ax.set_xlim(-.8,14.8);ax.set_xticks([0,4,8,12,14])
    ax.set_title(d['graph'],fontsize=17);ax.set_xlabel('core number')
    ax.tick_params(direction='in',which='both',top=True,right=True)
axes[0].set_ylabel('全頂点に占める割合（%）')
fig.tight_layout();save(fig,'core_distribution')

fig,axes=plt.subplots(1,3,figsize=(10,3.2))
for ax,n in zip(axes,[4,3,5]):
    g=nx.cycle_graph(n);pos=nx.circular_layout(g)
    s=[u%2 for u in range(n)]
    graph(ax,g,pos,[TEAL if s[u]==0 else NAVY for u in g],[(u,v) for u,v in g.edges if s[u]!=s[v]])
    ax.set_title({4:'四角形：4 / 4 本',3:'三角形：最大2 / 3 本',5:'五角形：最大4 / 5 本'}[n],fontsize=17)
save(fig,'cycles')

fig,ax=plt.subplots(figsize=(9,3.1))
t=np.linspace(0,1,120)
ax.plot(t,70+5*(1-np.exp(-6*t)),lw=2.8,color=ORANGE,label='候補A：初期値は高い')
ax.plot(t,64+20*(1-np.exp(-3*t)),lw=2.8,color=TEAL,label='候補B：後から追い越す')
ax.scatter([0,0],[70,64],c=[ORANGE,TEAL],s=55,zorder=3)
ax.set_xlabel('後段探索に使った予算');ax.set_ylabel('到達した最良cut（概念図）')
ax.set_xticks([0,1],['開始','所定予算 B']);ax.set_yticks([])
ax.legend(frameon=False,loc='lower right');ax.tick_params(direction='in',top=True,right=True)
ax.text(.02,.97,'説明用の架空の軌跡。実験結果ではない。',transform=ax.transAxes,va='top',fontsize=13,color=NAVY)
fig.tight_layout();save(fig,'future_value')
print(OUT)
