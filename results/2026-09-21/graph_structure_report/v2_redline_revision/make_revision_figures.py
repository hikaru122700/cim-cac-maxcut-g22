"""Illustrations added in response to the handwritten review; vector PDF output."""
from pathlib import Path
import json
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
OUT = P / 'figures'
OUT.mkdir(exist_ok=True)
TEAL, NAVY, ORANGE, GREY = '#138B86', '#203C55', '#D77932', '#AAB5BF'
plt.rcParams.update({'font.family':'Yu Gothic','font.size':15,'axes.unicode_minus':False,
                     'pdf.fonttype':42,'text.color':NAVY,'axes.labelcolor':NAVY})

def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def draw(ax,g,pos,colors=None,special=(),labels=None):
    special={frozenset(e) for e in special}
    nx.draw_networkx_edges(g,pos,ax=ax,edgelist=[e for e in g.edges if frozenset(e) not in special],edge_color=GREY,width=2)
    nx.draw_networkx_edges(g,pos,ax=ax,edgelist=[e for e in g.edges if frozenset(e) in special],edge_color=ORANGE,width=3)
    nx.draw_networkx_nodes(g,pos,ax=ax,node_color=colors or TEAL,node_size=330,edgecolors='white',linewidths=1)
    if labels:nx.draw_networkx_labels(g,pos,labels=labels,ax=ax,font_family='Yu Gothic',font_color='white',font_size=12)
    ax.axis('off');ax.set_aspect('equal');ax.margins(.18)

# Bridge and articulation point are different objects; show removal explicitly.
fig,axes=plt.subplots(2,2,figsize=(10,5.7))
g=nx.Graph([(0,1),(1,2),(2,0),(2,3),(3,4),(4,5),(5,3)])
pos={0:(0,1),1:(0,-1),2:(1,0),3:(3,0),4:(4,1),5:(4,-1)}
draw(axes[0,0],g,pos,special=[(2,3)]);axes[0,0].set_title('橋：オレンジの「辺」を除く',fontsize=16)
h=g.copy();h.remove_edge(2,3);draw(axes[0,1],h,pos);axes[0,1].set_title('連結成分が1個から2個へ',fontsize=16)
g=nx.Graph([(0,1),(1,2),(2,0),(0,3),(3,4),(4,0)])
pos={0:(2,0),1:(0,1),2:(0,-1),3:(4,1),4:(4,-1)}
draw(axes[1,0],g,pos,[ORANGE if n==0 else TEAL for n in g]);axes[1,0].set_title('関節点：オレンジの「点」を除く',fontsize=16)
h=g.copy();h.remove_node(0);draw(axes[1,1],h,pos);axes[1,1].set_title('その点につながる辺も消える',fontsize=16)
fig.tight_layout(h_pad=2);save(fig,'bridge_articulation')

# Exact bridge-flip example: both 4-cycles are already optimal.
g=nx.disjoint_union(nx.cycle_graph(4),nx.cycle_graph(4));g.add_edge(1,4)
pos={0:(0,0),1:(1,0),2:(1,1),3:(0,1),4:(3,0),5:(4,0),6:(4,1),7:(3,1)}
fig,axes=plt.subplots(1,2,figsize=(10,3.4))
base={0:0,1:1,2:0,3:1,4:1,5:0,6:1,7:0}
for ax,flip,title in zip(axes,[False,True],['反転前：内部8本、橋0本をcut','右側を全反転：内部8本、橋1本']):
    s={n:(1-v if flip and n>=4 else v) for n,v in base.items()}
    draw(ax,g,pos,[TEAL if s[n]==0 else NAVY for n in g],[(u,v) for u,v in g.edges if s[u]!=s[v]])
    ax.set_title(title,fontsize=15)
    ax.text(2,-.35,'橋',ha='center',fontsize=14)
fig.tight_layout();save(fig,'bridge_flip')

# A degree is an actual edge count; a core number is a surviving minimum degree.
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
g=nx.star_graph(4);pos={0:(0,0),1:(-1,0),2:(1,0),3:(0,1),4:(0,-1)}
draw(axes[0],g,pos,[ORANGE if n==0 else TEAL for n in g]);axes[0].set_title('中心の次数4、core number 1',fontsize=16)
g=nx.complete_graph(4);g.add_edge(0,4);pos={0:(0,1),1:(-1,0),2:(0,-1),3:(1,0),4:(0,2)}
draw(axes[1],g,pos,[ORANGE if n==0 else TEAL for n in g]);axes[1].set_title('上の頂点の次数4、core number 3',fontsize=16)
fig.tight_layout();save(fig,'degree_vs_core')

# Three disjoint vertex categories used in the actual stacked bar chart.
g=nx.cycle_graph(4);g.add_edges_from([(1,4),(4,5),(2,6),(7,8)]);g.add_node(9)
pos={0:(0,0),1:(0,1),2:(1,1),3:(1,0),4:(-1,1.5),5:(-2,1.5),6:(2,1.5),7:(3.5,1.3),8:(4.6,1.3),9:(4,0)}
fig,ax=plt.subplots(figsize=(10,3.4));draw(ax,g,pos,[TEAL if n<4 else GREY if n==9 else ORANGE for n in g])
ax.text(.5,-.6,'緑：繰り返し除いても残る',ha='center',fontsize=15)
ax.text(-1.1,2.15,'橙：木状の枝',ha='center',fontsize=15)
ax.text(4,2.15,'橙：独立した木',ha='center',fontsize=15)
ax.text(4,-.6,'灰：元から孤立点',ha='center',fontsize=15)
ax.set_ylim(-1,2.6);save(fig,'vertex_categories')

# Keep the same vertex colors throughout the peeling example.
fig,axes=plt.subplots(1,3,figsize=(10,3.0))
g=nx.cycle_graph(4);g.add_edges_from([(1,4),(4,5),(2,6)]);g.add_node(7)
pos={0:(0,0),1:(0,1),2:(1,1),3:(1,0),4:(-.8,1.5),5:(-1.6,2),6:(1.8,1.7),7:(2,0)}
for ax,nodes,title in zip(axes,[list(g),[0,1,2,3,4],[0,1,2,3]],['最初のグラフ','1回目：先端と孤立点を除く','2回目：四角形が残る']):
    h=g.subgraph(nodes)
    draw(ax,h,pos,[TEAL if n<4 else GREY if n==7 else ORANGE for n in h])
    ax.set_xlim(-2,2.5);ax.set_ylim(-.5,2.5);ax.set_title(title,fontsize=15)
fig.tight_layout();save(fig,'core_peeling_revision')

# Fresh degree distribution from the actual input, not relabelled core data.
data=json.loads((P/'data/degree_and_cycle.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(10,3.1))
for ax,(name,d) in zip(axes,data.items()):
    h={int(k):v for k,v in d['degree_histogram'].items()}
    ax.bar(list(h),[v/d['vertices']*100 for v in h.values()],color=TEAL,width=.85)
    ax.set_xlim(-1,39);ax.set_ylim(0,32);ax.set_xticks([0,10,20,30]);ax.set_title(name,fontsize=17)
    ax.set_xlabel('次数（隣接する辺の本数）',fontsize=13);ax.tick_params(direction='in',top=True,right=True)
axes[0].set_ylabel('頂点の割合（%）');fig.tight_layout();save(fig,'degree_distribution')

# Previously measured GA results, normalized by the recorded best-known values.
names=['G22','G55','G70'];bks=np.array([13359,10299,9591]);a=np.array([.69,70.75,132.06])/bks*100;b=np.array([.88,87,95.81])/bks*100
fig,ax=plt.subplots(figsize=(8.5,3.4));x=np.arange(3)
ax.bar(x-.18,a,.36,color=TEAL,label='ランダム初期化');ax.bar(x+.18,b,.36,color=ORANGE,label='全CIM初期化')
for xx,ys in [(x-.18,a),(x+.18,b)]:
    for xx0,y in zip(xx,ys):ax.text(xx0,y+.035,f'{y:.3f}',ha='center',fontsize=12)
ax.set_xticks(x,names);ax.set_ylim(0,1.9);ax.set_ylabel('BKSとの差 / BKS（%）');ax.legend(frameon=False,loc='upper left',fontsize=13)
ax.tick_params(direction='in',top=True,right=True);fig.tight_layout();save(fig,'observed_gap')

# Verified odd cycle from G70, keeping the actual node IDs as external labels.
nodes=data['G70']['verified_odd_cycle_vertices'];g=nx.cycle_graph(5);pos=nx.circular_layout(g)
fig,ax=plt.subplots(figsize=(5.5,3.5));draw(ax,g,pos,[TEAL if n%2==0 else NAVY for n in g],[(u,v) for u,v in g.edges if u%2!=v%2])
for n,(x,y) in pos.items():ax.text(x*1.3,y*1.3,str(nodes[n]),ha='center',va='center',fontsize=16)
ax.set_xlim(-1.6,1.6);ax.set_ylim(-1.55,1.55);save(fig,'g70_odd_cycle')

# Connectivity spectrum: same two squares, different connecting-edge weights.
fig,axes=plt.subplots(1,3,figsize=(10,3.0))
vals=[]
for ax,w in zip(axes,[0,.2,1]):
    g=nx.disjoint_union(nx.cycle_graph(4),nx.cycle_graph(4));nx.set_edge_attributes(g,1.0,'weight')
    if w:g.add_edge(1,4,weight=w)
    pos={0:(0,0),1:(1,0),2:(1,1),3:(0,1),4:(2.5,0),5:(3.5,0),6:(3.5,1),7:(2.5,1)}
    A=nx.to_numpy_array(g,nodelist=range(8));deg=A.sum(1);S=A/np.sqrt(deg[:,None]*deg[None,:]);l2=np.linalg.eigvalsh(np.eye(8)-S)[1]
    if abs(l2)<1e-12:l2=0.0
    vals.append(float(l2))
    draw(ax,g,pos,special=[(1,4)] if w else [])
    ax.set_title('接続なし' if w==0 else f'接続辺の重み {w:g}',fontsize=16)
    ax.text(1.75,-.55,f'第2固有値 = {l2:.3f}',ha='center',fontsize=15)
    ax.set_ylim(-1,1.5)
fig.tight_layout();save(fig,'spectral_connectivity')
(P/'data/toy_spectrum.json').write_text(json.dumps({'bridge_weights':[0,.2,1],'normalized_laplacian_lambda2':vals},indent=2))

# Published MQLib single-algorithm selection performance, not our experiment.
fig,ax=plt.subplots(figsize=(7.5,2.6));ax.barh([1,0],[.27,.08],color=[GREY,TEAL],height=.5)
ax.set_yticks([1,0],['全問題でBUR02を使用','構造を見て探索法を選択']);ax.set_xlim(0,.34)
for y,v in [(1,.27),(0,.08)]:ax.text(v+.008,y,f'{v:.2f}%',va='center')
ax.set_xlabel('平均の期待偏差（%）');ax.tick_params(direction='in',top=True,right=True);fig.tight_layout();save(fig,'mqlib_result')
print('Revision figures generated; toy lambda2:',vals)
