"""Concrete examples for the handwritten instructions. Run after other builders."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

P=Path(__file__).resolve().parent
TEAL,NAVY,ORANGE,GREY='#138B86','#203C55','#D77932','#AAB5BF'
plt.rcParams.update({'font.family':'Yu Gothic','font.size':15,'axes.unicode_minus':False,'pdf.fonttype':42,'text.color':NAVY})

def graph(ax,g,pos,s):
    for u,v in g.edges:
        ax.plot([pos[u][0],pos[v][0]],[pos[u][1],pos[v][1]],color=ORANGE if s[u]!=s[v] else GREY,lw=2.6,zorder=1)
    for v in g:
        ax.scatter(*pos[v],s=280,color=TEAL if s[v]==1 else NAVY,edgecolor='white',lw=.8,zorder=3)
    ax.axis('off');ax.set_aspect('equal')

def save(fig,name):
    fig.savefig(P/'figures'/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)

# Name L and R directly on the drawing so the OPT terms have visible referents.
g=nx.disjoint_union(nx.cycle_graph(4),nx.cycle_graph(4));g.add_edge(1,4)
pos={0:(0,0),1:(1,0),2:(1,1),3:(0,1),4:(2.7,0),5:(3.7,0),6:(3.7,1),7:(2.7,1)}
base={0:1,1:-1,2:1,3:-1,4:-1,5:1,6:-1,7:1}
fig,axes=plt.subplots(1,2,figsize=(10,2.8))
for ax,flip in zip(axes,[False,True]):
    s={v:(-t if flip and v>=4 else t) for v,t in base.items()};graph(ax,g,pos,s)
    ax.text(.5,-.45,'左 L：4 点',ha='center',fontsize=16)
    ax.text(3.2,-.45,'右 R：4 点',ha='center',fontsize=16)
    ax.text(1.85,.45,'橋 w = 1',ha='center',fontsize=14)
    ax.set_title('① 橋が未cut：合計 8 点' if not flip else '② 右を全反転：合計 9 点',fontsize=16)
    ax.set_xlim(-.35,4.05);ax.set_ylim(-.7,1.4)
fig.tight_layout();save(fig,'bridge_flip')

# Two triangles share one uncut edge; their losses must not be summed twice.
g=nx.Graph([(0,1),(0,2),(1,2),(0,3),(1,3)])
pos={0:(0,.9),1:(0,-.9),2:(-1.5,0),3:(1.5,0)};s={0:1,1:1,2:-1,3:-1}
fig,ax=plt.subplots(figsize=(5.4,3.1));graph(ax,g,pos,s)
ax.text(.12,0,'共通の\n未cut辺',va='center',fontsize=14)
ax.text(0,1.35,'三角形は 2 個、未cut は 1 本',ha='center',fontsize=16)
ax.set_xlim(-2,2);ax.set_ylim(-1.2,1.7);save(fig,'shared_odd_cycles')

# Immediate gain can be counted from the current assignment without a model.
g=nx.star_graph(4);pos={0:(0,0),1:(-1,0),2:(1,0),3:(0,1),4:(0,-1)}
s={0:1,1:1,2:1,3:1,4:-1}
fig,axes=plt.subplots(1,2,figsize=(8.8,2.8))
for ax,flip in zip(axes,[False,True]):
    t=dict(s)
    if flip:t[0]*=-1
    graph(ax,g,pos,t)
    ax.set_title('反転前：cut 1 本、未cut 3 本' if not flip else '中心だけ反転：cut 3 本',fontsize=15)
    ax.set_xlim(-1.6,1.6);ax.set_ylim(-1.35,1.35)
fig.tight_layout();save(fig,'immediate_gain')

# Small exact checks of the examples, not a solver benchmark.
assert sum(s[u]!=s[v] for u,v in g.edges)==1
t=dict(s);t[0]*=-1;assert sum(t[u]!=t[v] for u,v in g.edges)==3
diamond=nx.Graph([(0,1),(0,2),(1,2),(0,3),(1,3)])
best=max(sum(((mask>>u)&1)!=((mask>>v)&1) for u,v in diamond.edges) for mask in range(16))
assert best==4
(P/'data/instruction_examples.json').write_text(json.dumps({'shared_triangles_edges':5,'shared_triangles_maxcut':4,'immediate_gain_before':1,'immediate_gain_after':3},indent=2))
print('Three instruction-focused figures generated and exact examples checked.')
