"""Recompute the decomposition counts and draw the second review's figures.

Run from the repository root. Output belongs to this report version.
"""
from pathlib import Path
import json
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

P = Path(__file__).resolve().parent
TEAL, NAVY, ORANGE, GREY = '#138B86', '#203C55', '#D77932', '#AAB5BF'
plt.rcParams.update({'font.family':'Yu Gothic', 'font.size':14,
                     'axes.unicode_minus':False, 'pdf.fonttype':42,
                     'text.color':NAVY})

records = {}
for name in ['G22','G55','G70']:
    lines = Path('input', name+'.txt').read_text().splitlines()
    n,m = map(int,lines[0].split()); g=nx.Graph();g.add_nodes_from(range(1,n+1))
    for line in lines[1:]:
        if line.strip():
            u,v,w=map(int,line.split());assert w==1;g.add_edge(u,v)
    h=nx.k_core(g,k=2)
    comps=sorted(nx.connected_components(g),key=len,reverse=True)
    main=g.subgraph(comps[0]);trees=[g.subgraph(c) for c in comps[1:] if len(c)>1]
    bridges=list(nx.bridges(g))
    assert all(nx.is_tree(t) for t in trees)
    assert h.number_of_edges()+len(bridges)==m
    records[name]={'core_vertices':len(h),'core_edges':h.number_of_edges(),
        'attached_vertices':len(main)-len(h),'attached_edges':main.number_of_edges()-h.number_of_edges(),
        'separate_tree_components':len(trees),'separate_tree_vertices':sum(len(t) for t in trees),
        'separate_tree_edges':sum(t.number_of_edges() for t in trees),
        'isolates':len(list(nx.isolates(g))),'bridges':len(bridges)}
(P/'data/decomposition.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

def save(fig,name):
    fig.savefig(P/'figures'/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

def line(ax,a,b,c=GREY,w=2):ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=w,zorder=1)
def node(ax,p,c=TEAL):ax.scatter(*p,s=180,color=c,edgecolor='white',linewidth=.7,zorder=3)

# Counts are measured; shapes are representative, not a full graph embedding.
fig,axes=plt.subplots(2,3,figsize=(11,5.5),gridspec_kw={'width_ratios':[1.5,1,0.9]})
for row,name in enumerate(['G55','G70']):
    d=records[name];ax=axes[row,0]
    ax.set_title(name+'：中心と付いた木',fontsize=18,pad=12)
    ax.add_patch(Ellipse((1,1),1.9,1.8,facecolor='#e1f3f0',edgecolor=TEAL,lw=1.5))
    ax.text(1,1,'中心部分 H\n'+f"{d['core_vertices']:,} 頂点\n{d['core_edges']:,} 辺",ha='center',va='center',fontsize=15)
    for a,b in [((1.95,1),(2.25,1.6)),((2.25,1.6),(2.85,1.6)),((2.25,1.6),(2.7,2.05)),((1.65,.35),(2.2,.2))]:
        line(ax,a,b,ORANGE);node(ax,b,ORANGE)
    ax.text(1.5,-.35,f"付いた木：{d['attached_vertices']:,} 頂点\n{d['attached_edges']:,} 辺",ha='center',va='top',fontsize=16)
    ax.set_xlim(-.1,3.3);ax.set_ylim(-1.5,2.5)
    ax=axes[row,1];ax.set_title('離れている木',fontsize=16,pad=12)
    if row:
        for a,b in [((.2,1.2),(.8,1.7)),((.8,1.7),(1.5,1.2)),((.4,.4),(1.2,.4))]:
            line(ax,a,b,ORANGE);node(ax,a,ORANGE);node(ax,b,ORANGE)
        ax.text(1.9,1,'…',fontsize=24)
        ax.text(1,-.35,'243 個の木\n633 頂点 / 390 辺',ha='center',va='top',fontsize=16)
    else:
        ax.text(1,1,'なし',ha='center',va='center',fontsize=20)
        ax.text(1,-.35,'0 頂点 / 0 辺',ha='center',va='top',fontsize=16)
    ax.set_xlim(0,2.3);ax.set_ylim(-1.5,2.5)
    ax=axes[row,2];ax.set_title('孤立点',fontsize=16,pad=12)
    for pt in [(.4,1.6),(1.3,1.6),(.85,.8)]:node(ax,pt,GREY)
    ax.text(1.7,1.1,'…',fontsize=24)
    ax.text(1,-.35,f"{d['isolates']:,} 頂点\n0 辺",ha='center',va='top',fontsize=16)
    ax.set_xlim(0,2.2);ax.set_ylim(-1.5,2.5)
    for ax in axes[row]:ax.axis('off')
fig.tight_layout(h_pad=2.4,w_pad=2);save(fig,'actual_decomposition')

# A disappearing neighbor reduces the current degree: show the full cascade.
fig,axes=plt.subplots(1,3,figsize=(11,3.5))
g=nx.cycle_graph(4);g.add_edges_from([(0,4),(4,5),(1,6)]);g.add_node(7)
pos={0:(0,0),1:(1.2,0),2:(1.2,-1.2),3:(0,-1.2),4:(-.7,.8),5:(-1.45,1.5),6:(1.8,1),7:(2,-1.2)}
for ax,keep,title in zip(axes,[list(g),[0,1,2,3,4],[0,1,2,3]],['① 最初：次数 0・1 を除く','② A も次数 1 になった','③ どの点も次数 2 で止まる']):
    h=g.subgraph(keep)
    for u,v in h.edges:line(ax,pos[u],pos[v])
    for v in h:
        node(ax,pos[v],TEAL if v<4 else GREY if v==7 else ORANGE)
        label=('A：' if v==4 else 'B：' if v==5 else '')+str(h.degree(v))
        ax.text(pos[v][0]+.12,pos[v][1]+.12,label,fontsize=13)
    ax.set_title(title,fontsize=14,pad=13);ax.set_xlim(-1.8,2.7);ax.set_ylim(-1.7,2.1);ax.axis('off');ax.set_aspect('equal')
fig.tight_layout();save(fig,'peeling_degrees')

# Equal initial degree, different survival. Each stage explicitly shows removal.
fig,axes=plt.subplots(2,3,figsize=(11,4.9))
for row in range(2):
    if row==0:
        g=nx.star_graph(4);pos={0:(0,0),1:(-1,0),2:(1,0),3:(0,1),4:(0,-1)}
        sets=[list(g),[0],[]]
        titles=['A：最初は 4 本','葉を除くと 0 本','中心も除かれて空']
    else:
        g=nx.complete_graph(4);g.add_edge(0,4);pos={0:(0,.6),1:(-1,-.2),2:(0,-1),3:(1,-.2),4:(0,1.6)}
        sets=[list(g),[0,1,2,3],[0,1,2,3]]
        titles=['B：最初は 4 本','葉を除いても 3 本ずつ','互いの辺が残り、停止']
    for ax,keep,title in zip(axes[row],sets,titles):
        h=g.subgraph(keep)
        for u,v in h.edges:line(ax,pos[u],pos[v])
        for v in h:node(ax,pos[v],ORANGE if v==0 else TEAL)
        if not keep:ax.text(0,0,'残る点なし',ha='center',va='center',fontsize=16)
        ax.set_title(title,fontsize=16);ax.set_xlim(-1.5,1.5);ax.set_ylim(-1.3,2);ax.axis('off');ax.set_aspect('equal')
fig.tight_layout(h_pad=2);save(fig,'core_survival')
print('Saved decomposition, peeling, and core survival figures.')
