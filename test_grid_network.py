import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import hydro_network as hn




G = hn.create_networks(nodes_num = 40)#, g_type= 'grid')
nodelist = list(G)
pos = graphviz_layout(G, prog = 'dot')
nx.draw(G,pos=pos,with_labels=False)
nx.draw_networkx_nodes(G,pos = pos)
# print(G.nodes)
xy = np.asarray([pos[v] for v in nodelist])
print(xy)
ax = plt.axes()
H = nx.grid_2d_graph(4,4)
nx.draw(H,ax=ax)
plt.show()

def my_grid_graph(m,n,create_using=nx.DiGraph()):
    G=nx.empty_graph(0,create_using)
    rows = range(m)
    columns = range(n)
    k = 0
    node_dict = {}
    for i in rows:
        for j in columns:
            node_dict[(i,j)] = k
            k = k + 1
    # print(node_dict)
    G.add_nodes_from( node_dict[(i,j)] for i in rows for j in columns )
    G.add_edges_from( (node_dict[(i,j)],node_dict[(i-1,j)]) for i in rows for j in columns if i>0 )
    G.add_edges_from( (node_dict[(i,j)],node_dict[(i,j-1)]) for i in rows for j in columns if j>0 )
    if G.is_directed():
        G.add_edges_from((v, u) for u, v in G.edges() if (u < v))
    xy = np.asarray([pos[v] for v in nodelist])
    print(xy)
    
    return G
G = my_grid_graph(4,4)
nx.draw(G, pos=pos,with_labels=True)
plt.show()

def grid_2d_graph(m,n,periodic=False,create_using=None):
    """ Return the 2d grid graph of mxn nodes,
        each connected to its nearest neighbors.
        Optional argument periodic=True will connect
        boundary nodes via periodic boundary conditions.
    """
    G=empty_graph(0,create_using)
    G.name="grid_2d_graph"
    rows=range(m)
    columns=range(n)
    G.add_nodes_from( (i,j) for i in rows for j in columns )
    G.add_edges_from( ((i,j),(i-1,j)) for i in rows for j in columns if i>0 )
    G.add_edges_from( ((i,j),(i,j-1)) for i in rows for j in columns if j>0 )
    if G.is_directed():
        G.add_edges_from( ((i,j),(i+1,j)) for i in rows for j in columns if i<m-1 )
        G.add_edges_from( ((i,j),(i,j+1)) for i in rows for j in columns if j<n-1 )
    if periodic:
        if n>2:
            G.add_edges_from( ((i,0),(i,n-1)) for i in rows )
            if G.is_directed():
                G.add_edges_from( ((i,n-1),(i,0)) for i in rows )
        if m>2:
            G.add_edges_from( ((0,j),(m-1,j)) for j in columns )
            if G.is_directed():
                G.add_edges_from( ((m-1,j),(0,j)) for j in columns )
        G.name="periodic_grid_2d_graph(%d,%d)"%(m,n)
    return G