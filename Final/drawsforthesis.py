import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.cm as cm


def draw(G, pos, measures, measure_name):
    
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250, cmap=plt.cm.plasma, 
                                   node_color=list(measures.values()),
                                   nodelist=measures.keys())
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1, base=10))
    # labels = nx.draw_networkx_labels(G, pos)
    edges = nx.draw_networkx_edges(G, pos)

    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis('off')
    plt.show()
G=nx.powerlaw_cluster_graph(10,2,0.5)
pos = nx.spring_layout(G, seed=675)
draw(G,pos, nx.degree_centrality(G), 'Degree Centrality')
draw(G, pos, nx.pagerank(G), 'PageRank Centrality')
draw(G, pos, nx.betweenness_centrality(G), 'Betweenness Centrality')
draw(G, pos, nx.eigenvector_centrality(G), 'Eigenvector Centrality')