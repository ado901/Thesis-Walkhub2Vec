import ogb
import numpy as np #
import pandas as pd
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric import utils
import networkx as nx
import settings
settings.init()
from tqdm import tqdm
if __name__ == "__main__":
    dataset = PygNodePropPredDataset(name = 'ogbn-arxiv',root='arxiv/dataset')
    prova=dataset[0].cuda(device=0)
    print(prova)
    print(type(prova.node_year))
    print(f'Number of nodes: {prova.num_nodes}')
    print(f'Number of edges: {prova.num_edges}')
    print(f'Average node degree: {prova.num_edges / prova.num_nodes:.2f}')
    print(f'Contains isolated nodes: {prova.has_isolated_nodes()}')
    print(f'Contains self-loops: {prova.has_self_loops()}')
    print(f'Is undirected: {prova.is_undirected()}')
    nodes={'id':[], 'Year':[], 'Label':[]}
    edges=[]
    #tolist() ritorna n liste da un solo valore
    d={'id':range(0,prova.num_nodes),
        'Year':[x[0] for x in prova.node_year.tolist()],
        'Label':[x[0] for x in prova.y.tolist()]}
    nodesdf= pd.DataFrame(data=d)
    print(nodesdf)
    with open(f'{settings.DIRECTORY}nodes/nodescomplete.csv','w+', newline='')as fnodes:
        nodesdf.to_csv(fnodes,index=False)
    d={'Source':prova.edge_index[0].tolist(),
        'Target':prova.edge_index[1].tolist()}
    nodesdf.rename(columns={'id':'Source'},inplace=True)
    edgesdf=pd.DataFrame(data=d)
    edgesdf=pd.merge(edgesdf,nodesdf[['Source','Year']], how='left',on='Source')
    with open(f'{settings.DIRECTORY}edgescumulative/edges.csv','w+', newline='') as fedges:
        edgesdf.to_csv(fedges,index=False)



