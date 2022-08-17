from tkinter import Y
import pandas as pd
import settings
settings.init()
import networkx as nx
from gensim.models import Word2Vec
from gensim.models.word2vec import Vocab
import os
from utils import read_edges_list,extract_hub_component,Deepwalk,incremental_embedding,export_graph,extract_embedding_for_Hub_nodes,parallel_incremental_embedding
EDGES_DIR_CUMULATIVE='edgescumulative'
EDGES_LIST_CUMULATIVE = f"{EDGES_DIR_CUMULATIVE}/edges{settings.YEAR_START}.csv"
EDGES_DIR='edges'
EDGES_LIST=f"{EDGES_DIR}/edges{settings.YEAR_START}.csv"
EMBED_G = True
EMBEDDING_WORKERS= 6

""" edges= pd.read_csv('edgescumulative/edges.csv')
years= edges['Year'].unique()
for year in years:
    with open(f'edges/edges{year}.csv','w+', newline='') as f:
        edgesyear= edges[edges['Year']==year]
        edgesyear.to_csv(f, index=False) """

if __name__=='__main__':
    G = nx.Graph()
    if settings.DIRECTED:
        G = nx.DiGraph()
    G = read_edges_list(EDGES_LIST_CUMULATIVE,G)
    H = extract_hub_component(G,settings.CUT_THRESHOLD,verbose=True)
   # print(H.nodes())
    """ print(H.has_node(188483))
    print(H.degree(188483)) """

    dfedges=nx.to_pandas_edgelist(H)
    for node in H.nodes():
        if H.degree(node)==0:
            print(f'nodo {node} isolato nel grafo hub')
            dfedges.loc[len(dfedges.index)] = [node,node]
    if not os.path.exists(f"{settings.NAME_DATA}{settings.YEAR_START}_G_edges.csv"):
        for node in G.nodes():
            if G.out_degree(node)==0:
                G.add_edge(node,node)
        edgesG=nx.to_pandas_edgelist(G)
        with open(f"{settings.NAME_DATA}{settings.YEAR_START}_G_edges.csv","w+", newline='') as f:
            edgesG.to_csv(f, index=False,sep=',', header=['Source','Target'])

    with open(f"{settings.NAME_DATA}_H_edges.csv","w+", newline='') as f:
        dfedges.to_csv(f,header=False, index=False,sep=' ')

    if EMBED_G:
        G_model= Deepwalk(f"{settings.NAME_DATA}{settings.YEAR_START}_G_edges.csv",settings.DIRECTED,settings.EMBEDDING_DIR,f"{settings.NAME_DATA}{settings.YEAR_START}_G",EMBEDDING_WORKERS,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS,separator=',')
    else: G_model = Word2Vec.load(f"{settings.EMBEDDING_DIR}bin/{settings.NAME_DATA}{settings.YEAR_START}_G.bin")
    print('embedding ottenuto')
    nodes_list=[]
    edges_lists=[]
    G_model.wv.save_word2vec_format(f'./{settings.YEAR_START}model.csv')
    with open(f'./{settings.YEAR_START}model.csv', 'r') as fin:
            data = fin.read().splitlines(True)
    with open(f'./{settings.YEAR_START}model.csv', 'w') as fout:
            fout.writelines(data[1:])
    #print(G_model[703939])
    #add nodes of the following year
    data2011=pd.read_csv(f'{EDGES_DIR}/edges{settings.YEAR_START+1}.csv')
    data2011=data2011[['Source','Target']]
    nodes_list=data2011['Source'].unique().tolist()
    edges_lists=[]
    #print(G_model.wv.vocab.keys())
    for node in nodes_list:
        edges_list=[]
        listedges=data2011.loc[data2011['Source']==node].values.tolist()
        """ for edge in listedges:
            #TODO devo ricordarmi perchè l'ho fatto
            if edge[1] not in G.nodes() and edge[1] not in nodes_list:
                print(edge)
                listedges.remove(edge) """
        edges_lists.append(listedges)
    #edges_list=data1997[['Source','Target']].values.tolist()
    

    parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,6)
    #salvo il modello per poi mergearlo con quello incremental
    