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
EMBED_G = False
EMBEDDING_WORKERS= 4

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
    with open(f"{settings.NAME_DATA}_H_edges.csv","w+", newline='') as f:
        dfedges.to_csv(f,header=False, index=False,sep=' ')

    if EMBED_G:
        G_model= Deepwalk(EDGES_LIST_CUMULATIVE,settings.DIRECTED,settings.EMBEDDING_DIR,settings.NAME_DATA+"_G",EMBEDDING_WORKERS,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS,separator=',')
    else: G_model = Word2Vec.load(settings.EMBEDDING_DIR+"bin/"+settings.NAME_DATA+"_G.bin")
    nodes_list=[]
    edges_lists=[]
    #print(G_model[703939])
    #add nodes of the following year
    data1997=pd.read_csv(f'{EDGES_DIR}/edges{settings.YEAR_START+1}.csv')
    data1997=data1997[['Source','Target']]
    nodes_list=data1997['Source'].unique().tolist()
    edges_lists=[]
    #print(G_model.wv.vocab.keys())
    for node in nodes_list:
        edges_list=[]
        listedges=data1997.loc[data1997['Source']==node].values.tolist()
        """ for edge in listedges:
            #TODO devo ricordarmi perchè l'ho fatto
            if edge[1] not in G.nodes() and edge[1] not in nodes_list:
                print(edge)
                listedges.remove(edge) """
        edges_lists.append(listedges)
    #edges_list=data1997[['Source','Target']].values.tolist()
    

    parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,8)
    G_model.wv.save_word2vec_format('./1996model.csv')