import time
import pandas as pd
import deepwalk
import settings
settings.init()
import networkx as nx
import deepwalk
from gensim.models import KeyedVectors
import os
if settings.BASE_ALGORITHM == 'node2vec':
    import torch_geometric.nn as nn
    import torch
from utils import extract_hub_component, getTorchData,parallel_incremental_embedding
EDGES_DIR_CUMULATIVE='edgescumulative'
EDGES_LIST_CUMULATIVE = f"{EDGES_DIR_CUMULATIVE}/edges{settings.YEAR_START}.csv"
EDGES_DIR='edges'
EDGES_LIST=f"{EDGES_DIR}/edges{settings.YEAR_START}.csv"
EMBED_G = False

EMBEDDING_WORKERS= 4

if __name__=='__main__':
    os.makedirs(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}/bin/', exist_ok=True)
    os.makedirs(f'{settings.DIRECTORY}tmp/', exist_ok=True)
    os.makedirs(f'{settings.DIRECTORY}logs/', exist_ok=True)
    G = nx.Graph()
    if settings.DIRECTED:
        G = nx.DiGraph()
    edgestart=pd.read_csv(f"{settings.DIRECTORY}{EDGES_DIR_CUMULATIVE}/edges{settings.YEAR_START}.csv")
    G = nx.from_pandas_edgelist(edgestart,source='Source',target='Target',create_using=G)
    print(f'numeri nodi in G: {len(G.nodes())}')
    
    H = extract_hub_component(G,settings.CUT_THRESHOLD,verbose=True)

    if EMBED_G:
        start_time= time.process_time()
        if settings.BASE_ALGORITHM == "node2vec":
            torchG,inv_map=getTorchData(G=G)
            print(torchG)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = nn.Node2Vec(torchG.edge_index, embedding_dim=settings.DIMENSION, walk_length=settings.LENGTH_WALKS,
                    context_size=settings.WINDOWS_SIZE, walks_per_node=settings.NUM_WALKS,
                    num_negative_samples=1, p=1, q=1, sparse=True).to(device)
            loader = model.loader(batch_size=128, shuffle=True, num_workers=EMBEDDING_WORKERS)
            optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
            def train():
                model.train()
                total_loss = 0
                for pos_rw, neg_rw in loader:
                    optimizer.zero_grad()
                    loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(loader)
            for epoch in range(1, 30):
                loss = train()
                #acc = test()
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            
            embG=pd.DataFrame(model.forward().tolist())
            #save in word2vec format
            with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}modelpreload.csv','w+', newline='',encoding='utf-8') as f:
                f.write(f'{embG.shape[0]} {embG.shape[1]}\n')
                embG['id'] = embG.index
                embG=embG.replace({"id": inv_map})
                embG=embG.set_index('id')
                embG.to_csv(f, header=False, index=True,sep=' ')
                print(f'numeri di nodi in embedding: {embG.shape[0]}')
            G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}modelpreload.csv')
            torch.cuda.empty_cache()
        
            
        else:
            #word2vec inside deepwalk works only with strings
            intostr={x: str(x) for x in list(G.nodes())}
            intostr_inv={x: int(x) for x in list(G.nodes())}
            G = nx.relabel_nodes(G, intostr)
            G_model= deepwalk.DeepWalk(G,settings.LENGTH_WALKS,workers=EMBEDDING_WORKERS, num_walks=settings.NUM_WALKS)
            G_model.train(embed_size=settings.DIMENSION, window_size=settings.WINDOWS_SIZE, workers=EMBEDDING_WORKERS)
            G_model=pd.DataFrame.from_dict(G_model.get_embeddings(),orient='index')
            #save in word2vec format
            with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}modelpreload.csv','w+', newline='',encoding='utf-8') as f:
                f.write(f'{G_model.shape[0]} {G_model.shape[1]}\n')
                G_model['id'] = G_model.index
                G_model=G_model.set_index('id')
                G_model.to_csv(f, header=False, index=True,sep=' ')
                print(f'numeri di nodi in embedding: {G_model.shape[0]}')
            G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}modelpreload.csv')
            G = nx.relabel_nodes(G, intostr_inv)
        print(f'Tempo di calcolo embedding: {(time.process_time() - start_time):.2f} secondi')


    else: G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}modelpreload.csv')
    print('embedding ottenuto')
    nodes_list=[]
    edges_lists=[]
    G_model.save_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv')
    with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv', 'r') as fin:
            data = fin.read().splitlines(True)
    with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv', 'w') as fout:
            fout.writelines(data[1:])
    #add nodes of the following year
    data2011=pd.read_csv(f'{settings.DIRECTORY}{EDGES_DIR}/edges{settings.YEAR_START+1}.csv')
    data2011=data2011[['Source','Target']]
    nodes_list=data2011['Source'].unique().tolist()
    edges_lists=[]
    for node in nodes_list:
        edges_list=[]
        listedges=data2011.loc[data2011['Source']==node].values.tolist()
        edges_lists.append(listedges)
    #edges_list=data1997[['Source','Target']].values.tolist()
    

    parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,EMBEDDING_WORKERS)
    #salvo il modello per poi mergearlo con quello incremental
    