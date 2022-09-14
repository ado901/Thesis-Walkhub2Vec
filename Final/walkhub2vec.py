import time
import pandas as pd
import settings
settings.init()
import networkx as nx
import models.deepwalk as deepwalk
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
EMBED_G = True
START= settings.YEAR_START+settings.YEAR_CURRENT-1
JOIN=settings.YEAR_START+settings.YEAR_CURRENT
EMBEDDING_WORKERS= 25

if __name__=='__main__':
    os.makedirs(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}/bin/', exist_ok=True)
    os.makedirs(f'{settings.DIRECTORY}tmp/', exist_ok=True)
    os.makedirs(f'{settings.DIRECTORY}logs/', exist_ok=True)
    G = nx.Graph()
    print(f'actual year: {JOIN}')
    if settings.DIRECTED:
        G = nx.DiGraph()
    edgestart=pd.read_csv(f"{settings.DIRECTORY}{EDGES_DIR_CUMULATIVE}/edges{START}.csv")
    G = nx.from_pandas_edgelist(edgestart,source='Source',target='Target',create_using=G)
    edgeyearplusone=pd.read_csv(f"{settings.DIRECTORY}edges/edges{JOIN}.csv")
    #aggiunta nodi negli anni precedenti che vengono citati
    nodes=pd.read_csv(f"{settings.DIRECTORY}nodes/nodescomplete.csv")
    nodes=nodes[nodes['Year']<=START]
    for target in edgeyearplusone.Target.values:
        if target not in G.nodes() and target in nodes.id.values:
            year=edgeyearplusone[edgeyearplusone.Target==target].Year.values[0]
            edgestart=pd.concat([edgestart,pd.DataFrame({"Source":[target],"Target":[target], 'Year':[year]})])
            G = nx.from_pandas_edgelist(edgestart,source='Source',target='Target',create_using=G)
    print(f'numeri nodi in G: {len(G.nodes())}')

    H = extract_hub_component(G,settings.CUT_THRESHOLD,verbose=True)

    if EMBED_G and settings.YEAR_START==START:
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
            with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}modelpreload.csv','w+', newline='',encoding='utf-8') as f:
                embG['id'] = embG.index
                embG=embG.replace({"id": inv_map})
                embG=embG.set_index('id')
                embG.to_csv(f, header=False, index=True,sep=' ')
                print(f'numeri di nodi in embedding: {embG.shape[0]}')
            G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}modelpreload.csv', no_header=True, binary=False)
            torch.cuda.empty_cache()
        
            
        elif settings.BASE_ALGORITHM == "deepwalk":

            #word2vec inside deepwalk works only with strings
            intostr={x: str(x) for x in list(G.nodes())}
            intostr_inv={x: int(x) for x in list(G.nodes())}
            G = nx.relabel_nodes(G, intostr)
            G_model= deepwalk.DeepWalk(G,settings.LENGTH_WALKS,workers=EMBEDDING_WORKERS, num_walks=settings.NUM_WALKS)
            G_model.train(embed_size=settings.DIMENSION, window_size=settings.WINDOWS_SIZE, workers=EMBEDDING_WORKERS)
            G_model=pd.DataFrame.from_dict(G_model.get_embeddings(),orient='index')
            #save in word2vec format
            with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}model.csv','w+', newline='',encoding='utf-8') as f:
                G_model['id'] = G_model.index
                G_model=G_model.set_index('id')
                G_model.to_csv(f, header=False, index=True,sep=' ')
                print(f'numeri di nodi in embedding: {G_model.shape[0]}')
            G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}model.csv',no_header=True)
            G = nx.relabel_nodes(G, intostr_inv)
        print(f'Tempo di calcolo embedding: {(time.process_time() - start_time):.2f} secondi')       


    else:
        if settings.YEAR_CURRENT==1:
            G_model = KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}model.csv', no_header=True)
        else:
            dfembeddings=pd.read_csv(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv', header=None,delim_whitespace=True)
            for i in range (1,settings.YEAR_CURRENT):
                dfyear=pd.read_csv(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.NAME_DATA}_incremental_{settings.BASE_ALGORITHM}_{settings.YEAR_START+i}.csv', header=None,delim_whitespace=True)
                dfembeddings=pd.concat([dfembeddings,dfyear],ignore_index=True)
            with open(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}model.csv', 'w+',newline='') as f:
                dfembeddings.to_csv(f,header=False, index=False,sep=' ')
            G_model=KeyedVectors.load_word2vec_format(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{START}model.csv',no_header=True)
    #add nodes of the following year
    data2011=pd.read_csv(f'{settings.DIRECTORY}{EDGES_DIR}/edges{JOIN}.csv')
    gnewyear=nx.DiGraph() if settings.DIRECTED else nx.Graph()
    gnewyear=nx.from_pandas_edgelist(data2011, source='Source', target='Target',create_using=gnewyear)
    data2011=data2011[['Source','Target']]
    nodes_list=[]
    nodes_year=pd.read_csv(f'{settings.DIRECTORY}nodes/nodescomplete.csv')
    nodes_year=nodes_year[nodes_year['Year']==JOIN]
    edges_lists=[]
    for node in list(gnewyear.nodes()):
        if node in nodes_year['id'].values:
            nodes_list.append(node)
            edges_lists.append([list(ele) for ele in list(gnewyear.edges(node))])
    
    #edges_list=data1997[['Source','Target']].values.tolist()
    print(f'numeri di nodi entranti: {len(nodes_list)}')
    parallel_incremental_embedding(nodes_list,edges_lists,H,G,G_model,EMBEDDING_WORKERS)
    #salvo il modello per poi mergearlo con quello incremental
    