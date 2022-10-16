import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import settings
import networkx as nx
from utils import getTorchData
import models.deepwalk as deepwalk
import models.tnodeembedding as tnodeembedding
from models import CTDNE
import time
from gensim.models import KeyedVectors
settings.init()

from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils import shuffle

EMBEDDING_WORKERS=4
NEED_EMBEDDING= True
def staticscore(STATIC_ALGORITHM,YEAR_CURRENT, CENTRALITY): 
    DIRECTORY=f'{settings.FOLDER}{CENTRALITY}/'
    if settings.NAME_DATA=='CORA':
        if settings.SPLIT_NODES:
            DIRECTORY+=f'split/'
        else: DIRECTORY+=f'nosplit/'
    if STATIC_ALGORITHM == "node2vec":
        import torch_geometric.nn as nn
        import torch
        from torch_geometric.utils.convert import from_networkx, to_networkx
        from sklearn.preprocessing import LabelEncoder
        from torch_geometric.data import Data
    print((settings.YEAR_START,YEAR_CURRENT))
    edgesstatic= pd.read_csv(f'{DIRECTORY}edgescumulative/edges{settings.YEAR_START+YEAR_CURRENT}.csv')
    edges1996=edgesstatic[edgesstatic['Year']<settings.YEAR_START+YEAR_CURRENT]
    edges1997=edgesstatic[edgesstatic['Year']==settings.YEAR_START +YEAR_CURRENT]
    nodes=pd.read_csv(f'{DIRECTORY}nodes/nodescomplete.csv')
    G= nx.DiGraph() if settings.DIRECTED else nx.Graph()
    G=nx.from_pandas_edgelist(edgesstatic,source='Source',target='Target',create_using=G,edge_attr='Year')
    if NEED_EMBEDDING:
        start_time=time.process_time()
        if STATIC_ALGORITHM == "deepwalk":
            intostr={x: str(x) for x in list(G.nodes())}
            intostr_inv={str(x): int(x) for x in list(G.nodes())}
            G = nx.relabel_nodes(G, intostr)
            model_i= deepwalk.DeepWalk(G,settings.LENGTH_WALKS,workers=EMBEDDING_WORKERS, num_walks=settings.NUM_WALKS)
            model_i.train(embed_size=settings.DIMENSION, window_size=settings.WINDOWS_SIZE, workers=EMBEDDING_WORKERS)
            model_i=pd.DataFrame.from_dict(model_i.get_embeddings(),orient='index')
            with open(f'./{DIRECTORY}{settings.EMBEDDING_DIR}{STATIC_ALGORITHM}_{settings.NAME_DATA}_{settings.YEAR_START+YEAR_CURRENT}_embeddingsstatic.csv','w+', newline='',encoding='utf-8') as f:
                model_i['id'] = model_i.index
                model_i=model_i.set_index('id')
                model_i.to_csv(f, header=False, index=True,sep=' ')
            
        elif STATIC_ALGORITHM == "node2vec":
            # A node2vec implementation using torch.
            torchG,inv_map=getTorchData(G=G)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = nn.Node2Vec(torchG.edge_index, embedding_dim=settings.DIMENSION, walk_length=settings.LENGTH_WALKS,
                    context_size=settings.WINDOWS_SIZE, walks_per_node=settings.NUM_WALKS,
                    num_negative_samples=1, p=1, q=1,sparse=True).to(device)
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
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
            
            
            embG=pd.DataFrame(model().detach().cpu().numpy())
            print(embG)
            torch.cuda.empty_cache()
            with open(f'./{DIRECTORY}{settings.EMBEDDING_DIR}{STATIC_ALGORITHM}_{settings.NAME_DATA}_{settings.YEAR_START+YEAR_CURRENT}_embeddingsstatic.csv','w+', newline='',encoding='utf-8') as f:
                embG['id'] = embG.index
                embG=embG.replace({"id": inv_map})
                embG=embG.set_index('id')
                embG.to_csv(f, header=False, index=True,sep=' ')
        elif STATIC_ALGORITHM == "tnodeembedding":
            nodescheck=nodes[nodes['Year']<=settings.YEAR_START+1]
            node_dict={row.id:{'label':row.Label} for row in nodescheck.itertuples()}
            graph_nx= nx.DiGraph() if settings.DIRECTED else nx.Graph()
            graph_nx = tnodeembedding.loader.dataset_loader.df2graph(edgesstatic, source='Source', target='Target', time='Year', create_using=graph_nx)
            
            tnodeembed = tnodeembedding.models.tNodeEmbed(graph_nx, task='node_classification', dump_folder=f'{DIRECTORY}embeddings/bin/',time='Year',dimensions=settings.DIMENSION, walk_length= settings.LENGTH_WALKS, num_walks=settings.NUM_WALKS, workers= EMBEDDING_WORKERS)
            graph_nx=tnodeembed.graph_nx
            nodesss=list(graph_nx.nodes(data=True))
            g_dict={x[0]:x[1][YEAR_CURRENT+settings.YEAR_START].tolist() for x in nodesss}
            g_model=pd.DataFrame.from_dict(g_dict,orient='index')
            with open(f'./{DIRECTORY}{settings.EMBEDDING_DIR}{STATIC_ALGORITHM}_{settings.NAME_DATA}_{settings.YEAR_START+YEAR_CURRENT}_embeddingsstatic.csv','w+', newline='',encoding='utf-8') as f:
                g_model.to_csv(f, header=False, index=True,sep=' ')
            os.remove(f'{DIRECTORY}embeddings/bin/init.emb')

            pass #TODO: strano modo di fornire gli embeddings, devo sentire lombardo
           
            
            
        elif STATIC_ALGORITHM == 'ctdne':
            edgesstaticrename= edgesstatic.rename(columns={'Year':'time'})
            Grelabel=nx.Graph()
            Grelabel=nx.from_pandas_edgelist(edgesstaticrename,source='Source',target='Target',create_using=Grelabel,edge_attr='time')
            CTDNE_model = CTDNE(Grelabel, dimensions=settings.DIMENSION, walk_length=settings.LENGTH_WALKS, num_walks=settings.NUM_WALKS, workers=EMBEDDING_WORKERS)
            model = CTDNE_model.fit(window=settings.WINDOWS_SIZE,vector_size=settings.DIMENSION, workers=EMBEDDING_WORKERS).wv
            model.save_word2vec_format(f'./{DIRECTORY}{settings.EMBEDDING_DIR}{STATIC_ALGORITHM}_{settings.NAME_DATA}_{settings.YEAR_START+YEAR_CURRENT}_embeddingsstatic.csv',write_header=False)
            #print(model)
            

            

        print(f'Time taken: {(time.process_time() - start_time):.2f} seconds')
                
                

    #machine learning  
    df= pd.read_csv(f'./{DIRECTORY}{settings.EMBEDDING_DIR}{STATIC_ALGORITHM}_{settings.NAME_DATA}_{settings.YEAR_START+YEAR_CURRENT}_embeddingsstatic.csv',delim_whitespace=True, header=None)
    df= df.sort_values(by=[0]).reset_index(drop=True)
    df.rename(columns = {0:'id'}, inplace = True)
    
    targets= pd.read_csv(f'{DIRECTORY}nodes/nodescomplete.csv')
    targets= targets[targets['Year']<=settings.YEAR_START+YEAR_CURRENT]
    
    df= pd.merge(df,targets,how='left', on='id')
    df.rename(columns = {'Year':129}, inplace = True)
    df=shuffle(df)
    dfstart=df[df[129]<=settings.YEAR_START+YEAR_CURRENT-1]
    dfend=df[df[129]==settings.YEAR_START+YEAR_CURRENT]
    y_train= dfstart.pop('Label')
    y_test=dfend.pop('Label')
    idtrain=dfstart.pop('id')
    idtest=dfend.pop('id')
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(dfstart, y_train)
    #top_k_list = [len(l) for l in y_test]
    y_pred=clf.predict(dfend)
    #prec= precision_score(y_test,y_pred,average=None)
    #recall= recall_score(y_test,y_pred,average=None)
    file=open(f'results{settings.NAME_DATA}.csv','a+',newline='')
    print(f'STATIC ALGORITHM: {STATIC_ALGORITHM}, predictor: Logistic Regression')
    print(f'Train:{dfstart.shape}\nTest:{dfend.shape}')
    averages = ["micro", "macro"]
    """ print(df['Label'].value_counts())
    print(len(df['Label'].value_counts())) """
    for average in averages:
        file.write(f'{settings.YEAR_START+YEAR_CURRENT},{STATIC_ALGORITHM},{average}-F1,{f1_score(y_test,y_pred, average=average)},{dfend.shape[0]},{dfstart.shape[0]},Logistic Regression,{CENTRALITY}\n')
        print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')

    """ print(prec)
    print(recall)"""
    clf = RandomForestClassifier().fit(dfstart, y_train)
    #top_k_list = [len(l) for l in y_test]
    y_pred=clf.predict(dfend)
    #prec= precision_score(y_test,y_pred,average=None)
    #recall= recall_score(y_test,y_pred,average=None)
    print(f'predictor: Random Forest')
    print(f'Train:{dfstart.shape}\nTest:{dfend.shape}')
    averages = ["micro", "macro"]
    """ print(df['Label'].value_counts())
    print(len(df['Label'].value_counts())) """
    for average in averages:
        file.write(f'{settings.YEAR_START+YEAR_CURRENT},{STATIC_ALGORITHM},{average}-F1,{f1_score(y_test,y_pred, average=average)},{dfend.shape[0]},{dfstart.shape[0]},Random Forest,{CENTRALITY}\n')
        print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')
    file.close()
    #print(confusion_matrix(y_test,y_pred))
