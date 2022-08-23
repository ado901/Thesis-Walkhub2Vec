import pandas as pd
from utils import Deepwalk
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import settings
import networkx as nx
from utils import getTorchData

settings.init()
if settings.TORCH:
    import torch_geometric.nn as nn
    import torch
    from torch_geometric.utils.convert import from_networkx, to_networkx
    from sklearn.preprocessing import LabelEncoder
    from torch_geometric.data import Data
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

EMBEDDING_WORKERS=4
NEED_EMBEDDING= True
if __name__== '__main__':
    edgesstatic= pd.read_csv(f'edgescumulative/edges{settings.YEAR_START+1}.csv')
    edges1996=edgesstatic[edgesstatic['Year']<=settings.YEAR_START]
    print(edges1996)
    edges1997=edgesstatic[edgesstatic['Year']>settings.YEAR_START]
    print(edges1997)
    print(edgesstatic)
    if NEED_EMBEDDING:
        if settings.BASE_ALGORITHM == "deepwalk":
            G_model= Deepwalk(f'edgescumulative/edges{settings.YEAR_START+1}.csv',settings.DIRECTED,settings.EMBEDDING_DIR,settings.NAME_DATA+"STATIC_G",4,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS,separator=',')
            print('Deepwalk done')  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
            
        else:
            # A node2vec implementation using torch.
            if settings.TORCH:
                G=nx.from_pandas_edgelist(edgesstatic,source='Source',target='Target',create_using=nx.DiGraph())
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
                for epoch in range(1, 11):
                    loss = train()
                    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
                
                
                embG=pd.DataFrame(model().detach().cpu().numpy())
                print(embG)
                torch.cuda.empty_cache()
                with open(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv','w+', newline='',encoding='utf-8') as f:
                    embG['id'] = embG.index
                    embG=embG.replace({"id": inv_map})
                    embG=embG.set_index('id')
                    embG.to_csv(f, header=False, index=True,sep=' ')
            else:
                G= nx.from_pandas_edgelist(edgesstatic, source='Source', target='Target', create_using=nx.DiGraph())
                node2vec = Node2Vec(G, dimensions=settings.DIMENSION, walk_length=settings.LENGTH_WALKS, num_walks=settings.NUM_WALKS, workers=EMBEDDING_WORKERS)
                G_model = node2vec.fit(window=settings.WINDOWS_SIZE, min_count=1, batch_words=4, workers=EMBEDDING_WORKERS)
                G_model.wv.save_word2vec_format(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv')
                with open(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv', 'r') as fin:
                    data = fin.read().splitlines(True)
                with open(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv', 'w') as fout:
                    fout.writelines(data[1:])
                print('node2vec done')

        
    df= pd.read_csv(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv',delim_whitespace=True, header=None)
    df= df.sort_values(by=[0]).reset_index(drop=True)
    df.rename(columns = {0:'id'}, inplace = True)
    
    targets= pd.read_csv('nodes/nodescomplete.csv')
    targets= targets[targets['Year']<=settings.YEAR_START+1]
    
    df= pd.merge(df,targets,how='left', on='id')
    df.rename(columns = {'Year':129}, inplace = True)
    print(df)
    dfstart=df[df[129]<=settings.YEAR_START]
    print(dfstart)
    dfend=df[df[129]>settings.YEAR_START]
    print(dfend)

    y_train= dfstart.pop('Label')
    y_test=dfend.pop('Label')
    idtrain=dfstart.pop('id')
    idtest=dfend.pop('id')
    model=LabelEncoder()
    model.fit(df['Label'])
    df['Label']=model.transform(df['Label'])
    y_train=model.transform(y_train)
    y_test=model.transform(y_test)
    print(dfstart.dtypes)
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(dfstart, y_train)
    #top_k_list = [len(l) for l in y_test]
    y_pred=clf.predict(dfend)
    #prec= precision_score(y_test,y_pred,average=None)
    #recall= recall_score(y_test,y_pred,average=None)
    averages = ["micro", "macro"]
    print(df['Label'].value_counts())
    print(len(df['Label'].value_counts()))
    print(f'Train shape: {dfstart.shape}\nTest shape: {dfend.shape}')
    for average in averages:
        
        print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')

    """ print(prec)
    print(recall)"""

    print(confusion_matrix(y_test,y_pred))
