import pandas as pd
from utils import Deepwalk
from node2vec import Node2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import settings
import networkx as nx
settings.init()
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

EMBEDDING_WORKERS=4
NEED_EMBEDDING= False
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
            G= nx.from_pandas_edgelist(edgesstatic, source='Source', target='Target', create_using=nx.DiGraph())
            node2vec = Node2Vec(G, dimensions=settings.DIMENSION, walk_length=settings.LENGTH_WALKS, num_walks=settings.NUM_WALKS, workers=EMBEDDING_WORKERS)
            G_model = node2vec.fit(window=settings.WINDOWS_SIZE, min_count=1, batch_words=4, workers=EMBEDDING_WORKERS)
            print('node2vec done')

        G_model.wv.save_word2vec_format(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv')
        with open(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv', 'r') as fin:
            data = fin.read().splitlines(True)
        with open(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv', 'w') as fout:
            fout.writelines(data[1:])
    df= pd.read_csv(f'./{settings.BASE_ALGORITHM}_{settings.NAME_DATA}_embeddingsstatic.csv',delim_whitespace=True, header=None)
    df= df.sort_values(by=[0]).reset_index(drop=True)
    df.rename(columns = {0:'id'}, inplace = True)
    
    targets= pd.read_csv('nodes/nodescomplete.csv')
    targets= targets[targets['Year']<=settings.YEAR_START+1]
    df= pd.merge(df,targets,how='left', on='id')
    print(df)
    dfstart=df[df['Year']<=settings.YEAR_START]
    print(dfstart)
    dfend=df[df['Year']>settings.YEAR_START]
    print(dfend)

    y_train= dfstart.pop('Label')
    y_test=dfend.pop('Label')
    idtrain=dfstart.pop('id')
    idtest=dfend.pop('id')
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
