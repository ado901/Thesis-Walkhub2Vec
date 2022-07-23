
'''node2vec ha bisogno di una versione di gensim più recente di quella utilizzata da Lombardo'''
#from node2vec import Node2Vec
#from gensim.models import Word2Vec, KeyedVectors

from re import A
import pandas as pd
import settings
settings.init()
from utils import Deepwalk, read_edges_list
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
edgesstatic= pd.read_csv('edges/edges1997.csv')
NEED_EMBEDDING= True
if __name__== '__main__':
    if NEED_EMBEDDING:
        G_model= Deepwalk('edgescumulative/edges1997.csv',settings.DIRECTED,settings.EMBEDDING_DIR,settings.NAME_DATA+"STATIC_G",4,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS,separator=',')
        print('Deepwalk done')  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
        G_model.wv.save_word2vec_format('./embeddingsstatic.csv')
        with open('./embeddingsstatic.csv', 'r') as fin:
            data = fin.read().splitlines(True)
        with open('./embeddingsstatic.csv', 'w') as fout:
            fout.writelines(data[1:])
    df= pd.read_csv("embeddingsstatic.csv",delim_whitespace=True, header=None)
    df= df.sort_values(by=[0]).reset_index(drop=True)
    df.rename(columns = {0:'id'}, inplace = True)

    targets= pd.read_csv('nodes/nodescomplete.csv')
    print(targets)
    df= pd.merge(df,targets,how='left', on='id')
    print(df)

    df1996=df[df['Year']<=1996]
    df1997=df[df['Year']==1997]
    model=OrdinalEncoder()
    model.fit(df[['Year']])
    df1997['Year']=model.transform(df1997[['Year']])
    df1996['Year']=model.transform(df1996[['Year']])
    print(df['Label'].value_counts())
    print(df1997)

    #non è bilanciato ovviamente
    y_train= df1996.pop('Label')
    y_test=df1997.pop('Label')
    idtrain=df1996.pop('id')
    idtest=df1997.pop('id')
    model=LabelEncoder()
    model.fit(df['Label'])
    y_train=model.transform(y_train)
    y_test=model.transform(y_test)
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(df1996, y_train)
    #top_k_list = [len(l) for l in y_test]
    y_pred=clf.predict(df1997)
    #prec= precision_score(y_test,y_pred,average=None)
    #recall= recall_score(y_test,y_pred,average=None)
    averages = ["micro", "macro"]
    for average in averages:
        
        print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')

    """ print(prec)
    print(recall)"""

    print(confusion_matrix(y_test,y_pred))
