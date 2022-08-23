import pandas as pd
from utils import Deepwalk
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import settings
settings.init()
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
edgesstatic= pd.read_csv(f'edges/edges{settings.YEAR_START+1}.csv')
NEED_EMBEDDING= False
if __name__== '__main__':
    if NEED_EMBEDDING:
            G_model= Deepwalk(f'edgescumulative/edges{settings.YEAR_START+1}.csv',settings.DIRECTED,settings.EMBEDDING_DIR,settings.NAME_DATA+"STATIC_G",4,settings.WINDOWS_SIZE,settings.DIMENSION,settings.NUM_WALKS,settings.LENGTH_WALKS,separator=',')
            print('Deepwalk done')  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
            G_model.wv.save_word2vec_format('./embeddingsstatic.csv')
            with open('./embeddingsstatic.csv', 'r') as fin:
                data = fin.read().splitlines(True)
            with open('./embeddingsstatic.csv', 'w') as fout:
                fout.writelines(data[1:])
    df= pd.read_csv("embeddingsstatic.csv",delim_whitespace=True, header=None)
    df= df.sort_values(by=[0]).reset_index(drop=True)
    df.rename(columns = {0:'id'}, inplace = True)

    targets= pd.read_csv('nodes.csv')
    targets= targets[targets['Year']<=settings.YEAR_START+1]
    df= pd.merge(df,targets,how='left', on='id')
    print(df)
    dfstart=df[df['Year']<=settings.YEAR_START]
    dfend=df[df['Year']>settings.YEAR_START]

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
