from platform import node
import pandas as pd
import settings
settings.init()
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.utils import shuffle
def dynamicScore():
    dfstart=pd.read_csv(f'./{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START+settings.YEAR_CURRENT-1}model.csv',sep=' ', header=None)

    dfend=pd.read_csv(f'{settings.DIRECTORY}{settings.EMBEDDING_DIR}{settings.INCREMENTAL_MODEL}_{settings.BASE_ALGORITHM}_{settings.YEAR_START+settings.YEAR_CURRENT}.csv',sep=' ', header=None)
    dfend= dfend.sort_values(by=[0]).reset_index(drop=True)
    dfstart.rename(columns = {0:'id'}, inplace = True)
    print(dfstart.isna().any())
    dfend.rename(columns = {0:'id'}, inplace = True)
    dfstart=shuffle(dfstart)
    dfend=shuffle(dfend)
    targets= pd.read_csv(f'{settings.DIRECTORY}nodes/nodescomplete.csv')
    dfstart= pd.merge(dfstart,targets,how='left', on='id')
    dfend=pd.merge(dfend,targets,how='left', on='id')
    dfstart.rename(columns = {'Year':129}, inplace = True)
    dfend.rename(columns = {'Year':129}, inplace = True)
    """ dfstart=dfstart.drop(129,axis=1)
    dfend=dfend.drop(129,axis=1) """
    y_train= dfstart.pop('Label')
    y_test=dfend.pop('Label')
    idstart=dfstart.pop('id')
    idend=dfend.pop('id')

    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(dfstart, y_train)
    #top_k_list = [len(l) for l in y_test]
    y_pred=clf.predict(dfend)
    #prec= precision_score(y_test,y_pred,average=None)
    #recall= recall_score(y_test,y_pred,average=None)
    file=open(f'results{settings.NAME_DATA}.csv','a+',newline='')
    print(f'predictor: Logistic Regression')
    print(f'Train:{dfstart.shape}\nTest:{dfend.shape}')
    averages = ["micro", "macro"]
    for average in averages:
        file.write(f'{settings.YEAR_START+settings.YEAR_CURRENT},WALKHUBS2VEC,{average}-F1,{f1_score(y_test,y_pred, average=average)},{dfend.shape[0]},{dfstart.shape[0]},Logistic Regression\n')
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
        file.write(f'{settings.YEAR_START+settings.YEAR_CURRENT},WALKHUBS2VEC,{average}-F1,{f1_score(y_test,y_pred, average=average)},{dfend.shape[0]},{dfstart.shape[0]},Random Forest,{settings.CENTRALITY}\n')
        print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')
    file.close()

#print(confusion_matrix(y_test,y_pred))