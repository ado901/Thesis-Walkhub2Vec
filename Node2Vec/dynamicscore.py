import pandas as pd
import settings
settings.init()
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
dfstart=pd.read_csv(f'{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv',sep=' ', header=None)
print(len(dfstart))
dfend=pd.read_csv(f'{settings.INCREMENTAL_MODEL}_{settings.BASE_ALGORITHM}_{settings.YEAR_START+1}.csv',sep=' ', header=None)
dfend= dfend.sort_values(by=[0]).reset_index(drop=True)
dfstart.rename(columns = {0:'id'}, inplace = True)
dfend.rename(columns = {0:'id'}, inplace = True)
targets= pd.read_csv('nodes/nodescomplete.csv')
dfstart= pd.merge(dfstart,targets,how='left', on='id')
print(dfstart)
dfend=pd.merge(dfend,targets,how='left', on='id')
y_train= dfstart.pop('Label')
y_test=dfend.pop('Label')
idstart=dfstart.pop('id')
idend=dfend.pop('id')
clf = OneVsRestClassifier(LogisticRegression(solver='liblinear')).fit(dfstart, y_train)
#top_k_list = [len(l) for l in y_test]
y_pred=clf.predict(dfend)
#prec= precision_score(y_test,y_pred,average=None)
#recall= recall_score(y_test,y_pred,average=None)
averages = ["micro", "macro"]
print(f'Train shape: {dfstart.shape}\nTest shape: {dfend.shape}')
for average in averages:
    
    print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')

""" print(prec)
print(recall)"""

print(confusion_matrix(y_test,y_pred))