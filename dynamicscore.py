import pandas as pd
from gensim.models import Word2Vec
import settings
settings.init()
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

df1996=pd.read_csv('1996model.csv',sep=' ', header=None)
print(df1996.head())
df1996= df1996.sort_values(by=[0]).reset_index(drop=True)
df1997=pd.read_csv('edges_incremental.csv',sep=' ', header=None)
df1997= df1997.sort_values(by=[0]).reset_index(drop=True)
df1996.rename(columns = {0:'id'}, inplace = True)
df1997.rename(columns = {0:'id'}, inplace = True)
targets= pd.read_csv('nodes/nodescomplete.csv')
df1996= pd.merge(df1996,targets,how='left', on='id')
df1997=pd.merge(df1997,targets,how='left', on='id')
years=pd.concat([df1996[["Year"]],df1997[["Year"]]])
model=OrdinalEncoder()
model.fit(years)
df1997['Year']=model.transform(df1997[['Year']])
df1996['Year']=model.transform(df1996[['Year']])
print(df1996)
print(df1997)
print(df1996['Label'].value_counts())
print(df1997['Label'].value_counts())
#non è bilanciato ovviamente
y_train= df1996.pop('Label')
y_test=df1997.pop('Label')
id96=df1996.pop('id')
id97=df1997.pop('id')
y=pd.concat([y_train,y_test])
model=LabelEncoder()
model.fit(y)
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