import pandas as pd
from gensim.models import Word2Vec
import settings
settings.init()
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df1996=pd.read_csv('1996model.csv',sep=' ', header=None)
print(df1996.head())
df1996= df1996.sort_values(by=[0]).reset_index(drop=True)
df1997=pd.read_csv('edges_incremental.csv',sep=' ', header=None)
df1997= df1997.sort_values(by=[0]).reset_index(drop=True)
df=pd.concat([df1996,df1997])
df=df.sort_values(by=[0]).reset_index(drop=True)
df.rename(columns = {0:'id'}, inplace = True)
targets= pd.read_csv('nodes/nodes.csv')
df= pd.merge(df,targets,how='left', on='id')
print(df)
print(df['Label'].value_counts())
#non è bilanciato ovviamente
y= df.pop('Label')
id=df.pop('id')
y = LabelEncoder().fit_transform(y)
X_train, X_test, y_train,y_test= train_test_split(df,y,test_size=0.30)
clf = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
#top_k_list = [len(l) for l in y_test]
y_pred=clf.predict(X_test)
#prec= precision_score(y_test,y_pred,average=None)
#recall= recall_score(y_test,y_pred,average=None)
averages = ["micro", "macro"]
for average in averages:
    
    print(f'{average} F1: {f1_score(y_test,y_pred, average=average)}')

""" print(prec)
print(recall)"""

print(confusion_matrix(y_test,y_pred))