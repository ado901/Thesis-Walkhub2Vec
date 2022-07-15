
'''node2vec ha bisogno di una versione di gensim più recente di quella utilizzata da Lombardo'''
#from node2vec import Node2Vec
#from gensim.models import Word2Vec, KeyedVectors

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score,recall_score,confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
edgesstatic= pd.read_csv('edges/edges1997.csv')
""" G=nx.from_pandas_edgelist(edgesstatic,'Source','Target','Year')

node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs
print('node2vec done')
model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)
print('fit done')
model.wv.save_word2vec_format('./embeddingsstatic.emb') """
df= pd.read_csv("embeddingsstatic.csv",delim_whitespace=True, header=None)
df= df.sort_values(by=[0]).reset_index(drop=True)
df.rename(columns = {0:'id'}, inplace = True)
print(df)
targets= pd.read_csv('nodes/nodes.csv')
print(targets)
34
df= pd.merge(df,targets,how='left', on='id')

print(df['Label'].value_counts())

#non è bilanciato ovviamente
y= df.pop('Label')
id=df.pop('id')
y = LabelEncoder().fit_transform(y)
print(df)
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
