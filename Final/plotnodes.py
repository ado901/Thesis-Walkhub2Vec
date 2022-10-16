import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
targets= pd.read_csv(f'cora/degree/nodes/nodescomplete.csv')
targets= targets[targets['Year']<=1991]
tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
pca = PCA(n_components=50)

fig = plt.figure(figsize=plt.figaspect(0.5))
#fig.set_size_inches(15, 6)
for n,model in enumerate(['walkhubs2vec','deepwalk','ctdne','tnodeembedding']):
    ax = fig.add_subplot(2, 2, n+1)
    if model=='walkhubs2vec':
        dfwalkhubs=pd.read_csv(f'risultati/incremental/cora/degree/embeddings/deepwalk_CORA1991model.csv',sep=' ',header=None)
    else:
        dfwalkhubs=pd.read_csv(f'risultati/incremental/cora/degree/embeddings/{model}_CORA_1991_embeddingsstatic.csv',sep=' ',header=None)

    dfwalkhubs.rename(columns = {0:'id'}, inplace = True)


    dfwalkhubs= pd.merge(dfwalkhubs,targets,how='left', on='id')


    nodeswalkhubs=dfwalkhubs.pop('id')


    labelwalkhubs=dfwalkhubs.pop('Label')
    tsne_resultswalkhubs = tsne.fit_transform(dfwalkhubs)
    
    dfwalkhubs['tsne-2d-one'] = tsne_resultswalkhubs[:,0]
    dfwalkhubs['tsne-2d-two'] = tsne_resultswalkhubs[:,1]
    dfwalkhubs['id']=nodeswalkhubs
    dfwalkhubs['Label']=labelwalkhubs
    handles=[]
    labels=[]
    for labelwalkhubs, color in zip(sorted(dfwalkhubs['Label'].unique()),['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a']):
        dflabel=dfwalkhubs[dfwalkhubs['Label']==labelwalkhubs]
        f=ax.scatter(dflabel['tsne-2d-one'],dflabel['tsne-2d-two'],c=color)
        handles.append(f)
        labels.append(labelwalkhubs)
        ax.set_title(model)
    fig.legend(handles, labels, loc='upper center',ncol=5, labelspacing=0.)
fig.savefig("tsne1991.png", dpi = 300)

#ax[0].legend(df['Label'].unique())

plt.show()
""" 

fig, ax=plt.subplots(1,2)
ax[0].scatter(df['tsne-2d-one'],df['tsne-2d-two'],c='#1b9e77')
df[0]=nodes
plt.show()
for label in df['Label'].unique():
    ax[1].scatter(df[df['label']==label]['tsne-2d-one'],df[df['label']==label]['tsne-2d-two'],label=label) """
""" nodes=pd.read_csv(f'arxiv/nodes/nodescomplete.csv')
sns.histplot(data=nodes, x="Label", hue='Label',legend=False).set(xticklabels=[])
plt.show()
#edges=pd.read_csv(f'cora/edgescumulative/edges.csv')
 """
