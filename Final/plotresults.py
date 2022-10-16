import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from models import tnodeembedding
sns.set_theme('paper')
""" cora= pd.read_csv('resultsCORA.csv')
cora['ANNO'] = cora['ANNO'].apply(str)
corarandom=cora[cora['PREDICTOR']=='Random Forest']
coralogistic=cora[cora['PREDICTOR']=='Logistic Regression'] """


""" sns.relplot(
    data=corarandom, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show() """
""" sns.relplot(
    data=coralogistic, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show() """
""" arxiv=pd.read_csv('resultsARXIV.csv')
arxiv['ANNO'] = arxiv['ANNO'].apply(str)
arxivrandom=arxiv[arxiv['PREDICTOR']==' Random Forest']
arxivlogistic=arxiv[arxiv['PREDICTOR']==' Logistic Regression']

fig=sns.relplot(
    data=arxivrandom, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()
fig=sns.relplot(
    data=arxivlogistic, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show() """
""" plt.rc('legend',fontsize=13)
deepwalk=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
fig, ax=plt.subplots(1,2)
ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[0].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
deepwalk=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[1].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('CORA degree')
fig.set_size_inches(15, 6)
fig.savefig("cora degree.png", dpi = 300)
plt.show() """

""" deepwalk=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
fig, ax=plt.subplots(1,2)
ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[0].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
deepwalk=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[1].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('Betweenness centrality')
fig.set_size_inches(15, 6)
fig.savefig("betweenness centrality.png", dpi = 300)
plt.show()

deepwalk=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
fig, ax=plt.subplots(1,2)
ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[0].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
deepwalk=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[1].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('Eigenvector centrality')
fig.set_size_inches(15, 6)
fig.savefig("eigenvector centrality.png", dpi = 300)
plt.show()

deepwalk=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
fig, ax=plt.subplots(1,2)
ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[0].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
deepwalk=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
ctdne=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
ax[1].legend(['deepwalk','ctdne','tnodeembed','walkhubs2vec'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('PageRank centrality')
fig.set_size_inches(15, 6)
fig.savefig("pagerank centrality.png", dpi = 300)
plt.show()

walkhubs2vecdegree=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2vecbetweenness=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2veceigenvector=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2vecpagerank=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
fig,ax=plt.subplots(1,2)
ax[0].plot(walkhubs2vecdegree['ANNO'],walkhubs2vecdegree['VALUE'],'#1b9e77')
ax[0].plot(walkhubs2vecbetweenness['ANNO'],walkhubs2vecbetweenness['VALUE'],'#d95f02')
ax[0].plot(walkhubs2veceigenvector['ANNO'],walkhubs2veceigenvector['VALUE'],'#7570b3')
ax[0].plot(walkhubs2vecpagerank['ANNO'],walkhubs2vecpagerank['VALUE'],'#e7298a')
ax[0].legend(['degree','betweenness','eigenvector','PageRank'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
walkhubs2vecdegree=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2vecbetweenness=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2veceigenvector=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
walkhubs2vecpagerank=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
ax[1].plot(walkhubs2vecdegree['ANNO'],walkhubs2vecdegree['VALUE'],'#1b9e77')
ax[1].plot(walkhubs2vecbetweenness['ANNO'],walkhubs2vecbetweenness['VALUE'],'#d95f02')
ax[1].plot(walkhubs2veceigenvector['ANNO'],walkhubs2veceigenvector['VALUE'],'#7570b3')
ax[1].plot(walkhubs2vecpagerank['ANNO'],walkhubs2vecpagerank['VALUE'],'#e7298a')
ax[1].legend(['degree','betweenness','eigenvector','PageRank'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('Centralities with WalkHubs2Vec')
fig.set_size_inches(15, 6)
fig.savefig("centralities.png", dpi = 300)
plt.show() """
cora3stepprefix=pd.read_csv(f'risultati/pre-fix/3stepsincremental/resultsCORA.csv')
coraincrementalprefix=pd.read_csv(f'risultati/pre-fix/incremental/resultsCORA.csv')
coraincrementalpostfix=pd.read_csv(f'risultati/post-fix/incremental/resultsCORA.csv')
coraincrementalsplitpostfix=pd.read_csv(f'risultati/post-fix/incrementalsplit/13years/resultsCORA.csv')

walkhubs3stepprefix=cora3stepprefix[(cora3stepprefix['CENTRALITY']=='degree') &  (cora3stepprefix['SCORE']=='macro-F1') & (cora3stepprefix['ALGORITMO']=='WALKHUBS2VEC') & (cora3stepprefix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalprefix=coraincrementalprefix[(coraincrementalprefix['CENTRALITY']=='degree') &  (coraincrementalprefix['SCORE']=='macro-F1') & (coraincrementalprefix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalprefix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalpostfix=coraincrementalpostfix[(coraincrementalpostfix['CENTRALITY']=='degree') &  (coraincrementalpostfix['SCORE']=='macro-F1') & (coraincrementalpostfix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalpostfix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalsplitpostfix=coraincrementalsplitpostfix[(coraincrementalsplitpostfix['CENTRALITY']=='degree') &  (coraincrementalsplitpostfix['SCORE']=='macro-F1') & (coraincrementalsplitpostfix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalsplitpostfix['PREDICTOR']=='Logistic Regression')]

fig,ax=plt.subplots(1,2)
ax[0].plot(walkhubs3stepprefix['ANNO'],walkhubs3stepprefix['VALUE'],'#1b9e77')
ax[0].plot(walkhubsincrementalprefix['ANNO'],walkhubsincrementalprefix['VALUE'],'#d95f02')
ax[0].plot(walkhubsincrementalpostfix['ANNO'],walkhubsincrementalpostfix['VALUE'],'#7570b3')
ax[0].plot(walkhubsincrementalsplitpostfix['ANNO'],walkhubsincrementalsplitpostfix['VALUE'],'#e7298a')
ax[0].legend(['3steps prefix','normale prefix','normale postfix','split anni postfix'])
ax[0].set_xlabel('Anno')
ax[0].set_title('Macro-F1')
walkhubs3stepprefix=cora3stepprefix[(cora3stepprefix['CENTRALITY']=='degree') &  (cora3stepprefix['SCORE']=='micro-F1') & (cora3stepprefix['ALGORITMO']=='WALKHUBS2VEC') & (cora3stepprefix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalprefix=coraincrementalprefix[(coraincrementalprefix['CENTRALITY']=='degree') &  (coraincrementalprefix['SCORE']=='micro-F1') & (coraincrementalprefix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalprefix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalpostfix=coraincrementalpostfix[(coraincrementalpostfix['CENTRALITY']=='degree') &  (coraincrementalpostfix['SCORE']=='micro-F1') & (coraincrementalpostfix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalpostfix['PREDICTOR']=='Logistic Regression')]
walkhubsincrementalsplitpostfix=coraincrementalsplitpostfix[(coraincrementalsplitpostfix['CENTRALITY']=='degree') &  (coraincrementalsplitpostfix['SCORE']=='micro-F1') & (coraincrementalsplitpostfix['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalsplitpostfix['PREDICTOR']=='Logistic Regression')]
ax[1].plot(walkhubs3stepprefix['ANNO'],walkhubs3stepprefix['VALUE'],'#1b9e77')
ax[1].plot(walkhubsincrementalprefix['ANNO'],walkhubsincrementalprefix['VALUE'],'#d95f02')
ax[1].plot(walkhubsincrementalpostfix['ANNO'],walkhubsincrementalpostfix['VALUE'],'#7570b3')
ax[1].plot(walkhubsincrementalsplitpostfix['ANNO'],walkhubsincrementalsplitpostfix['VALUE'],'#e7298a')
ax[1].legend(['3steps prefix','normale prefix','normale postfix','split anni postfix'])
ax[1].set_xlabel('Anno')
ax[1].set_title('Micro-F1')
fig.suptitle('Confronto post/pre-fix')
fig.set_size_inches(15, 6)
fig.savefig("post-prefix.png", dpi = 300)
plt.show()