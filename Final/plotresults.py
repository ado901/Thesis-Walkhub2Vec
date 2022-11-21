import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
sns.set_theme('paper')
dataname='ARXIV'
cora= pd.read_csv(f'results{dataname}.csv')
cora['ANNO'] = cora['ANNO'].apply(str)
predictor='Logistic Regression'
predictorstr=''
if predictor=='Random Forest':
    predictorstr='random'
coralogistic=cora[cora['PREDICTOR']==predictor]
typeoftask='split'
coralogistic['%deleted']=[float(x[:-1]) for x in coralogistic['%deleted']]


def degree():
    plt.rc('legend',fontsize=13)
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    fig, ax=plt.subplots(1,1)
    ax.plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax.plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax.plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax.legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax.set_xlabel('Anno')
    ax.set_title('Macro-F1')
    ax.set_ylim([-0.1, 1.1])
    fig.set_size_inches(8, 6)
    fig.savefig(f"{dataname}{typeoftask}{predictorstr}macro-f1.pdf", dpi = 300)
    
    plt.show() 
    fig, ax=plt.subplots(1,1)
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    ax.plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax.plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax.plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax.legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax.set_xlabel('Anno')
    ax.set_title('Micro-F1')
    ax.set_ylim([-0.1, 1.1])
    fig.set_size_inches(8, 6)
    fig.savefig(f"{dataname}{typeoftask}{predictorstr}micro-f1.pdf", dpi = 300)
    plt.show() 
def betweenness():
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    fig, ax=plt.subplots(1,2)
    ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[0].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[0].set_xlabel('Anno')
    ax[0].set_title('Macro-F1')
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[1].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[1].set_xlabel('Anno')
    ax[1].set_title('Micro-F1')
    ax[0].set_ylim([-0.1, 1.1])
    ax[1].set_ylim([-0.1, 1.1])
    fig.suptitle('Betweenness centrality')
    fig.set_size_inches(15, 6)
    fig.savefig("betweenness centrality.png", dpi = 300)
    plt.show()
def eigenvector():
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    fig, ax=plt.subplots(1,2)
    ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[0].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[0].set_xlabel('Anno')
    ax[0].set_title('Macro-F1')
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[1].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[1].set_xlabel('Anno')
    ax[1].set_title('Micro-F1')
    fig.suptitle('Eigenvector centrality')
    fig.set_size_inches(15, 6)
    fig.savefig("eigenvector centrality.png", dpi = 300)
    plt.show()
def pagerank():
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    fig, ax=plt.subplots(1,2)
    ax[0].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[0].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[0].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[0].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[0].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[0].set_xlabel('Anno')
    ax[0].set_title('Macro-F1')
    deepwalk=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='deepwalk')]
    #ctdne=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='ctdne')]
    tnodeembedding=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='tnodeembedding')]
    walkhubs2vec=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    ax[1].plot(deepwalk['ANNO'],deepwalk['VALUE'],'#1b9e77')
    #ax[1].plot(ctdne['ANNO'],ctdne['VALUE'],'#d95f02')
    ax[1].plot(tnodeembedding['ANNO'],tnodeembedding['VALUE'],'#7570b3')
    ax[1].plot(walkhubs2vec['ANNO'],walkhubs2vec['VALUE'],'#e7298a')
    ax[1].legend(['deepwalk','tnodeembed','walkhubs2vec'])
    ax[1].set_xlabel('Anno')
    ax[1].set_title('Micro-F1')
    fig.suptitle('PageRank centrality')
    fig.set_size_inches(15, 6)
    fig.savefig("pagerank centrality.png", dpi = 300)
    plt.show()
def confrontowalkhubs2vec():
    walkhubs2vecdegree=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecbetweenness=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2veceigenvector=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecpagerank=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    fig,ax=plt.subplots(1,1)
    ax.plot(walkhubs2vecdegree['ANNO'],walkhubs2vecdegree['VALUE'],'#1b9e77')
    ax.plot(walkhubs2vecbetweenness['ANNO'],walkhubs2vecbetweenness['VALUE'],'#d95f02')
    ax.plot(walkhubs2veceigenvector['ANNO'],walkhubs2veceigenvector['VALUE'],'#7570b3')
    ax.plot(walkhubs2vecpagerank['ANNO'],walkhubs2vecpagerank['VALUE'],'#e7298a')
    ax.legend(['Degree','Betweenness','Eigenvector','Pagerank'])
    ax.set_xlabel('Anno')
    ax.set_title('Macro-F1')
    fig.set_size_inches(8, 6)
    fig.savefig("centralitiesmacrof1.pdf", dpi = 600)
    plt.show()
    fig,ax=plt.subplots(1,1)
    walkhubs2vecdegree=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecbetweenness=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2veceigenvector=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecpagerank=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='micro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    ax.plot(walkhubs2vecdegree['ANNO'],walkhubs2vecdegree['VALUE'],'#1b9e77')
    ax.plot(walkhubs2vecbetweenness['ANNO'],walkhubs2vecbetweenness['VALUE'],'#d95f02')
    ax.plot(walkhubs2veceigenvector['ANNO'],walkhubs2veceigenvector['VALUE'],'#7570b3')
    ax.plot(walkhubs2vecpagerank['ANNO'],walkhubs2vecpagerank['VALUE'],'#e7298a')
    ax.legend(['Degree','Betweenness','Eigenvector','Pagerank'])
    ax.set_xlabel('Anno')
    ax.set_title('Micro-F1')
    fig.set_size_inches(8, 6)
    fig.savefig("centralitiesmicrof1.pdf", dpi = 600)
    
    plt.show()
def final():
    coraincremental=pd.read_csv(f'risultati/post-fix/incremental/results{dataname}.csv')
    coraincrementalsplit=pd.read_csv(f'risultati/post-fix/incrementalsplit/13years/results{dataname}.csv')
    cora3yearssnapshot=pd.read_csv(f'risultati/post-fix/3yearssnapshot/results{dataname}.csv')

    
    walkhubsincremental=coraincremental[(coraincremental['CENTRALITY']=='degree') &  (coraincremental['SCORE']=='macro-F1') & (coraincremental['ALGORITMO']=='WALKHUBS2VEC') & (coraincremental['PREDICTOR']==predictor)]
    walkhubsincrementalsplit=coraincrementalsplit[(coraincrementalsplit['CENTRALITY']=='degree') &  (coraincrementalsplit['SCORE']=='macro-F1') & (coraincrementalsplit['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalsplit['PREDICTOR']==predictor)]
    walkhubs3yearssnapshot=cora3yearssnapshot[(cora3yearssnapshot['CENTRALITY']=='degree') &  (cora3yearssnapshot['SCORE']=='macro-F1') & (cora3yearssnapshot['ALGORITMO']=='WALKHUBS2VEC') & (cora3yearssnapshot['PREDICTOR']==predictor)]

    fig,ax=plt.subplots(1,1)
    ax.plot(walkhubsincremental['ANNO'],walkhubsincremental['VALUE'],'#d95f02')
    ax.plot(walkhubsincrementalsplit['ANNO'],walkhubsincrementalsplit['VALUE'],'#7570b3')
    ax.plot(walkhubs3yearssnapshot['ANNO'],walkhubs3yearssnapshot['VALUE'],'#e7298a')
    ax.legend(['batch annuali','ingresso su due batch ordinati','embedding statico ogni 3 anni'])
    ax.set_xlabel('Anno')
    ax.set_ylim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_title('Macro-F1')
    fig.set_size_inches(8, 6)
    fig.savefig(f"{dataname}final{predictorstr}macro-f1.pdf", dpi = 300)
    plt.show()
    walkhubsincremental=coraincremental[(coraincremental['CENTRALITY']=='degree') &  (coraincremental['SCORE']=='micro-F1') & (coraincremental['ALGORITMO']=='WALKHUBS2VEC') & (coraincremental['PREDICTOR']==predictor)]
    walkhubsincrementalsplit=coraincrementalsplit[(coraincrementalsplit['CENTRALITY']=='degree') &  (coraincrementalsplit['SCORE']=='micro-F1') & (coraincrementalsplit['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalsplit['PREDICTOR']==predictor)]
    walkhubs3yearssnapshot=cora3yearssnapshot[(cora3yearssnapshot['CENTRALITY']=='degree') &  (cora3yearssnapshot['SCORE']=='micro-F1') & (cora3yearssnapshot['ALGORITMO']=='WALKHUBS2VEC') & (cora3yearssnapshot['PREDICTOR']==predictor)]
    fig,ax=plt.subplots(1,1)
    ax.plot(walkhubsincremental['ANNO'],walkhubsincremental['VALUE'],'#d95f02')
    ax.plot(walkhubsincrementalsplit['ANNO'],walkhubsincrementalsplit['VALUE'],'#7570b3')
    ax.plot(walkhubs3yearssnapshot['ANNO'],walkhubs3yearssnapshot['VALUE'],'#e7298a')
    ax.legend(['batch annuali','ingresso su due batch ordinati','embedding statico ogni 3 anni'])
    ax.set_xlabel('Anno')
    ax.set_title('Micro-F1')
    ax.set_ylim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    fig.set_size_inches(8, 6)
    fig.savefig(f"{dataname}final{predictorstr}micro-f1.pdf", dpi = 300)
    plt.show()
def deletednodes():
    coraincremental=pd.read_csv(f'risultati/post-fix/incremental/results{dataname}.csv')
    coraincrementalsplit=pd.read_csv(f'risultati/post-fix/incrementalsplit/13years/results{dataname}.csv')
    cora3yearssnapshot=pd.read_csv(f'risultati/post-fix/3yearssnapshot/results{dataname}.csv')

    coraincremental['%deleted']=[float(x[:-1]) for x in coraincremental['%deleted']]
    coraincrementalsplit['%deleted']=[float(x[:-1]) for x in coraincrementalsplit['%deleted']]
    cora3yearssnapshot['%deleted']=[float(x[:-1]) for x in cora3yearssnapshot['%deleted']]

    coraincremental=coraincremental[(coraincremental['CENTRALITY']=='degree') &  (coraincremental['SCORE']=='macro-F1') & (coraincremental['ALGORITMO']=='WALKHUBS2VEC') & (coraincremental['PREDICTOR']==predictor)]
    coraincrementalsplit=coraincrementalsplit[(coraincrementalsplit['CENTRALITY']=='degree') &  (coraincrementalsplit['SCORE']=='macro-F1') & (coraincrementalsplit['ALGORITMO']=='WALKHUBS2VEC') & (coraincrementalsplit['PREDICTOR']==predictor)]
    cora3yearssnapshot=cora3yearssnapshot[(cora3yearssnapshot['CENTRALITY']=='degree') &  (cora3yearssnapshot['SCORE']=='macro-F1') & (cora3yearssnapshot['ALGORITMO']=='WALKHUBS2VEC') & (cora3yearssnapshot['PREDICTOR']==predictor)]
    """ walkhubs2vecdegree=coralogistic[(coralogistic['CENTRALITY']=='degree') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecbetweenness=coralogistic[(coralogistic['CENTRALITY']=='betweenness') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2veceigenvector=coralogistic[(coralogistic['CENTRALITY']=='eigenvector') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')]
    walkhubs2vecpagerank=coralogistic[(coralogistic['CENTRALITY']=='pagerank') &  (coralogistic['SCORE']=='macro-F1') & (coralogistic['ALGORITMO']=='WALKHUBS2VEC')] """
    
    coraincremental=coraincremental[['ANNO','%deleted']]
    coraincrementalsplit=coraincrementalsplit[['ANNO','%deleted']]
    cora3yearssnapshot=cora3yearssnapshot[['ANNO','%deleted']]
    #walkhubs2vecpagerank=walkhubs2vecpagerank[['ANNO','%deleted']]

    fig,ax=plt.subplots(1,1)
    ax.plot(coraincremental['ANNO'],coraincremental['%deleted'],'#1b9e77')
    ax.plot(coraincrementalsplit['ANNO'],coraincrementalsplit['%deleted'],'#d95f02')
    ax.plot(cora3yearssnapshot['ANNO'],cora3yearssnapshot['%deleted'],'#7570b3')
    #ax.plot(walkhubs2vecpagerank['ANNO'],walkhubs2vecpagerank['%deleted'],'#e7298a')
    ax.legend(['batch annuali','ingresso su due split ordinati','embedding statico ogni 3 anni'])
    ax.set_xlabel('Anno')
    ax.set_ylabel('% nodi eliminati')
    fig.set_size_inches(8, 6)
    fig.savefig("deletednodesfinal.pdf", dpi = 300)
    plt.show()
def nodesperlabel():
    arxivnodesperlabel=pd.read_csv(f'nodesperlabel.csv')
    fig,ax=plt.subplots(1,1)
    ax.barh(arxivnodesperlabel['Label'],arxivnodesperlabel['Nodes'])
    fig.set_size_inches(8, 6)
    fig.savefig("nodesperlabel.pdf", dpi = 300)
    plt.show()

if __name__ == "__main__":
    #degree()
    #betweenness()
    #confrontowalkhubs2vec()
    #deletednodes()
    final()
    #nodesperlabel()