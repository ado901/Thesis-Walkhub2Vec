from venv import create
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tqdm
import numpy as np
import settings
import networkx as nx
import os
from utils import read_edges_list_no_file, extract_hub_component
settings.init()

def transform_ids(nodes: pd.DataFrame,edges:pd.DataFrame):
    """
    It reads in a dataframe of nodes, counts the number of nodes per year, and then prints the
    cumulative sum of the number of nodes per year
    
    :param nodes: a dataframe of nodes
    :type nodes: pd.DataFrame
    :param edges: a dataframe of edges
    :type edges: pd.DataFrame
    :return: The nodes and edges dataframes are being returned.
    """
    model=LabelEncoder()
    model.fit(nodes['id'])
    print(edges)
    print(nodes)
    nodes['id']=model.transform(nodes['id'])
    with open(f'{settings.DIRECTORY}nodes/nodescomplete.csv','w+', newline='') as f:
        nodes.to_csv(f, sep=',',index=False)
    with open(f'{settings.DIRECTORY}nodes/nodes.csv','w+', newline='') as f:
        nodes[['id','Label']].to_csv(f, sep=',',index=False)
    edges['Source']=model.transform(edges['Source'])
    edges['Target']=model.transform(edges['Target'])
    with open(f'{settings.DIRECTORY}edgescumulative/edges.csv','w+', newline='') as f:
        edges.to_csv(f, sep=',',index=False)
    print(edges)
    print(nodes)
    return nodes, edges
    
def count_occurrences(nodes:pd.DataFrame):
    """
    It reads in a dataframe of nodes, counts the number of nodes per year, and then prints the
    cumulative sum of the number of nodes per year
    
    :param nodes: a dataframe of nodes
    :type nodes: pd.DataFrame
    """
    print(nodes['Year'].value_counts(ascending=True))
    print(nodes['Year'].value_counts(ascending=True).cumsum())
    edges2011= pd.read_csv(f'{settings.DIRECTORY}edges/edges{settings.YEAR_START}.csv')
    print(len(edges2011['Source'].unique()))
    edges2011= pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges{settings.YEAR_START}.csv')
    print(len(edges2011['Source'].unique()))

def create_files(yearsunique:list,edges:pd.DataFrame):
    """
    For each year in the list of years, create a file with the edges that are cumulative up to that
    year, and another file with the edges that are only in that year
    
    :param yearsunique: list of years in the data
    :type yearsunique: list
    :param edges: the dataframe containing the edges
    :type edges: pd.DataFrame
    """
    with open(f'{settings.DIRECTORY}edgescumulative/edges.csv','w+', newline='') as f:
        edges.to_csv(f, sep=',',index=False)
    for i in yearsunique:
        edgescumulative=edges[edges['Year']<=i]
        edgestmp=edges[edges['Year']==i]
        with open(f'{settings.DIRECTORY}edgescumulative/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgescumulative.to_csv(f1, index=False)
        with open(f'{settings.DIRECTORY}edges/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgestmp.to_csv(f1, index=False)

def del_inconsistences(edges:pd.DataFrame,nodes:pd.DataFrame):
    """
    It deletes edges that are inconsistent with the nodes
    
    :param edges: a dataframe with the edges of the network
    :type edges: pd.DataFrame
    :param nodes: a dataframe with the nodes of the network
    :type nodes: pd.DataFrame
    """
    edgestodelete=[]
    print(len(edges))
    total= len(edges)
    #elimino archi inconsistenti (anno oppure loop)
    with tqdm.tqdm(total=total) as pbar:
        for row in edges.itertuples(index=True):
            node= nodes[nodes['id']==row.Target]
            yearnode=node.Year.values[0]
            yearrow=row.Year
            source=row.Source
            target=row.Target
            if (yearnode>yearrow) or source==target:
                edgestodelete.append(row)
                edges=edges.drop(row.Index,axis='index')
            pbar.update(1)
    print(len(edges))
    return edges
def find_problematic_nodes(edges:pd.DataFrame, year:int):
    edgesyearcumulative=pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges{settings.YEAR_START+1}.csv')
    edgesyear= pd.read_csv(f'{settings.DIRECTORY}edges/edges{settings.YEAR_START+1}.csv')
    gnewyear=nx.DiGraph() if settings.DIRECTED else nx.Graph()
    gnewyear=nx.from_pandas_edgelist(edgesyear,source='Source',target='Target', create_using=gnewyear)
    nodes_year=pd.read_csv(f'{settings.DIRECTORY}nodes/nodescomplete.csv')
    nodes_year=nodes_year[nodes_year['Year']==settings.YEAR_START+1]
    if settings.DIRECTED:
        G=nx.DiGraph()
        H=nx.DiGraph()
    else:
        G=nx.Graph()
        H=nx.Graph()
    G=nx.from_pandas_edgelist(edgesyearcumulative,source='Source',target='Target', create_using=G)
    H = extract_hub_component(nx.from_pandas_edgelist(pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges{settings.YEAR_START}.csv'),source='Source',target='Target',create_using=H),settings.CUT_THRESHOLD,verbose=True)
    if os.path.exists(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv'):
        os.remove(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv')
    filenodetoeliminate= open(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv','a+')
    for node in tqdm.tqdm(list(gnewyear.nodes())):
        if node in nodes_year['id']:
            edges_list= [list(ele) for ele in list(gnewyear.edges(node))]
            check_compatibility(H,G,filenodetoeliminate, node, edges_list)
    filenodetoeliminate.close()
    
def check_compatibility(H,G,filenodetoeliminate, node,edges_list):
    tmp = nx.Graph()
    if settings.DIRECTED:
        tmp = nx.DiGraph()
    tmp = read_edges_list_no_file(edges_list,tmp)
    H_plus_node = H.copy()
    H_init_edges_number = len(H_plus_node.edges())

    
    #da qui in poi ho fatto molte modifiche
    #controlla se il nodo è collegato direttamente con un hub: caso migliore
    for e in tmp.edges():

        if (e[1]) in H.nodes() or (not settings.DIRECTED and e[0] in H.nodes()):
            #if node has a link with someone in Hubs
            H_plus_node.add_edge(e[0],e[1])
            return

    if(H_init_edges_number == len(H_plus_node.edges())):
        #if node has NOT ANY link with someone in Hubs
        found = False
        it=0
        incident_vertexes=[]
        exist=False

        
        while(not found and it<len(tmp.edges())):
            e = list(tmp.edges())[it]
            for incident_vertex in e:
                
                #non ha collegamenti con hub, quindi prova a vedere se c'è una path verso un hub attraverso un nodo vicino
                if incident_vertex != node:
                    if incident_vertex in G.nodes():
                        #vertex linked with node is in G
                        
                        found = True
                        #G.add_edge(e[0],e[1])
                        

                        #scorro la lista degli hub invece di fare una random choice e vedere se ha path
                        for hubtmp in H.nodes():
                            #h_node = random.choice(list(H.nodes()))
                            exist = nx.has_path(G, source=node, target=hubtmp)
                            
                            #se esiste una path va bene e va direttamente allo step successivo
                            if exist:
                                return
                        #se non esiste rimuovo gli archi che stavo analizzando
                        if not exist:
                            incident_vertexes.append(incident_vertex)
                            found=False

            it+=1
        #il nodo non ha archi verso hubs, di conseguenza si prova ad aggiungere un arco verso un hub random
        if not exist and len(incident_vertexes)>0:
            filenodetoeliminate.write(f'{node} non ci sono path verso Hubs\n')
            return
        #teoricamente caso irraggiungibile col nuovo modo
        elif (not found):
            filenodetoeliminate.write(f'{node} non ci sono archi  con nodi esistenti in G \n')
            return

def deletenodes(yearsunique:list,edges:pd.DataFrame):
    '''It takes a list of years and a dataframe of edges, and for each year, it creates a new dataframe of
    edges that excludes the nodes that are incompatible
    
    Parameters
    ----------
    yearsunique : list
        list of years in the dataset
    edges : pd.DataFrame
        the dataframe of edges
    
    '''
    with open(f'{settings.DIRECTORY}tmp/nodetoeliminate.csv','r', newline='',encoding='utf-8') as f:
        rowsnodestoeliminate=f.readlines()
        nodestoeliminate=[int(row.split()[0]) for row in rowsnodestoeliminate]
    print(nodestoeliminate)
    
    print(len(edges))
    edges=edges[~edges['Source'].isin(nodestoeliminate)]
    edges=edges[~edges['Target'].isin(nodestoeliminate)]
    print(len(edges))
    
    for i in yearsunique:
        edgescumulative=edges[edges['Year']<=i]
        edgestmp=edges[edges['Year']==i]
        with open(f'{settings.DIRECTORY}edgescumulative/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgescumulative.to_csv(f1, index=False)
        with open(f'{settings.DIRECTORY}edges/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgestmp.to_csv(f1, index=False)
def check_embeddings():
    emb=pd.read_csv(f'{settings.DIRECTORY}embeddings/{settings.BASE_ALGORITHM}_{settings.NAME_DATA}{settings.YEAR_START}model.csv', delim_whitespace=True)
    print(len(emb))
    data=pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges{settings.YEAR_START}.csv')
    print(len(data['Source'].unique()))
    

if __name__ == '__main__':
    edges=pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges.csv')
    nodes=pd.read_csv(f'{settings.DIRECTORY}nodes/nodescomplete.csv')
    yearsunique= nodes['Year'].sort_values().unique()
    
    nodes,edges= transform_ids(nodes,edges)
    edges=del_inconsistences(edges,nodes)
    create_files(yearsunique,edges)
    #count_occurrences(nodes)
    
    find_problematic_nodes(edges,settings.YEAR_START+1)
    #deletenodes(yearsunique,edges)
    #check_embeddings()

    