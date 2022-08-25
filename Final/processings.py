import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tqdm
import numpy as np
import settings
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
    edges2011= pd.read_csv(f'{settings.DIRECTORY}edgescumulative/{settings.YEAR_START}.csv')
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
        for row in tqdm.tqdm(edges.itertuples(index=True)):
            node= nodes[nodes['id']==row.Target]
            yearnode=node.Year.values[0]
            yearrow=row.Year
            source=row.Source
            target=row.Target
            if (yearnode>yearrow) or source==target:
                edgestodelete.append(row)
                edges=edges.drop(row.Index,axis='index')
            pbar.update(1)
    
    total= len(nodes)
    with tqdm.tqdm(total=total) as pbar:
        for node in tqdm.tqdm(nodes.itertuples(index=True)):
            id= node.id
            year= node.Year
            if node.id not in edges['Source'].unique():
                edges=edges.append({'Source':id,'Target':id,'Year':year},ignore_index=True)
            pbar.update(1)
    print(len(edges))
    return edges

if __name__ == '__main__':
    edges=pd.read_csv(f'{settings.DIRECTORY}edgescumulative/edges.csv')
    nodes=pd.read_csv(f'{settings.DIRECTORY}nodes/nodescomplete.csv')
    yearsunique= nodes['Year'].sort_values().unique()
    
    nodes,edges= transform_ids(nodes,edges)
    edges=del_inconsistences(edges,nodes)
    create_files(yearsunique,edges)
    #count_occurrences(nodes)

    