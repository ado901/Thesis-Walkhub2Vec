import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tqdm


def transform_ids(nodes: pd.DataFrame,edges:pd.DataFrame):
    """
    It takes a dataframe of nodes and a dataframe of edges, and transforms the ids in the nodes and
    edges dataframes to integers
    
    :param nodes: a dataframe containing the nodes of the graph
    :type nodes: pd.DataFrame
    :param edges: a pandas dataframe with the columns 'Source' and 'Target'
    :type edges: pd.DataFrame
    """

    model=LabelEncoder()
    model.fit(nodes['id'])
    print(edges)
    print(nodes)
    nodes['id']=model.transform(nodes['id'])
    with open('nodes/nodescomplete.csv','w+', newline='') as f:
        nodes.to_csv(f, sep=',',index=False)
    with open('nodes/nodes.csv','w+', newline='') as f:
        nodes[['id','Label']].to_csv(f, sep=',',index=False)
    edges['Source']=model.transform(edges['Source'])
    edges['Target']=model.transform(edges['Target'])
    print(edges)
    print(nodes)
    
def count_occurrences(nodes:pd.DataFrame):
    """
    It reads in a dataframe of nodes, counts the number of nodes per year, and then prints the
    cumulative sum of the number of nodes per year
    
    :param nodes: a dataframe of nodes
    :type nodes: pd.DataFrame
    """
    print(nodes['Year'].value_counts(ascending=True))
    print(nodes['Year'].value_counts(ascending=True).cumsum())
    edges2011= pd.read_csv('edges/edges2011.csv')
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
        with open('edgescumulative/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgescumulative.to_csv(f1, index=False)
        with open('edges/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgestmp.to_csv(f1, index=False)

def del_inconsistences(edges:pd.DataFrame,nodes:pd.DataFrame):
    """
    It deletes edges that are inconsistent with the nodes
    
    :param edges: a pandas dataframe with the following columns:
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
            if (node.Year>row.Year).bool() or row.Source==row.Target:
                edgestodelete.append(row)
                edges=edges.drop(row.Index,axis='index')
            pbar.update(1)
    print(len(edges))

if __name__ == '__main__':
    edges=pd.read_csv('edgescumulative/edges.csv')
    nodes=pd.read_csv('nodes/nodescomplete.csv')
    yearsunique= nodes['Year'].sort_values().unique()
    transform_ids(nodes,edges)
    create_files(yearsunique,edges)
    del_inconsistences(edges,nodes)
    