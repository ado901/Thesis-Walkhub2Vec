import pandas as pd
import tqdm
edges=pd.read_csv('edges.csv')
nodes=pd.read_csv('nodes.csv')
yearsunique= nodes['Year'].sort_values().unique()
if __name__ == '__main__':
    print()
def count_occurrences():
    print(nodes['Year'].value_counts(ascending=True))
    print(nodes['Year'].value_counts(ascending=True).cumsum())
    edges2011= pd.read_csv('edges/edges2011.csv')
    print(len(edges2011['Source'].unique()))

def create_files():
    for i in yearsunique:
        edgescumulative=edges[edges['Year']<=i]
        edgestmp=edges[edges['Year']==i]
        with open('edgescumulative/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgescumulative.to_csv(f1, index=False)
        with open('edges/edges'+str(i)+'.csv','w+', newline='') as f1:
            edgestmp.to_csv(f1, index=False)

def del_inconsistences():
    edgestodelete=[]
    print(len(edges))
    total= len(edges)
    #elimino archi inconsistenti (anno oppure loop)
    with tqdm.tqdm(total=total) as pbar:
        for row in tqdm.tqdm(edges.itertuples(index=True)):
            node= nodes[nodes['id']==row.Target]       
            if (node.Year>row.Year).bool():
                edgestodelete.append(row)
                edges=edges.drop(row.Index,axis='index')
            pbar.update(1)
    print(len(edges))
""" 


edges_incremental=pd.read_csv('edges_incremental.csv', sep= ' ', header=None)
edges2011head=edges2011[~edges2011['Source'].isin(edges_incremental[0])]
print(len(edges2011head))
print(edges2011)
edges2011=pd.merge(edges2011,edges2011head, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
edges=pd.merge(edges,edges2011head, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
edges2011=pd.concat([edges2011head,edges.loc[:]]).reset_index(drop=True)
edges=pd.concat([edges2011head,edges.loc[:]]).reset_index(drop=True)
print(edges2011)
# A dataframe containing all the nodes that are not in the edges dataframe.
nodesoutdegree0=nodes[~nodes['id'].isin(edges['Source'])]
nodesoutdegree0=nodesoutdegree0[['id','Year']]
print(nodesoutdegree0)

for row in nodesoutdegree0.itertuples():
    edges=edges.append(pd.Series({'Source':row.id, 'Target':row.id, 'Year':row.Year}), ignore_index=True)
with open('edges.csv','w+', newline='') as fedges:
    edges.to_csv(fedges,index=False)
#provo a fare da 2016 a 2017
 """