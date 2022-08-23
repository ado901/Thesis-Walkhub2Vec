from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
nodes= pd.read_csv('nodes/nodes.csv')
edges=pd.read_csv('edgescumulative/edges.csv')
totedges= len(edges)
totnodes=len(nodes)
#print(edges.groupby(['Year']).size().cumsum())

nodescomplete= pd.read_csv('nodes/nodescomplete.csv')
yearsunique= nodescomplete['Year'].sort_values().unique()
print(edges)
edgestmp= pd.read_csv('edgescumulative/edges.csv')
countnodes=nodescomplete[~nodescomplete['id'].isin(edgestmp['Target'].tolist())]
print(countnodes) #2265 nodi non presenti come target
print(len(countnodes.index))
countnodes=nodescomplete[nodescomplete['Year']>1900]
countnodes=countnodes[~countnodes['id'].isin(edgestmp['Source'].tolist())]
print(countnodes) #1423 nodi non presenti come Source
print(len(countnodes.index))
""" edgesyearcumulative=None
for year, i in enumerate(yearsunique):
    edgesyear= edgestmp[edgestmp['Year']==year]
    edgesyear=edgesyear[nodescomplete['id'].isin(edgesyear['Target'])]
    edgesyearcumulative = edgesyearcumulative.append(edgesyear, ignore_index=True)

with open('prova.csv','w+', newline='') as f1:
        edgesyearcumulative.to_csv(f1, index=False) """
        

""" for i in yearsunique:
    print(edges[edges['Year']<=i])
    edgestmp=edges[edges['Year']<=i]
    with open('edges'+str(i)+'.csv','w+', newline='') as f1:
        edgestmp.to_csv(f1, index=False) """

#print(nodescomplete.groupby(['Year']).size().cumsum())
#edges=nx.from_pandas_edgelist(edges,'Source','Target','Year')
