import pandas as pd
import re
def delete_nodes():
    lines=[]
    nodes=[]
    with open('tmp/nodetoeliminate.csv','r') as f:
        lines=f.readlines()
        
    with open('tmp/nodetoeliminate.csv','w',newline='') as f:
        for line in lines:
            f.write(line.replace('[[',''))
    with open('tmp/nodetoeliminate.csv', 'r') as f:
        lines= f.readlines()
    for line in lines:
        nodes.append(line.split(' ')[0])
    edges=pd.read_csv('edges/edges1997.csv')
    print(edges)
    before=(len(edges))
    print(before)
    for node in nodes:
        edges= edges[edges['Source']!=int(node)]
        edges=edges[edges['Target']!=int(node)]
    print(before-len(edges))
    edgescumulative=pd.read_csv('edgescumulative/edges1997.csv')
    print(edgescumulative)
    before=(len(edgescumulative))
    print(before)
    for node in nodes:
        edgescumulative= edgescumulative[edgescumulative['Source']!=int(node) ]
        edgescumulative=edgescumulative[edgescumulative['Target']!=int(node)]
    print(before-len(edgescumulative))
    with open('edges/edges1997.csv','w', newline='') as f:
        edges.to_csv(f,sep=',',index=False)
    with open('edgescumulative/edges1997.csv','w', newline='') as f:
        edgescumulative.to_csv(f,sep=',',index=False)
def check_edges_consistency():
    edgescumulative=pd.read_csv('edgescumulative/edges1997.csv',sep=',')
    edges1996cumulative=pd.read_csv('edgescumulative/edges1996.csv',sep=',')
    edges1997=pd.read_csv('edges/edges1997.csv',sep=',')
    edges199697=pd.concat([edges1996cumulative,edges1997],ignore_index=True)
    print(edges199697)
    notpresent=pd.concat([edges199697,edgescumulative],ignore_index=True).drop_duplicates(['Source','Target','Year'],keep=None)
    print(len(notpresent))
    with open('edgescumulative/edges1997.csv','w',newline='') as f:
        edges199697.to_csv(f,sep=',',index=False)

def delete_unconsistent_edges():
    YEAR=1997
    path=f'edges/edges{YEAR}.csv'
    edges=pd.read_csv(path)
    nodescomplete= pd.read_csv(f'nodes/nodescomplete.csv')
    yearsunique= nodescomplete['Year'].sort_values().unique()
    edgestodelete=[]
    for row in edges.itertuples(index=True):
        node= nodescomplete[nodescomplete['id']==row.Target]       
        if (node.Year>row.Year).bool() or (row.Target==row.Source):
            edgestodelete.append(row)
            print(len(edges))
            edges=edges.drop(row.Index,axis='index')
            
    print(len(edgestodelete))
    with open(path, 'w', newline='') as f:
        edges.to_csv(f,sep=',',index=False)
if __name__ == '__main__':
    #delete_unconsistent_edges()
    #delete_nodes()
    check_edges_consistency()
    
