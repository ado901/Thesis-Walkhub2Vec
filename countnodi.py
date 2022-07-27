import pandas as pd
import re

lines=[]
nodes=[]
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
print(before-len(edges))
nodesdf=pd.read_csv('nodes/nodes.csv')
print(nodesdf)
before=(len(nodesdf))
print(before)
for node in nodes:
    nodesdf= nodesdf[nodesdf['id']!=int(node)]
print(before-len(nodesdf))
""" import ogb
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name = 'ogbn-arxiv') 
print(dataset[0]) """