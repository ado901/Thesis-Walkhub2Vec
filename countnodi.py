import pandas as pd

df= pd.read_csv('nodes/nodescomplete.csv')
print(len(df[df['Year']<=1996]))

import ogb
from ogb.nodeproppred import PygNodePropPredDataset
dataset = PygNodePropPredDataset(name = 'ogbn-arxiv') 
print(dataset[0])