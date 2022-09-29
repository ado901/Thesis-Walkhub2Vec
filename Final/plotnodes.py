import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from collections import Counter
nodes=pd.read_csv(f'arxiv/nodes/nodescomplete.csv')
sns.histplot(data=nodes, x="Label", hue='Label',legend=False).set(xticklabels=[])
plt.show()
#edges=pd.read_csv(f'cora/edgescumulative/edges.csv')
