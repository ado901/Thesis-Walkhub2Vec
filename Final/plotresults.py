import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme('paper')
cora= pd.read_csv('resultscora.csv')
cora['ANNO'] = cora['ANNO'].apply(str)
sns.relplot(
    data=cora, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()
arxiv=pd.read_csv('resultsarxiv.csv')
arxiv['ANNO'] = arxiv['ANNO'].apply(str)

fig=sns.relplot(
    data=arxiv, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()