import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set_theme('paper')
cora= pd.read_csv('resultsCORA.csv')
cora['ANNO'] = cora['ANNO'].apply(str)
corarandom=cora[cora['PREDICTOR']==' Random Forest']
coralogistic=cora[cora['PREDICTOR']==' Logistic Regression']


sns.relplot(
    data=corarandom, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()
sns.relplot(
    data=coralogistic, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()
""" arxiv=pd.read_csv('resultsARXIV.csv')
arxiv['ANNO'] = arxiv['ANNO'].apply(str)
arxivrandom=arxiv[arxiv['PREDICTOR']==' Random Forest']
arxivlogistic=arxiv[arxiv['PREDICTOR']==' Logistic Regression']

fig=sns.relplot(
    data=arxivrandom, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show()
fig=sns.relplot(
    data=arxivlogistic, kind="line",
    x="ANNO", y="VALUE",
    hue="ALGORITMO", size=None, col='SCORE'
)
plt.show() """