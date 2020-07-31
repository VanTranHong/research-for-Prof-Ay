import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

df = pd.read_csv('/Users/vantran/Desktop/PLOTS/heatmapfeatures.csv')
result = df.pivot('Feature Selection','Risk Factor', 'Probability')


fig,ax = plt.subplots(figsize=(12,7))

Features = ['KNN-F1','KNN-Ac','SVM-F1','SVM-Ac','RF-F1','RF-Ac','NB-F1','NB-Ac','LR-F1','LR-Ac','XGB-F1','XGB-Ac','IG-20','ReF-20','IG-10','ReF-10','CFS','MRMR','FCBF' ]
Columns = [1,12,13,18, 24,7,2,3,4,5, 6,8,9,10,11,14,15,16,17,19,20,21,22,23,25,26,27]
result = result.reindex(index = Features,columns = Columns)
# plt.title(title,fontsize = 30)
# df['Feature Selection']=pd.Categorical(df['Feature Selection'], categories=Features,ordered=True)
plt.xlabel("Risk Factor", fontsize = 20)
plt.ylabel("Feature Selection", fontsize = 20)
# ttl = ax.title
# ttl.set_position([0.5,1.05])
ax.set_xticks([])
ax.set_yticks([])
ax.set


# res = sns.heatmap(pd.crosstab(df['Feature Selection'],df['Feature']),cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax)

res = sns.heatmap(result,cmap ='RdYlGn',annot = False,linewidths=0.30,ax=ax )#annot_kws={"size": 10},fmt ='.2%'
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =15)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize =15,rotation =45)
colorbar  = res.collections[0].colorbar
colorbar.ax.locator_params(nbins =3)

plt.savefig('feature.png')
plt.show()

#RdYlGn
