import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


######## reading csv file to draw the heapmap of accuracy or F1 score either for no boost or bag, boost or bag
df = pd.read_csv('/Users/vantran/Desktop/PLOTS/accuracy_noboostbag.csv')
result = df.pivot(index = 'Feature Selection',columns = 'Classifier', values = 'Accuracy')

fig,ax = plt.subplots(figsize=(12,7))
Feature_Selection = ['FULL','IG-10','ReF-10','IG-20','ReF-20','CFS','FCBF','MRMR','SFFS']
Columns = ['KNN','NB','LR','SVM','RF','XGB']
result = result.reindex(index = Feature_Selection,columns = Columns)
# plt.title(title,fontsize = 30)
plt.xlabel(" Feature Selection", fontsize = 25)
plt.ylabel(" Classifier", fontsize = 25)
# ttl = ax.title
# ttl.set_position([0.5,1.05])
ax.set_xticks([])
ax.set_yticks([])
ax.set






res = sns.heatmap(result,fmt ='.0%',cmap ='RdYlGn',cbar=False,annot = True,annot_kws={"size": 16},linewidths=0.30,ax=ax )
res.set_xticklabels(res.get_xmajorticklabels(),fontsize =20)
res.set_yticklabels(res.get_ymajorticklabels(),fontsize =20,rotation =45)
# res.colorbar(False)
# colorbar  = res.collections[0].colorbar
# colorbar.ax.locator_params(nbins =4)
# colorbar.ax.tick_params(labelsize = 15)


sns.clustermap(result, cbar=True,annot = True,annot_kws={"size": 16},linewidths=0.30)
plt.savefig('cluster.png')
plt.savefig('heatmap_Accuracynoboostbag.png')
plt.show()

#RdYlGn
