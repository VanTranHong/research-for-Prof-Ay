import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
import math



#### reading the csv file that contains the frequency of features #####
df = pd.read_csv('/Users/vantran/Desktop/PLOTS/All_based.csv')

### number of rows and columns, row the frequency for each feature, columns is the feature selection method
row = df.shape[0]
cols = df.shape[1]


average = []
std = []

##### taking the average and standard error across the feature selection methods  for each feature ##### 
for i in range(row):
    rw = df.iloc[i].tolist()
    rw = [i/10 for i in rw]
    average.append(np.mean(rw))
    std.append(np.std(rw)/math.sqrt(cols))
    
new_ave =[]
new_std =[]
Features = [1,12,13,18, 24,7,2,3,4,5, 6,8,9,10,11,14,15,16,17,19,20,21,22,23,25,26,27]
for i in Features:
    new_ave.append(average[i-1])
    new_std.append(std[i-1])
x_pos = np.arange(len(Features))




fig, ax = plt.subplots()
ax.bar(x_pos,new_ave,yerr = new_std,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xlabel('Risk factor', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(Features)
# ax.grid(False)

ax.yaxis.grid(False)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('All_based_plot_with_error_bars.png')
plt.show()



    
    # average.append()