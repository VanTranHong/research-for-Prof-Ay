import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([1,1])
AgeGroups_1_0  = np.array([1,1,])
AgeGroups_2_0 = np.array([1,1])
AnyAllr = np.array([0,1])
AnyPars3 = np.array([1,2])
# CookArea = np.array([1,	1,	1	,0.36	])
# DEWORM  = np.array([0	,1	,1,	0.18])
# FamilyGrouped_1_0 = np.array([0	,0.84	,0	,0.4])
# FamilyGrouped_2_0 = np.array([0,	0.58,	0.94	,0.24])
# GCAT_1  = np.array([0,	1	,0.7,	0.9	])
# GCAT_2 = np.array([1	,1,	1,	1	])
# GCOW = np.array([0,	1	,0.54,	0.06	])
# GDOG_1_0  = np.array([1	,0.26,	1	,0])
# GDOG_2_0 = np.array([0	,0.44,	0	,0.16	])
# GELEC_1_0  = np.array([1	,0.96	,1,	0.3])
# GELEC_2_0  = np.array([0,	1,	1	,1	,0.455324675	,1,	1	,1,	1	,1])
# GFLOOR6A_1_0 = np.array([1,	0	,1,	0])
# GFLOOR6A_2_0 = np.array([0	,0.42,	0	,0.12])
# GFLOOR6A_9_0 = np.array([1	,0,	1	,0])
# GWASTE_1_0  = np.array([0.54	,0.3,	1	,0.02])
# GWASTE_2_0  = np.array([0,	1,	0.14,	1	])
# GWASTE_3_0  = np.array([0	,0.88	,0.64	,0.4	])
# HCIGR6A = np.array([	1	,0.96,	1	,0.04])
# ToiletType_1_0 = np.array([	0	,1,	1	,1	])
# ToiletType_2_0 = np.array([0.46,	1	,1,	0.9	])
# WaterSource_1_0 = np.array([0	,0.94,	1	,0.36	])
# WaterSource_2_0 = np.array([1,	0.54	,1,	0	])




######## calculate the average
ADD_mean=np.mean(ADD)
AgeGroups_1_0_mean=np.mean(AgeGroups_1_0)
AgeGroups_2_0_mean=np.mean(AgeGroups_2_0)
AnyAllr_mean =np.mean(AnyAllr)
AnyPars3_mean=np.mean(AnyPars3)
CookArea_mean=np.mean(CookArea)
DEWORM_mean=np.mean(DEWORM)
FamilyGrouped_1_0_mean=np.mean(FamilyGrouped_1_0)
FamilyGrouped_2_0_mean=np.mean(FamilyGrouped_2_0)
GCAT_1_mean=np.mean(GCAT_1)
GCAT_2_mean=np.mean(GCAT_2)
GCOW_mean =np.mean(GCOW)
GDOG_1_0_mean=np.mean(GDOG_1_0)
GDOG_2_0_mean=np.mean(GDOG_2_0)
GELEC_1_0_mean=np.mean(GELEC_1_0)
GELEC_2_0_mean=np.mean(GELEC_2_0)
GFLOOR6A_1_0_mean=np.mean(GFLOOR6A_1_0)
GFLOOR6A_2_0_mean=np.mean(GFLOOR6A_2_0)
GFLOOR6A_9_0_mean=np.mean(GFLOOR6A_9_0)
GWASTE_1_0_mean=np.mean(GWASTE_1_0)
GWASTE_2_0_mean=np.mean(GWASTE_2_0)
GWASTE_3_0_mean=np.mean(GWASTE_3_0)
HCIGR6A_mean=np.mean(HCIGR6A)
ToiletType_1_0_mean=np.mean(ToiletType_1_0)
ToiletType_2_0_mean=np.mean(ToiletType_2_0)
WaterSource_1_0_mean=np.mean(WaterSource_1_0)
WaterSource_2_0_mean=np.mean(WaterSource_2_0)





####### calculate the standard deviation

ADD_std = np.std(ADD)/math.sqrt(16)
AgeGroups_1_0_std=np.std(AgeGroups_1_0)/math.sqrt(16)
AgeGroups_2_0_std=np.std(AgeGroups_2_0)/math.sqrt(16)
AnyAllr_std=np.std(AnyAllr)/math.sqrt(16)
AnyPars3_std=np.std(AnyPars3)/math.sqrt(16)
CookArea_std=np.std(CookArea)/math.sqrt(16)
DEWORM_std=np.std(DEWORM)/math.sqrt(16)
FamilyGrouped_1_0_std=np.std(FamilyGrouped_1_0)/math.sqrt(16)
FamilyGrouped_2_0_std=np.std(FamilyGrouped_2_0)/math.sqrt(16)
GCAT_1_std=np.std(GCAT_1)/math.sqrt(16)
GCAT_2_std=np.std(GCAT_2)/math.sqrt(16)
GCOW_std=np.std(GCOW)/math.sqrt(16)
GDOG_1_0_std=np.std(GDOG_1_0)/math.sqrt(16)
GDOG_2_0_std=np.std(GDOG_2_0)/math.sqrt(16)
GELEC_1_0_std=np.std(GELEC_1_0)/math.sqrt(16)
GELEC_2_0_std=np.std(GELEC_2_0)/math.sqrt(16)
GFLOOR6A_1_0_std=np.std(GFLOOR6A_1_0)/math.sqrt(16)
GFLOOR6A_2_0_std=np.std(GFLOOR6A_2_0)/math.sqrt(16)
GFLOOR6A_9_0_std=np.std(GFLOOR6A_9_0)/math.sqrt(16)
GWASTE_1_0_std=np.std(GWASTE_1_0)/math.sqrt(16)
GWASTE_2_0_std=np.std(GWASTE_2_0)/math.sqrt(16)
GWASTE_3_0_std=np.std(GWASTE_3_0)/math.sqrt(16)
HCIGR6A_std=np.std(HCIGR6A)/math.sqrt(16)
ToiletType_1_0_std=np.std(ToiletType_1_0)/math.sqrt(16)
ToiletType_2_0_std=np.std(ToiletType_2_0)/math.sqrt(16)
WaterSource_1_0_std=np.std(WaterSource_1_0)/math.sqrt(16)
WaterSource_2_0_std=np.std(WaterSource_2_0)/math.sqrt(16)




######### create lists for the plot

Features = ['Rf1','Rf2','Rf3','Rf4','Rf5','Rf6','Rf7','Rf8','Rf9','Rf10','Rf11','Rf12','Rf13','Rf14','Rf15','Rf16','Rf17','Rf18','Rf19','Rf20','Rf21','Rf22','Rf23','Rf24','Rf25','Rf26','Rf27']


x_pos = np.arange(len(Features))
CTEs = [ADD_mean,CookArea_mean,HCIGR6A_mean,GCAT_2_mean,GELEC_1_0_mean,GELEC_2_0_mean,DEWORM_mean,GCAT_1_mean,ToiletType_1_0_mean,AnyPars3_mean,AnyAllr_mean,GWASTE_2_0_mean,GDOG_1_0_mean,GCOW_mean,FamilyGrouped_2_0_mean,GWASTE_1_0_mean,AgeGroups_1_0_mean,GFLOOR6A_9_0_mean,ToiletType_2_0_mean,GFLOOR6A_1_0_mean,GWASTE_3_0_mean,WaterSource_1_0_mean,FamilyGrouped_1_0_mean,AgeGroups_2_0_mean,GFLOOR6A_2_0_mean,WaterSource_2_0_mean,GDOG_2_0_mean]
error = [ADD_std,CookArea_std,HCIGR6A_std,GCAT_2_std,GELEC_1_0_std,GELEC_2_0_std,DEWORM_std,GCAT_1_std,ToiletType_1_0_std,AnyPars3_std,AnyAllr_std,GWASTE_2_0_std,GDOG_1_0_std,GCOW_std,FamilyGrouped_2_0_std,GWASTE_1_0_std,AgeGroups_1_0_std,GFLOOR6A_9_0_std,ToiletType_2_0_std,GFLOOR6A_1_0_std,GWASTE_3_0_std,WaterSource_1_0_std,FamilyGrouped_1_0_std,AgeGroups_2_0_std,GFLOOR6A_2_0_std,WaterSource_2_0_std,GDOG_2_0_std]

#########build the plot
#########build the plotpython
fig, ax = plt.subplots()
ax.bar(x_pos,CTEs,yerr = error,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(Features)
ax.set_title('Ranking Based',fontsize =30)
ax.yaxis.grid(True)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#
plt.tight_layout()
plt.savefig('ranking_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
