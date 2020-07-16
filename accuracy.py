import numpy as np
import matplotlib.pyplot as plt
import math

###########entering raw data


ADD = np.array([1,	1	,1	,1	,1,	1])
AgeGroups_1_0  = np.array([	0.151168831,0.946666667,	0.786666667,	0.704571429	,0.368333333	,0.19])
AgeGroups_2_0 = np.array([	0.156883117	,0.886666667,	0.73	,0.688571429,	0.393333333	,0.32])
AnyAllr = np.array([0.918701299,	0.286666667	,0.95,	0.848571429,	0.863333333,	0.79])
AnyPars3 = np.array([	0.929090909	,0.893333333,	0.95,	0.825714286	,0.863333333,	0.74])
CookArea = np.array([0.885974026,	0.78	,0.973333333	,0.828571429	,0.79,	0.69])
DEWORM  = np.array([	0.802597403	,0.96,	0.996666667	,0.822857143,	0.828333333	,0.77])
FamilyGrouped_1_0 = np.array([	0.165714286,	0.8	,0.773333333	,0.757714286	,0.493333333,	0.58])
FamilyGrouped_2_0 = np.array([0.220779221	,0.893333333,	0.986666667,	0.84,	0.68,	0.61 ])
GCAT_1  = np.array([0.788831169	,0.773333333	,0.946666667	,0.770285714,	0.683333333	,0.37])
GCAT_2 = np.array([0.815844156,	0.866666667	,0.57,	0.805714286	,0.433333333,	0.28])
GCOW = np.array([0.652727273	,0.753333333	,0.663333333	,0.724571429	,0.426666667,	0.11])
GDOG_1_0  = np.array([	0.853766234,	0.9	,0.233333333,	0.34,	0.508333333	,0.46])
GDOG_2_0 = np.array([0.161038961,	0.886666667	,0.836666667	,0.725142857	,0.383333333,	0.2])
GELEC_1_0  = np.array([0.45038961	,0.973333333,	0.973333333	,0.988,	0.915,	0.9])
GELEC_2_0  = np.array([0.455324675	,1,	1	,1,	1	,1])
GFLOOR6A_1_0 = np.array([	0.278961039	,0.913333333,	0.14,0.359428571	,0.446666667,	0.39])
GFLOOR6A_2_0 = np.array([0.196363636	,0.926666667,	0.776666667	,0.724	,0.388333333	,0.87])
GFLOOR6A_9_0 = np.array([	0.260779221	,0.946666667	,0.396666667,	0.469142857,	0.408333333	,0.68])
GWASTE_1_0  = np.array([0.236103896,	0.553333333	,0.97	,0.804571429	,0.543333333	,0.65])
GWASTE_2_0  = np.array([0.418441558	,0.96,	0.87,	0.929142857,	0.916666667,	0.47])
GWASTE_3_0  = np.array([0.182857143	,0.706666667	,0.833333333	,0.750857143,	0.38,	0])
HCIGR6A = np.array([	0.930649351	,0.986666667,	0.92,	0.98,	0.911666667	,0.69])
ToiletType_1_0 = np.array([0.414805195	,0.966666667	,0.913333333,	0.98,	0.876666667,	0.76])
ToiletType_2_0 = np.array([0.170649351,	0.906666667	,0.033333333,	0.290285714,	0.206666667,	0.29])
WaterSource_1_0 = np.array([0.152987013	,0.833333333	,0.476666667	,0.758857143,	0.298333333	,0.19])
WaterSource_2_0 = np.array([0.136883117,	0.846666667,	0.076666667	,0.325142857,	0.151666667	,0.26])




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

ADD_std = np.std(ADD)/math.sqrt(6)
AgeGroups_1_0_std=np.std(AgeGroups_1_0)/math.sqrt(6)
AgeGroups_2_0_std=np.std(AgeGroups_2_0)/math.sqrt(6)
AnyAllr_std=np.std(AnyAllr)/math.sqrt(6)
AnyPars3_std=np.std(AnyPars3)/math.sqrt(6)
CookArea_std=np.std(CookArea)/math.sqrt(6)
DEWORM_std=np.std(DEWORM)/math.sqrt(6)
FamilyGrouped_1_0_std=np.std(FamilyGrouped_1_0)/math.sqrt(6)
FamilyGrouped_2_0_std=np.std(FamilyGrouped_2_0)/math.sqrt(6)
GCAT_1_std=np.std(GCAT_1)/math.sqrt(6)
GCAT_2_std=np.std(GCAT_2)/math.sqrt(6)
GCOW_std=np.std(GCOW)/math.sqrt(6)
GDOG_1_0_std=np.std(GDOG_1_0)/math.sqrt(6)
GDOG_2_0_std=np.std(GDOG_2_0)/math.sqrt(6)
GELEC_1_0_std=np.std(GELEC_1_0)/math.sqrt(6)
GELEC_2_0_std=np.std(GELEC_2_0)/math.sqrt(6)
GFLOOR6A_1_0_std=np.std(GFLOOR6A_1_0)/math.sqrt(6)
GFLOOR6A_2_0_std=np.std(GFLOOR6A_2_0)/math.sqrt(6)
GFLOOR6A_9_0_std=np.std(GFLOOR6A_9_0)/math.sqrt(6)
GWASTE_1_0_std=np.std(GWASTE_1_0)/math.sqrt(6)
GWASTE_2_0_std=np.std(GWASTE_2_0)/math.sqrt(6)
GWASTE_3_0_std=np.std(GWASTE_3_0)/math.sqrt(6)
HCIGR6A_std=np.std(HCIGR6A)/math.sqrt(6)
ToiletType_1_0_std=np.std(ToiletType_1_0)/math.sqrt(6)
ToiletType_2_0_std=np.std(ToiletType_2_0)/math.sqrt(6)
WaterSource_1_0_std=np.std(WaterSource_1_0)/math.sqrt(6)
WaterSource_2_0_std=np.std(WaterSource_2_0)/math.sqrt(6)




######### create lists for the plot

Features = ['Place of residence','Cook Area','Any smokers in home','Have cat kept outside','Sometimes use electricity','Never use electricity','Deworming status','Have cat live inside','Toilet type is pit','Any parasites found','Any allergic disease','Dispose waste in open field','Have dog live inside','Have cow','Family size >5','Dispose waste in pit','Age 6-10','Floor type not cement, wood, mud','Toilet is open field','Floor type is wood','Dispose waste by burning','Water source is well','Family size from 4-5','Age 11-15','Floor type is mud','Water source is river/rainwater','Have a dog kept outside']

x_pos = np.arange(len(Features))
CTEs = [ADD_mean,CookArea_mean,HCIGR6A_mean,GCAT_2_mean,GELEC_1_0_mean,GELEC_2_0_mean,DEWORM_mean,GCAT_1_mean,ToiletType_1_0_mean,AnyPars3_mean,AnyAllr_mean,GWASTE_2_0_mean,GDOG_1_0_mean,GCOW_mean,FamilyGrouped_2_0_mean,GWASTE_1_0_mean,AgeGroups_1_0_mean,GFLOOR6A_9_0_mean,ToiletType_2_0_mean,GFLOOR6A_1_0_mean,GWASTE_3_0_mean,WaterSource_1_0_mean,FamilyGrouped_1_0_mean,AgeGroups_2_0_mean,GFLOOR6A_2_0_mean,WaterSource_2_0_mean,GDOG_2_0_mean]
error = [ADD_std,CookArea_std,HCIGR6A_std,GCAT_2_std,GELEC_1_0_std,GELEC_2_0_std,DEWORM_std,GCAT_1_std,ToiletType_1_0_std,AnyPars3_std,AnyAllr_std,GWASTE_2_0_std,GDOG_1_0_std,GCOW_std,FamilyGrouped_2_0_std,GWASTE_1_0_std,AgeGroups_1_0_std,GFLOOR6A_9_0_std,ToiletType_2_0_std,GFLOOR6A_1_0_std,GWASTE_3_0_std,WaterSource_1_0_std,FamilyGrouped_1_0_std,AgeGroups_2_0_std,GFLOOR6A_2_0_std,WaterSource_2_0_std,GDOG_2_0_std]

#########build the plot
fig, ax = plt.subplots()
ax.bar(x_pos,CTEs,yerr = error,align = 'center',alpha = 0.5,color = 'grey',ecolor = 'black',error_kw=dict(lw=1, capsize=2, capthick=1),capsize=4)
ax.set_ylabel('Probability', fontsize =15)
ax.set_xticks(x_pos)
ax.set_xticklabels(Features)
ax.set_title('Accuracy Based',fontsize =30)
ax.yaxis.grid(True)

# Save the figure and show
plt.xticks(rotation =90)
plt.yticks(np.arange(0,1.5,step=0.5),fontsize =15)#

# Save the figure and show
plt.tight_layout()
plt.savefig('accuracy_based_plot_with_error_bars.png')
plt.show()





######## save the figure and show
